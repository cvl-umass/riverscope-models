# Adopted from https://github.com/pytorch/examples/blob/main/imagenet/main.py#L393
# NOTE: this is for using multiple GPUs for training

import os
from enum import Enum
import torch
import fnmatch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
import torch.distributed as dist
import argparse
import torch.utils.data.sampler as sampler
import shutil
from models.get_model import get_model
from utils_dir.evaluation_dist import BinMaskMeter
import numpy as np
from progress.bar import Bar as Bar
from utils_dir import Logger, AverageMeter, mkdir_p
from utils_dir.dense_losses import get_dense_tasks_losses
from dataset.sentinel2_landcover import Sentinel2Data

from loguru import logger as lgr
from datetime import datetime


DATE_STR = datetime.now().strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser(description='Baselines (SegNet)')
parser.add_argument('--data_dir', default='/datasets/ai/allenai/satlas_pretrain/sentinel2', type=str, help='Path to dataset')
parser.add_argument('--loss_type', default='bce', type=str, help="Type of loss for training. choices: [adaptive_maxpool, bce]")
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
parser.add_argument('--find_unused_param', default=1, type=int, help='Find unused param for distrib training')
parser.add_argument('--scheduler', default="none", type=str, help='Scheduler to use (if any) - choices: "none", "steplr"')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate to use')

parser.add_argument('--ckpt_path', default=None, type=str, help='specify location of checkpoint')

parser.add_argument('--weight', default='uniform', type=str, help='multi-task weighting: uniform, gradnorm, mgda, uncert, dwa, gs')
parser.add_argument('--backbone', default='resnet50', type=str, help='shared backbone')
parser.add_argument('--head', default='no_head', type=str, help='task-specific decoder')
parser.add_argument('--segment_model', default="unet", type=str, help='task-specific decoder. choices: [deeplabv3, deeplabv3plus, unet, fpn, dpt]')
parser.add_argument('--adaptor', default="linear", type=str, help='Type of adaptor to convert 4-chan to 3-channels. choices: [linear, drop, no_init]. NOTE: other options not supported for sentinel data')
parser.add_argument('--resize_size', default=512, type=int, help='Size of the input image')

parser.add_argument('--tasks', default=["water_mask"], nargs='+', help='Task(s) to be trained')
parser.add_argument('--method', default='vanilla', type=str, help='vanilla or mtan')
parser.add_argument('--out', default='./results/satlas-sentinel2-stl', help='Directory to output the result')

# The following params are for multiple GPU training
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument("--pretrained", default=1, type=int, help="Set to 1 to use pretrained model. 0 otherwise.")
parser.add_argument("--is_sampled", default=0, type=int, help="Set to 1 to use sampled data (50k/10k/10k). 0 otherwise.")
parser.add_argument("--is_finetune", default=0, type=int, help="Set to 1 to finetune (means the start_epoch won't be based on ckpt_path). 0 otherwise.")



def save_checkpoint(state, is_best, opt, filename='checkpoint.pth.tar'):
    filepath = os.path.join(opt.out, '{}--{}--{}--{}--{}--mtl_baselines_{}_{}_'.format(DATE_STR, opt.segment_model, opt.adaptor, opt.backbone, opt.head, opt.method, opt.weight) + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(opt.out, '{}--{}--{}--{}--{}--mtl_baselines_{}_{}_'.format(DATE_STR, opt.segment_model, opt.adaptor, opt.backbone, opt.head, opt.method, opt.weight) + 'model_best.pth.tar'))

# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    opt = parser.parse_args()
    lgr.debug(f"opt: {opt}")


    if not os.path.isdir(opt.out):
        mkdir_p(opt.out)

    if opt.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    opt.distributed = opt.world_size > 1 or opt.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        opt.world_size = ngpus_per_node * opt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        # Simply call main_worker function
        main_worker(opt.gpu, ngpus_per_node, opt)
    


# best_loss = 100000
best_perf = -1
def main_worker(gpu, ngpus_per_node, opt):
    global best_perf
    opt.gpu = gpu
    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))
    if opt.distributed:
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        if opt.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            opt.rank = opt.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

    # define model, optimiser and scheduler
    tasks = opt.tasks
    num_inp_feats = 6   # number of channels in input
    tasks_outputs_tmp = {
        "water_mask": 1,
    }
    tasks_outputs = {t: tasks_outputs_tmp[t] for t in tasks}
    model = get_model(opt, tasks_outputs=tasks_outputs, num_inp_feats=num_inp_feats, pretrained=(opt.pretrained==1))
    params = []
    params += model.parameters()


    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif opt.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                model.cuda(opt.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                opt.batch_size = int(opt.batch_size / ngpus_per_node)
                opt.workers = int((opt.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=(opt.find_unused_param==1))
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=(opt.find_unused_param==1))
    elif opt.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        model = model.cuda(opt.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if opt.gpu:
            device = torch.device('cuda:{}'.format(opt.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    optimizer = optim.Adam(params, lr=opt.lr)
    scheduler = None
    if opt.scheduler == "steplr":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    lgr.debug(f"Using scheduler: {scheduler}")

    start_epoch = 0
    if opt.ckpt_path is not None:
        lgr.debug(f"Loading checkpoint: {opt.ckpt_path}")
        if opt.gpu is None:
            checkpoint = torch.load(opt.ckpt_path)
        elif torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(opt.gpu)
            checkpoint = torch.load(opt.ckpt_path, map_location=loc)
        if opt.is_finetune==0:
            lgr.warning(f"Chaning start epoch, and loading optimizer")
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            lgr.warning(f"Not chaning start epoch, since finetuning. NOT loading optimizer.")
        tmp = model.load_state_dict(checkpoint['state_dict'], strict=False)
        lgr.debug(f"model load state: {tmp}")
        
        # scheduler.load_state_dict(checkpoint['scheduler'])
        lgr.debug(f"Successfully loaded checkpoint. Starting from epoch: {start_epoch}")
    else:
        lgr.debug(f"No checkpoint loaded.")

    title = 'S2Mask'
    logger = Logger(os.path.join(opt.out, '{}--{}--{}--{}--{}--mtl_baselines_{}_{}_log.txt'.format(DATE_STR, opt.segment_model, opt.adaptor, opt.backbone, opt.head, opt.method, opt.weight)), title=title)
    logger_names = ['Epoch', 'T.Lwm', 'T.wmF1', 'T.wmRec', 'T.wmPrec',
        'V.Lwm', 'V.wmF1', 'V.wmRec', 'V.wmPrec', 'opt.lr']
    logger.set_names(logger_names)

    lgr.debug(f"LOSS FORMAT: {logger_names}\n")

    # define dataset path
    train_dataset1 = Sentinel2Data(root=opt.data_dir, split="train", resize_size=opt.resize_size, adaptor=opt.adaptor)
    val_dataset1 = Sentinel2Data(root=opt.data_dir, split="val", resize_size=opt.resize_size, adaptor=opt.adaptor)
    lgr.debug(f"Found {len(train_dataset1)} train_dataset1 and {len(val_dataset1)} val_dataset1")

    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset1, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset1, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset1, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset1, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    lgr.debug(f"Found {len(train_loader)} train_loader and {len(val_loader)} val_loader")

    # define parameters
    total_epoch = opt.epochs
    train_batch = len(train_loader)
    val_batch = len(val_loader)
    lgr.debug(f"train_batch={train_batch} val_batch={val_batch}")
    avg_cost = np.zeros([total_epoch, len(logger_names)-2], dtype=np.float32)   # 2 are epoch, lr
    
    isbest = False
    for epoch in range(start_epoch, total_epoch):
        cost = np.zeros(len(logger_names)-2, dtype=np.float32)

        bar = Bar('Training', max=train_batch)

        # iteration for all batches
        model.train()
        train_dataset = iter(train_loader)
        
        train_loss0 = AverageMeter('trainLoss0', ':.4e')
        wmask_train_met = BinMaskMeter()
        for k in range(train_batch):
            train_data, train_labels = next(train_dataset)
            train_data = train_data.to(device, non_blocking=True)
            for task_name in tasks:
                train_labels[task_name] = train_labels[task_name].to(device, non_blocking=True)
            train_pred, feat = model(train_data, feat=True)
            train_loss = get_dense_tasks_losses(train_pred, train_labels, tasks, returndict=False, loss_type=opt.loss_type)
            loss = train_loss[0]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            wmask_train_met.update(train_pred['water_mask'].to(device, non_blocking=True), train_labels['water_mask'].to(device, non_blocking=True))
            train_loss0.update(train_loss[0].item(), train_data.shape[0])
            cost[0] = train_loss[0].item()
            
            avg_cost[epoch, :4] += cost[:4] / train_batch
            bar.suffix  = '{} => ({batch}/{size}) | LossWm: {loss_wm:.4f}.'.format(
                        opt.weight,
                        batch=k + 1,
                        size=train_batch,
                        loss_wm=cost[0],
                        )
            bar.next()
        bar.finish()
        train_loss0.all_reduce()
    
        if scheduler is not None:
            scheduler.step()
            
        avg_cost[epoch, 0] = train_loss0.avg
        avg_cost[epoch, 1:4] = wmask_train_met.get_metrics()

        # evaluating test data
        model.eval()
        wmask_met = BinMaskMeter()
        val_loss0 = AverageMeter('ValLoss0', ':.4e')
        
        with torch.no_grad():  # operations inside don't track history
            val_dataset = iter(val_loader)
            for k in range(val_batch):
                val_data, val_labels = next(val_dataset)
                val_data = val_data.to(device, non_blocking=True)
                for task_name in tasks:
                    val_labels[task_name] = val_labels[task_name].to(device, non_blocking=True)

                val_pred = model(val_data)
                val_loss = get_dense_tasks_losses(val_pred, val_labels, tasks, loss_type=opt.loss_type)

                wmask_met.update(val_pred['water_mask'].to(device, non_blocking=True), val_labels['water_mask'].to(device, non_blocking=True))
                val_loss0.update(val_loss[0].item(), val_data.shape[0])
            val_loss0.all_reduce()
            avg_cost[epoch, 4] = val_loss0.avg
            avg_cost[epoch, 5:8] = wmask_met.get_metrics()

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total_val_loss = torch.tensor([val_loss[0]], dtype=torch.float32, device=device)
        dist.all_reduce(total_val_loss, dist.ReduceOp.SUM, async_op=False)
        val_loss[0] = total_val_loss.tolist()[0]
        ave_loss = (val_loss[0])/1.0
        

        isbest = (avg_cost[epoch, 5] > best_perf)  # water mask F1 score
        if isbest:
            best_perf = avg_cost[epoch, 5]  # water mask F1 score

        lgr.debug(f"Epoch: {epoch:04d} | TRAIN: {[x for x in avg_cost[epoch, :4]]} | VAL: {[x for x in avg_cost[epoch, 4:]]}")
        log_data = [epoch]
        for i in range(len(logger_names)-2):
            log_data.append(avg_cost[epoch, i])
        log_data += [opt.lr]
        logger.append(log_data)
        
        # lgr.debug(f"cur_performance: {avg_cost[epoch, 4]}")
        # lgr.debug(f"best_perf: {best_perf}")
        if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed
                and opt.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_performance': best_perf,
                'cur_performance': avg_cost[epoch, 5],  # water mask F1 score (val)
                'optimizer' : optimizer.state_dict(),
                'opt': opt,
                'scheduler': scheduler if scheduler is not None else None,
            }, isbest, opt)
    lgr.debug(f"Epoch: {epoch:04d} | TRAIN: {[x for x in avg_cost[epoch, :4]]} | VAL: {[x for x in avg_cost[epoch, 4:]]}")

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)
        
if __name__ == "__main__":
    main()

