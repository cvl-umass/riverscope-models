import os
import torch
import rasterio
import cv2
import numpy as np
import torch.nn as nn
from models.get_model import get_model
import numpy as np
import pandas as pd
from utils_dir import mkdir_p 

from dataset.planet_segmentation import PlanetSegmentation
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

from loguru import logger
from datetime import datetime
from tqdm import tqdm
import argparse
from skimage.transform import rescale, resize

EPS=1e-7




parser = argparse.ArgumentParser(description='Baselines (SegNet)')
parser.add_argument('--data_dir', default='/work/pi_smaji_umass_edu/rdaroya/planet-benchmark/results/RiverScope_dataset', type=str, help='Path to dataset')
parser.add_argument('--batch_size', default=12, type=int, help='Batch size')
parser.add_argument('--is_distrib', default=True, type=int, help='Batch size')
parser.add_argument('--to_save_imgs', default=0, type=int, help='Set to 1 to save image outputs')
parser.add_argument('--is_downsample', default=0, type=int, help='Set to 1 to downsample then upsample input images (to simulate lower res)')
parser.add_argument('--thresh', default=None, type=float, help='Set to None to find threshold from val set. Otherwise, set to optimal thresh value 0-1')
parser.add_argument('--tasks', default=["water_mask"], nargs='+', help='Task(s) to be trained')
parser.add_argument('--ckpt_path', default=None, type=str, help='specify location of checkpoint')
# parser.add_argument('--weight', default='uniform', type=str, help='multi-task weighting: uniform, gradnorm, mgda, uncert, dwa, gs')
# parser.add_argument('--backbone', default='mobilenetv3', type=str, help='shared backbone')
# parser.add_argument('--head', default='mobilenetv3_head', type=str, help='task-specific decoder')
# parser.add_argument('--pretrained', default=False, type=int, help='using pretrained weight from ImageNet')

parser.add_argument('--method', default='vanilla', type=str, help='vanilla or mtan')
parser.add_argument('--out', default='./results/planet-test-eval', help='Directory to output the result')



def save_imgs(out_img_dir, input_fps, test_labels, pred_water_mask, to_save_rgb=False, to_save_gt=False):
    test_labels_np = test_labels.detach().cpu().numpy()
    pred_mask_np = pred_water_mask.detach().cpu().numpy()
    for idx in range(len(input_fps)):
        input_fp = input_fps[idx]
        test_label = test_labels_np[idx]    # 0 and 1 values
        pred_mask = pred_mask_np[idx]       # 0 and 1 values
        out_name = input_fp.split("/")[-3:]
        out_name = "--".join(out_name)
        out_fp_tif = os.path.join(out_img_dir, out_name)

        input_dataset = rasterio.open(input_fp)
        # Write prediction to TIFF
        kwargs = input_dataset.meta
        kwargs.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw')
        with rasterio.open(out_fp_tif, 'w', **kwargs) as dst:
            dst.write_band(1, pred_mask.astype(rasterio.float32))
        # Write prediction to PNG
        out_fp_png = out_fp_tif.replace(".tif", ".png")
        cv2.imwrite(out_fp_png, pred_mask*255)


        # Write GT to TIFF
        if to_save_gt:
            out_fp_gt_tif = out_fp_tif.replace(".tif", "--gt.tif")
            with rasterio.open(out_fp_gt_tif, 'w', **kwargs) as dst:
                dst.write_band(1, test_label.astype(rasterio.float32))

        # RGB image
        if to_save_rgb:
            img = input_dataset.read()
            img = np.transpose(img, (1,2,0))
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            out_name_rgb = out_fp_tif.replace(".tif", "--rgb.png")
            cv2.imwrite(out_name_rgb, img[:,:,:3]*255)
    

opt = parser.parse_args()
logger.debug(f"opt: {opt}")

if not os.path.isdir(opt.out):
    mkdir_p(opt.out)

to_save_imgs = (opt.to_save_imgs!=0)
out_img_dir = os.path.join(opt.out, opt.ckpt_path.split("/")[-1].split(".")[0])
logger.debug(f"to_save_imgs: {to_save_imgs}")
if to_save_imgs:
    logger.debug(f"Creating output folder for images: {out_img_dir}")
    mkdir_p(out_img_dir)

tasks = opt.tasks
num_inp_feats = 4   # number of channels in input
tasks_outputs_tmp = {
    "water_mask": 1,
}
tasks_outputs = {t: tasks_outputs_tmp[t] for t in tasks}
logger.debug(f"opt: {opt.__dict__}")


logger.debug(f"Loading weights from {opt.ckpt_path}")
checkpoint = torch.load(opt.ckpt_path, weights_only=False)
ckpt_fn = opt.ckpt_path.split("/")[-1].replace(".pth.tar", "")
logger.debug(f"ckpt_fn: {ckpt_fn}")
ckpt_opt = checkpoint["opt"]
ckpt_lr = ckpt_opt.lr
logger.debug(f"ckpt_lr: {ckpt_lr}")
model = get_model(ckpt_opt, tasks_outputs=tasks_outputs, num_inp_feats=num_inp_feats, pretrained=(ckpt_opt.pretrained==1))

# if opt.is_distrib:
new_ckpt = {k.split("module.")[-1]:v for k,v in checkpoint["state_dict"].items()}
checkpoint["state_dict"] = new_ckpt
tmp = model.load_state_dict(checkpoint["state_dict"], strict=True)

logger.debug(f"After loading ckpt: {tmp}")
logger.debug(f"Checkpoint epoch: {checkpoint['epoch']}. best_perf: {checkpoint['best_performance']}")
model.cuda()
model.eval()

# Find optimal threshold using validation set
# """
if opt.thresh is None:
    val_dataset1 = PlanetSegmentation(root=opt.data_dir, split="valid", resize_size=ckpt_opt.resize_size, adaptor=ckpt_opt.adaptor)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset1, batch_size=opt.batch_size, shuffle=False,
        num_workers=1, pin_memory=True, sampler=val_sampler, drop_last=False)
    val_batch = len(val_loader)
    val_dataset = iter(val_loader)
    logger.debug(f"Evaluating on {val_batch} val batches to find best threshold")
    rgbs = []
    thresh_choices = np.arange(0,1,0.1)
    thresh_metrics = {t: {th: {"TP":[], "FP":[], "FN":[], "TN":[], "num_px": []} for th in thresh_choices} for t in tasks}
    counts_tps = {t: {"TP":[], "FP":[], "FN":[], "TN":[], "num_px": []} for t in tasks}    # number of pixels
    with torch.no_grad():
        for thresh in thresh_choices:
            print(f"Evaluating on thresh={thresh}")
            val_dataset = iter(val_loader)
            for k in tqdm(range(val_batch)):
                val_data, val_labels = next(val_dataset)
                val_data = val_data.cuda()
                for task_name in tasks:
                    val_labels[task_name] = val_labels[task_name].cuda()
                
                val_pred, feat = model(val_data, feat=True)
                for t in tasks:
                    pred = torch.squeeze(val_pred[t], 1)
                    target = torch.squeeze(val_labels[t], 1)
                    thresh_pred = torch.where(pred > thresh, 1., 0.)

                    TP = torch.sum(torch.round(torch.clip(target * thresh_pred, 0, 1)))
                    FP = torch.sum(torch.round(torch.clip((1-target) * thresh_pred, 0, 1))) # target is 0, but pred is 1
                    FN = torch.sum(torch.round(torch.clip(target * (1-thresh_pred), 0, 1))) # target is 1, but pred is 0
                    TN = torch.sum(torch.round(torch.clip((1-target) * (1-thresh_pred), 0, 1))) # target is 0, and pred is 0
                    
                    thresh_metrics[t][thresh]["TP"].append(TP.item())
                    thresh_metrics[t][thresh]["FN"].append(FN.item())
                    thresh_metrics[t][thresh]["FP"].append(FP.item())
                    thresh_metrics[t][thresh]["TN"].append(TN.item())
                    s1,s2,s3 = pred.shape
                    num_px = s1*s2*s3
                    assert num_px == (TP+FN+FP+TN)
                    thresh_metrics[t][thresh]["num_px"].append(num_px)

    metric_names = ["f1", "rec", "prec", "acc", "iou"]
    task_metric_per_thresh = {t: {m: [] for m in metric_names} for t in tasks}
    for t in tasks:
        for th in thresh_choices:
            TP_tot = np.sum(np.array(thresh_metrics[t][th]["TP"]))
            FP_tot = np.sum(np.array(thresh_metrics[t][th]["FP"]))
            FN_tot = np.sum(np.array(thresh_metrics[t][th]["FN"]))
            TN_tot = np.sum(np.array(thresh_metrics[t][th]["TN"]))
            prec = TP_tot/(TP_tot + FP_tot + EPS)
            rec = TP_tot/(TP_tot + FN_tot + EPS)
            f1 = (2*prec*rec)/(prec+rec + EPS)
            acc = (TP_tot + TN_tot)/(TP_tot + TN_tot + FN_tot + FP_tot + EPS)
            miou = TP_tot/(TP_tot + FP_tot + FN_tot + EPS)
            task_metric_per_thresh[t]["f1"].append(f1)
            task_metric_per_thresh[t]["rec"].append(rec)
            task_metric_per_thresh[t]["prec"].append(prec)
            task_metric_per_thresh[t]["acc"].append(acc)
            task_metric_per_thresh[t]["iou"].append(miou)
    # thresh_metrics
    # task_metric_per_thresh

    # Find optimal threshold given metrics (based on f1 score)
    optim_threshes = {t:None for t in tasks}
    for t in tasks:
        optim_idx = np.argmax(task_metric_per_thresh[t]["f1"])
        optim_thresh = thresh_choices[optim_idx]
        optim_threshes[t] = optim_thresh
        print(f"{t} optim thresh: {optim_thresh} [f1: {task_metric_per_thresh[t]['f1'][optim_idx]}]")
    logger.debug(f"optim_threshes: {optim_threshes}")
# """
else:
    optim_threshes = {"water_mask": opt.thresh}
logger.debug(f"Using the following thresholds: {optim_threshes}")


test_dataset1 = PlanetSegmentation(root=opt.data_dir, split="test", resize_size=ckpt_opt.resize_size, adaptor=ckpt_opt.adaptor, return_fp=True)
logger.debug(f"Using batch size 1 for test loader")
test_sampler = None
test_loader = torch.utils.data.DataLoader(
    test_dataset1, batch_size=1, shuffle=False,
    num_workers=1, pin_memory=True, sampler=test_sampler, drop_last=False)
test_batch = len(test_loader)
test_dataset = iter(test_loader)


logger.debug(f"Evaluating on {test_batch} test batches")
metrics = {t: {"f1":[], "rec":[], "prec":[], "acc": []} for t in tasks}
counts_tps = {t: {"TP":[], "FP":[], "FN":[], "TN":[], "num_px": []} for t in tasks}    # number of pixels

fmask_counts_tps = {"TP":[], "FP":[], "FN":[], "TN":[], "num_px": []}
mndwi_counts_tps = {"TP":[], "FP":[], "FN":[], "TN":[], "num_px": []}
model_counts_tps = {"TP":[], "FP":[], "FN":[], "TN":[], "num_px": []}

gtpreds = {t: {"gt":[], "pred":[]} for t in tasks}


rgbs = []
per_img_results = []
with torch.no_grad():
    for k in tqdm(range(test_batch)):
        test_data, test_labels, input_fp = next(test_dataset)
        
        gt_water_mask = test_labels["water_mask"] 
        gt_water_mask = torch.squeeze(gt_water_mask, 1).cuda()

        test_data = test_data.cuda()
        test_pred, feat = model(test_data, feat=True)

        pred = test_pred["water_mask"]
        thresh_pred = torch.where(pred > optim_threshes["water_mask"], 1., 0.)
        pred_water_mask = thresh_pred
        pred_water_mask = torch.squeeze(pred_water_mask, 1)


        model_TP = torch.sum(torch.round(torch.clip(gt_water_mask * pred_water_mask, 0, 1)))
        model_FP = torch.sum(torch.round(torch.clip((1-gt_water_mask) * pred_water_mask, 0, 1))) # gt_water_mask is 0, but pred is 1
        model_FN = torch.sum(torch.round(torch.clip(gt_water_mask * (1-pred_water_mask), 0, 1))) # gt_water_mask is 1, but pred is 0
        model_TN = torch.sum(torch.round(torch.clip((1-gt_water_mask) * (1-pred_water_mask), 0, 1))) # gt_water_mask is 0, and pred is 0
        model_counts_tps["TP"].append(model_TP.item())
        model_counts_tps["FP"].append(model_FP.item())
        model_counts_tps["FN"].append(model_FN.item())
        model_counts_tps["TN"].append(model_TN.item())

        img_prec = model_TP/(model_TP + model_FP + EPS)
        img_rec = model_TP/(model_TP + model_FN + EPS)
        img_f1 = (2*img_prec*img_rec)/(img_prec+img_rec + EPS)
        img_iou = model_TP/(model_TP + model_FP + model_FN + EPS)
        out_name = input_fp[0].split("/")[-3:]
        out_name = "--".join(out_name)
        perc_water = torch.sum(test_labels["water_mask"])/(500*500)
        per_img_results.append([input_fp[0], out_name, model_TP.item(), model_FP.item(), model_FN.item(), model_TN.item(), img_f1.item(), img_prec.item(), img_rec.item(), img_iou.item(), perc_water.item()])

        if to_save_imgs:
            save_imgs(out_img_dir, input_fp, test_labels["water_mask"], pred_water_mask)

# NOTE: Uncomment the next 3 lines to save the metric per image
# out_fp_perimg = os.path.join(opt.out, f"PERIMG_{ckpt_fn}.csv")
# per_img_results_df = pd.DataFrame(per_img_results, columns=["input_fp", "out_name", "TP", "FP", "FN", "TN", "f1", "prec", "rec", "iou", "perc_water"])
# per_img_results_df.to_csv(out_fp_perimg, index=False)

print(f"model,optim_thresh_water,ckpt_lr,f1,rec,prec,acc,miou")

optim_thresh_water = optim_threshes["water_mask"]

TP_tot = np.sum(np.array(model_counts_tps["TP"]))
FP_tot = np.sum(np.array(model_counts_tps["FP"]))
FN_tot = np.sum(np.array(model_counts_tps["FN"]))
TN_tot = np.sum(np.array(model_counts_tps["TN"]))
prec = TP_tot/(TP_tot + FP_tot + EPS)
rec = TP_tot/(TP_tot + FN_tot + EPS)
f1 = (2*prec*rec)/(prec+rec + EPS)
acc = (TP_tot + TN_tot)/(TP_tot + TN_tot + FN_tot + FP_tot + EPS)
miou = TP_tot/(TP_tot + FP_tot + FN_tot + EPS)
print(f"model,{optim_thresh_water},{ckpt_lr},{f1},{rec},{prec},{acc},{miou}")

out_fp = os.path.join(opt.out, f"{ckpt_fn}.csv")
out_df_data = [[
    ckpt_opt.segment_model, ckpt_opt.backbone, ckpt_opt.head, ckpt_opt.adaptor, ckpt_opt.resize_size, 
    optim_thresh_water, ckpt_lr, f1, rec, prec, acc, miou, opt.ckpt_path
]]
out_df = pd.DataFrame(out_df_data, columns=[
    "segment_model", "backbone", "head", "adaptor", "resize_size",
    "thresh", "lr", "f1", "rec", "prec", "acc", "miou", "ckpt_path"
])
out_df.to_csv(out_fp, index=False)
