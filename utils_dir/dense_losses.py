#
# Authors: Wei-Hong Li

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import pdb
from loguru import logger

EPS = 1e-7  # for max pool loss

def get_dense_tasks_losses(outputs, labels, tasks, returndict=False, opt=None, loss_type=None):
    losses = {}
    for task in tasks:
        losses[task] = get_task_loss(outputs[task], labels[task], task, opt, loss_type=loss_type)
    if returndict:
        return losses
    else:
        return list(losses.values())

class MaxPool2dSame(torch.nn.MaxPool2d):
    # Since pytorch Conv2d does not support same padding the same way in Keras, had to do a workaround
    # Adopted from https://github.com/pytorch/pytorch/issues/67551#issuecomment-954972351
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size, s=self.stride, d=self.dilation)
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size, s=self.stride, d=self.dilation)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

def adaptive_maxpool_loss(y_pred, y_true, alpha=0.25):
    # logger.debug(f"y_pred: {y_pred.shape}, y_true: {y_true.shape}")
    # From https://github.com/isikdogan/deepwatermap/blob/master/metrics.py#L33-L41
    y_pred = torch.clip(y_pred, EPS, 1-EPS)
    positive = -y_true * torch.log(y_pred) * alpha
    negative = -(1. - y_true) * torch.log(1. - y_pred) * (1-alpha)
    pointwise_loss = positive + negative
    # max_loss = torch.nn.MaxPool2d(kernel_size=8, stride=1, padding="same")(pointwise_loss)
    max_loss = MaxPool2dSame(kernel_size=8, stride=1)(pointwise_loss)
    # logger.debug(f"max_loss: {max_loss.shape} pointwise_loss: {pointwise_loss.shape}")
    x = pointwise_loss * max_loss
    # logger.debug(f"x: {x.shape}")
    x = torch.mean(x, dim=1)   # channel is index 1
    # logger.debug(f"after mean x: {x.shape}")
    return torch.mean(x)

def get_task_loss(output, label, task, opt=None, loss_type=None):
    if task in ["water_mask", "cloudshadow_mask", "cloud_mask", "snowice_mask", "sun_mask"]:
        if loss_type == "bce":
            loss_fn = nn.BCELoss()
            loss = loss_fn(torch.squeeze(output,1), torch.squeeze(label,1))
            # loss = F.binary_cross_entropy(torch.squeeze(output,1), torch.squeeze(label,1))
        elif loss_type == "adaptive_maxpool":
            loss = adaptive_maxpool_loss(torch.squeeze(output,1), torch.squeeze(label,1))
        return loss
    if task == 'semantic':
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(output, label, ignore_index=-1)
        return loss

    if task == 'depth':
        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(label, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda()

        # depth loss: l1 norm
        loss = torch.sum(torch.abs(output - label) * binary_mask) / torch.nonzero(binary_mask).size(0)
        return loss
    if task == 'normal':
        if opt is None:
            binary_mask = (torch.sum(label, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda()
            # normal loss: dot product
            loss = 1 - torch.sum((output * label) * binary_mask) / torch.nonzero(binary_mask).size(0)
        elif opt == 'l1':
            valid_mask = (torch.sum(label, dim=1, keepdim=True) != 0).cuda()
            loss = torch.sum(F.l1_loss(output, label, reduction='none').masked_select(valid_mask)) \
                / torch.nonzero(valid_mask, as_tuple=False).size(0)
        return loss

def get_performances(output, label, task, tasks_outputs):
    if task in ["water_mask", "cloudshadow_mask", "cloud_mask", "snowice_mask", "sun_mask"]:
        rec, prec, f1, acc = compute_mask_metrics(output, label)
        return rec, prec, f1, acc
    elif task == 'semantic':
        miou = compute_miou(output, label, tasks_outputs['semantic'])
        iou = compute_iou(output, label)
        return miou, iou
    elif task == 'depth':
        return depth_error(output, label)
    elif task == 'normal':
        return normal_error(output, label)

def compute_mask_metrics(pred, target, thresh=0.5):
    # logger.debug(f"pred: {pred.shape}, target: {target.shape}")
    pred = torch.squeeze(pred, 1)
    target = torch.squeeze(target, 1)
    # logger.debug(f"pred shape: {pred.shape}")
    # logger.debug(f"target shape: {target.shape}")
    thresh_pred = torch.where(pred > thresh, 1., 0.)
    # rec, rec_vec = get_recall(thresh_pred, target)
    # prec, prec_vec = get_precision(thresh_pred, target)
    # f1, _ = get_f1(rec_vec, prec_vec)
    rec = get_recall(thresh_pred, target)
    prec = get_precision(thresh_pred, target)
    f1 = get_f1(thresh_pred, target)
    # f1 = (2*prec*rec)/(prec + rec) if (prec + rec) else torch.tensor(0)

    acc = get_acc(thresh_pred, target)
    return rec, prec, f1, acc

def get_acc(pred, gt):
    return torch.sum(torch.where(pred==gt, 1, 0))/torch.tensor(gt.shape[0] * gt.shape[1] * gt.shape[2])

# def get_recall(pred, gt):   # if this is small, it means model is under-predicting
#     TP = torch.sum(torch.where((gt==1) & (pred==1), 1., 0.), axis=[-2,-1])
#     FN = torch.sum(torch.where((gt==1) & (pred==0), 1., 0.), axis=[-2,-1])   # gt is 1 but pred is 0
#     # return (TP/(TP+FN)) if (TP+FN) else torch.tensor(0)
#     rec_vec = torch.where((TP+FN)!=0, (TP/(TP+FN)), torch.tensor(0).cuda())
#     return torch.mean(rec_vec), rec_vec

# def get_precision(pred, gt):    # if this is small, it means model is over-predicting
#     TP = torch.sum(torch.where((gt==1) & (pred==1), 1., 0.), axis=[-2,-1])
#     FP = torch.sum(torch.where((gt==0) & (pred==1), 1., 0.), axis=[-2,-1])   # pred is 1 but gt is 0
#     # return (TP/(TP+FP)) if (TP+FP) else torch.tensor(0)
#     prec_vec = torch.where((TP+FP)!=0, (TP/(TP+FP)), torch.tensor(0).cuda())
#     return torch.mean(prec_vec), prec_vec

# def get_f1(rec_vec, prec_vec):
#     f1_vec = torch.where((prec_vec+rec_vec)!=0, 2*prec_vec*rec_vec/(prec_vec + rec_vec), torch.tensor(0.).cuda())
#     return torch.mean(f1_vec), f1_vec



def get_recall(y_pred, y_true):
    with torch.no_grad():
        TP = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
        TP_FN = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
        recall = TP / (TP_FN + EPS)
    return recall

def get_precision(y_pred, y_true):
    with torch.no_grad():
        TP = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
        TP_FP = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
        precision = TP / (TP_FP + EPS)
    return precision

def get_f1(y_pred, y_true):
    with torch.no_grad():
        precision = get_recall(y_pred, y_true)
        recall = get_precision(y_pred, y_true)
    return 2 * ((precision * recall) / (precision + recall + EPS))
    

def compute_miou(x_pred, x_output, class_nb):
    _, x_pred_label = torch.max(x_pred, dim=1)
    x_output_label = x_output
    batch_size = x_pred.size(0)
    for i in range(batch_size):
        true_class = 0
        first_switch = True
        for j in range(class_nb):
            pred_mask = torch.eq(x_pred_label[i], j * torch.ones(x_pred_label[i].shape).type(torch.LongTensor).cuda())
            true_mask = torch.eq(x_output_label[i], j * torch.ones(x_output_label[i].shape).type(torch.LongTensor).cuda())
            mask_comb = pred_mask.type(torch.FloatTensor) + true_mask.type(torch.FloatTensor)
            union     = torch.sum((mask_comb > 0).type(torch.FloatTensor))
            intsec    = torch.sum((mask_comb > 1).type(torch.FloatTensor))
            if union == 0:
                continue
            if first_switch:
                class_prob = intsec / union
                first_switch = False
            else:
                class_prob = intsec / union + class_prob
            true_class += 1
        if i == 0:
            batch_avg = class_prob / true_class
        else:
            batch_avg = class_prob / true_class + batch_avg
    return batch_avg / batch_size

def compute_iou(x_pred, x_output):
    _, x_pred_label = torch.max(x_pred, dim=1)
    x_output_label = x_output
    batch_size = x_pred.size(0)
    for i in range(batch_size):
        if i == 0:
            pixel_acc = torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
                        torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))
        else:
            pixel_acc = pixel_acc + torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
                        torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))
    return pixel_acc / batch_size

def depth_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).cuda()
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)

def normal_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)
