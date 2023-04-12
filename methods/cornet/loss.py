import torch
from torch import nn
from torch.nn import functional as F

from util import *

def IOU(pred, target):
    inter = target * pred
    union = target + pred - target * pred
    iou_loss = 1 - torch.sum(inter, dim=(1, 2, 3)) / (torch.sum(union, dim=(1, 2, 3)) + 1e-7)
    return iou_loss.mean()

def Loss(preds, target, config):
    loss = 0
    
    for pred in preds['sal']:
        pred = nn.functional.interpolate(pred, size=target.size()[-2:], mode='bilinear')
        target = target.gt(0.5).float()
        loss += IOU(torch.sigmoid(pred), target)
        
    return loss