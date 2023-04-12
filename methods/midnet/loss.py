import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from util import *

import numpy as np
from math import exp

def IOU(pred, target):
    inter = target * pred
    union = target + pred - target * pred
    iou_loss = 1 - torch.sum(inter, dim=(1, 2, 3)) / (torch.sum(union, dim=(1, 2, 3)) + 1e-7)
    return iou_loss.mean()

def iou_loss(pred, gt):
    target = gt.gt(0.5).float()
    iou_out = IOU(pred, target)
    return iou_out

def Loss(preds, target, config):
    loss = 0
    ws = [1, 0.2]
    
    for pred, w in zip(preds['sal'], ws):
        pred = nn.functional.interpolate(pred, size=target.size()[-2:], mode='bilinear')
        loss += iou_loss(torch.sigmoid(pred), target) * w
    return loss