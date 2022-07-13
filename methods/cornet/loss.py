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

def Loss(preds, target, config):
    loss = 0
    
    target = target.gt(0.5).float()
    for pred in preds['sal']:
        pred = nn.functional.interpolate(pred, size=target.size()[-2:], mode='bilinear')
        loss += IOU(torch.sigmoid(pred), target)
        
    return loss