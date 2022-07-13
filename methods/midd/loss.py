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

def get_contour(label):
    lbl = label.gt(0.5).float()
    ero = 1 - F.max_pool2d(1 - lbl, kernel_size=5, stride=1, padding=2)  # erosion
    dil = F.max_pool2d(lbl, kernel_size=5, stride=1, padding=2)            # dilation
    
    edge = dil - ero
    return edge
    
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map#.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def IOU(pred, target):
    #inter = torch.sum(target * pred * mask, dim=(1, 2, 3))
    #union = torch.sum((target + pred) * mask, dim=(1, 2, 3)) - inter
    #iou_loss = 1 - (inter / union).mean()
    inter = target * pred
    union = target + pred - target * pred
    iou_loss = 1 - torch.sum(inter, dim=(1, 2, 3)) / (torch.sum(union, dim=(1, 2, 3)) + 1e-7)
    return iou_loss.mean()

def bce_ssim_loss(pred, gt):
    #mask = (1 - ((gt > 0.4) * (gt < 0.6)).float()).detach() #((gt > 0.3) * (gt < 0.7)).float().detach()
    #mask = get_contour(gt)
    
    target = gt.gt(0.5).float()
    #print(torch.max(pred), torch.max(target))
    #bce_out = nn.BCELoss(reduction='none')(pred, target)
    #ssim_out = 1 - SSIM(window_size=11,size_average=False)(pred, target)
    #ssim_loss = torch.sum(ssim_out * mask) / torch.sum(mask)
    iou_out = IOU(pred, target)
    #iou_out = IOU_mask(pred, target, mask)

    #print(bce_out.shape, ssim_out.shape, iou_out.shape, mask.shape)

    #print((bce_out * mask).mean(), (ssim_out * mask).mean(), iou_out)
    #loss = (bce_out * mask).mean() + (ssim_out * mask).mean() + iou_out
    #loss = bce_out.mean() + ssim_loss * 0.1 + iou_out
    #loss = bce_out.mean() + iou_out
    loss = iou_out

    return loss

def Loss(preds, target, config):
    loss = 0
    ws = [1, 0.2]
    
    for pred, w in zip(preds['sal'], ws):
        pred = nn.functional.interpolate(pred, size=target.size()[-2:], mode='bilinear')
        loss += bce_ssim_loss(torch.sigmoid(pred), target) * w
        #loss += bce_ssim_loss(pred, target) * w
        
        #js_loss = torch.mean(torch.sum(js_div(*preds['feat']), dim=1))
        #print(js_loss)
        #loss += (1 - js_loss) * 0.5
        
    return loss