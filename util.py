import os
import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import importlib
import scipy
import scipy.ndimage
from torch.optim import SGD, Adam
import torch

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from pydensecrf.utils import unary_from_softmax

def tran_param(config):
    param = {}
    param['scale'] = config['size'] + int(np.random.choice([-2, -1, 1, 2], 1) * 64)
    return param

def transform(image, is_mask=True, config=None):
    param = config['param']
    input_size = param['scale']
    if is_mask:
        temp = F.upsample(image, size=(input_size, input_size), mode='nearest')
    else:
        temp = F.upsample(image, size=(input_size, input_size), mode='bilinear', align_corners=True)
        
    return temp

def de_transform(image, is_mask=True, config=None):
    param = config['param']
    orig_size = config['size']
    mask = torch.ones_like(image)
    
    if is_mask:
        temp = F.upsample(image, size=(input_size, input_size), mode='nearest')
        mask = F.upsample(mask, size=(input_size, input_size), mode='nearest')
    else:
        temp = F.upsample(image, size=(input_size, input_size), mode='bilinear', align_corners=True)
    
    
    return temp, mask
    
def crf_inference_label(img, labels, t=10, n_labels=2, gt_prob=0.7):
    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)
'''
def crf_inference_label(img, labels, t=10, n_labels=2, gt_prob=0.7):
    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=5, compat=5)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)
'''
def crf_inference(img, probs, t=10, scale_factor=1, labels=1):
    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)
    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def cal_iou(p1, p2):
    bp1 = (p1 > 0.5)
    bp2 = (p2 > 0.5)

    inter = np.sum(bp1 * bp2)
    union = np.sum(((bp1 + bp2) > 0).astype(np.int8))
    iou = inter * 1. / union
    #print(inter, union, iou)
    
    return iou

def cal_dis(p1, p2):
    #bp1 = (p1 > 0.5)
    #bp2 = (p2 > 0.5)
    
    img1 = np.nonzero(p1 > 0.5)
    img2 = np.nonzero(p2 > 0.5)
    
    none_img = False
    
    if len(img1[0]) > 0:
        gt_x = np.mean(img1[0]) 
    else:
        none_img = True
    if len(img1[1]) > 0:
        gt_y = np.mean(img1[1])
    else:
        none_img = True
    if len(img2[0]) > 0:
        pred_x = np.mean(img2[0])
    else:
        none_img = True
    if len(img2[1]) > 0:
        pred_y = np.mean(img2[1])
    else:
        none_img = True
    
    if none_img:
        dis = 320
    else:
        dis = np.sqrt((pred_x - gt_x) * (pred_x - gt_x) + (pred_y - gt_y) * (pred_y - gt_y))
    
    #print(gt_x, gt_y, pred_x, pred_y, dis)
    
    return dis / (p2.shape[-1] + 1e-5)
    
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def freeze_bn(model):
    for m in model.base.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

def label_edge_prediction_old(label):
    fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
    fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)
    fx = np.reshape(fx, (1, 1, 3, 3))
    fy = np.reshape(fy, (1, 1, 3, 3))
    fx = Variable(torch.from_numpy(fx)).cuda()
    fy = Variable(torch.from_numpy(fy)).cuda()
    contour_th = 1.5
    
    # convert label to edge
    label = label.float()
    label = F.pad(label, (1, 1, 1, 1), mode='replicate')
    label_fx = F.conv2d(label, fx)
    label_fy = F.conv2d(label, fy)
    label_grad = torch.sqrt(torch.mul(label_fx, label_fx) + torch.mul(label_fy, label_fy))
    label_grad = torch.gt(label_grad, contour_th).float()

    return label_grad

def label_edge_prediction(label):
    ero = 1 - F.max_pool2d(1 - label, kernel_size=5, stride=1, padding=2)  # erosion
    dil = F.max_pool2d(label, kernel_size=5, stride=1, padding=2)            # dilation
    
    edge = dil - ero
    return edge

def mask_normalize(mask):
    return mask/(np.amax(mask)+1e-8)
