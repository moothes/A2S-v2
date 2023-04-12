import os
import torch
import numpy as np
import importlib
from torch.optim import SGD, Adam
import torch.optim.lr_scheduler as sche
from thop import profile
import torch.distributed as dist

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

def load_framework(net_name):
    # Load Configure
    config, schedule = importlib.import_module('methods.{}.config'.format(net_name)).get_config()
    print(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus']
    
    # Constructing network
    model = importlib.import_module('base.model').Network(net_name, config)
    
    loss = importlib.import_module('methods.{}.loss'.format(net_name)).Loss
    
    # Loading Saver if it exists
    if config['save'] and os.path.exists('methods/{}/saver.py'.format(net_name)):
        saver = importlib.import_module('methods.{}.saver'.format(net_name)).Saver
    else:
        saver = None
    
    gpus = range(len(config['gpus'].split(',')))
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus)
        orig_model = model.module
    else:
        orig_model = model
    
    model = model.cuda()
    
    # Set optimizer and schedule
    optim = config['optim']
    if optim == 'SGD':
        if 'params' in config.keys():
            module_lr = [{'params' : getattr(model, p[0]).parameters(), 'lr' : p[1]} for p in config['params']]
            optimizer = SGD(params=module_lr, lr=config['lr'], momentum=0.9, weight_decay=0.0005)
        else:
            encoder = []
            others = []
            for param in model.named_parameters():
                if 'encoder.' in param[0] or 't_net.' in param[0]:
                    encoder.append(param[1])
                else:
                    others.append(param[1])
            module_lr = [{'params' : encoder, 'lr' : config['lr']*0.1}, {'params' : others, 'lr' : config['lr']}]
            optimizer = SGD(params=module_lr, momentum=0.9)
    elif optim == 'Adam':
        optimizer = Adam(filter(lambda p: p.requires_grad, orig_model.parameters()), lr=config['lr'], weight_decay=0.0005)
    
    # If get_config function doesn't return a valid schedule, it will be set here.
    if schedule is None:
        schedule = config['schedule']
        if schedule == 'StepLR':
            scheduler = sche.MultiStepLR(optimizer, milestones=config['step_size'], gamma=config['gamma'])
        elif schedule == 'poly':
            scheduler = poly_scheduler(optimizer, config['epoch'], config['lr_decay'])
        elif schedule == 'pfa':
            scheduler = pfa_scheduler(optimizer, config['epoch'])
        elif schedule == 'warmup':
            scheduler = WarmupMultiStepLR(optimizer=optimizer, milestones=config['step_size'], gamma=0.1, warmup_factor=0.1, warmup_iters=100, warmup_method="linear", last_epoch=-1)
        else:
            scheduler = sche.MultiStepLR(optimizer, milestones=[15, 20], gamma=0.5)
        
    return config, model, optimizer, scheduler, loss, saver

def poly_scheduler(optimizer, total_num=50, lr_decay=0.9):
    def get_lr_coefficient(curr_epoch):
        nonlocal total_num
        coefficient = pow((1 - float(curr_epoch) / total_num), lr_decay)
        return coefficient

    scheduler = sche.LambdaLR(optimizer, lr_lambda=get_lr_coefficient)
    return scheduler

def pfa_scheduler(optimizer, total_num=50):
    def get_lr_coefficient(curr_epoch):
        nonlocal total_num
        e_drop = total_num / 8.
        coefficient = 1-abs(curr_epoch/total_num*2-1)
        return coefficient

    scheduler = sche.LambdaLR(optimizer, lr_lambda=get_lr_coefficient)
    return scheduler
    
     
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1/3,
        warmup_iters=100,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
 
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
 
    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]