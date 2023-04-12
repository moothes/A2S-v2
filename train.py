import sys
import os
import time
import random
import cv2
from math import exp

from progress.bar import Bar
from collections import OrderedDict
from util import *
from PIL import Image
from data import Train_Dataset, Test_Dataset, get_loader, get_test_list
from test import test_model
import torch
from torch.nn import utils
from base.framework_factory import load_framework

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

torch.set_printoptions(precision=5)


def main():
    # Loading model
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
        
    # Loading model
    config, model, optim, sche, model_loss, saver = load_framework(net_name)
    config['net_name'] = net_name
    stage = config['stage']
    
    if config['weight'] != '':
        print('Load weights from: {}.'.format(config['weight']))
        model.load_state_dict(torch.load(config['weight'], map_location='cpu'))
        
    train_loader = get_loader(config)
    
    test_sets = get_test_list(config['vals'], config)
    
    debug = config['debug']
    num_epoch = config['epoch']
    num_iter = len(train_loader)
    ave_batch = config['ave_batch']
    #batch = ave_batch * config['batch']
    trset = config['trset']
    batch_idx = 0
    model.zero_grad()
    for epoch in range(1, num_epoch + 1):
        model.train()
        torch.cuda.empty_cache()
        
        if debug:
            test_model(model, test_sets, config, epoch)
        
        bar = Bar('{:10}-{:8} | epoch {:2}:'.format(net_name, config['sub'], epoch), max=num_iter)
        
        st = time.time()
        loss_count, adb_count, ac_count, mse_count = 0, 0, 0, 0
        optim.zero_grad()
        
        fin_lr = 0.2
        for i, pack in enumerate(train_loader, start=1):
            cur_it = i + (epoch-1) * num_iter
            total_it = num_epoch * num_iter
            itr = (1 - cur_it / total_it) * (1 - fin_lr) + fin_lr
            mul = itr
            
            if stage == 1:
                tune = 2 if 'MSB-TR' in pack['name'][0] else 20
                if epoch > tune:
                    optim.param_groups[0]['lr'] = config['lr'] * mul * 0.01
                else:
                    optim.param_groups[0]['lr'] = 0
                
                optim.param_groups[1]['lr'] = config['lr'] * mul
            else:
                optim.param_groups[0]['lr'] = config['lr'] * mul * 0.1
                optim.param_groups[1]['lr'] = config['lr'] * mul
                
            images = pack['image'].float()
            gts = pack['gt'].float()
            gt_names = pack['name']
            flips = pack['flip']
            
            images, gts = images.cuda(), gts.cuda()
            
            priors = [images]
            if 'dep' in pack.keys():
                priors.append(pack['dep'].float().cuda())
            if 'of' in pack.keys():
                priors.append(pack['of'].float().cuda())
            if 'th' in pack.keys():
                priors.append(pack['th'].float().cuda())
            
            loss = 0
            if stage == 1:
                Y = model(images, 'train')
                
                config['param'] = tran_param(config)
                images_temp = transform(images, False, config)
                priors = torch.cat(priors, dim=1)
                priors_temp = transform(priors, False, config)
                
                Y_ref = model(images_temp, 'train')
                
                lr_weight = np.array(config['lrw'].split(',')).astype(np.float)
                if lr_weight is None or len(lr_weight) != 3:
                    lr_weight = [0.5, 0.05, 1]
                
                loss0, loss1, loss2 = model_loss(Y, priors, Y_ref, priors_temp, epoch, lr_weight, config, gt_names)
                loss += loss0 + loss1 + loss2
                    
                ac_count += loss1
                mse_count += loss2
                
            elif stage > 1:
                Y = model(priors, 'train')
                loss0 = model_loss(Y, gts.gt(0.5).float(), config)
                loss = loss0
                
            loss_count += loss.data
            adb_count += loss0
            loss.backward()

            batch_idx += 1
            if batch_idx == ave_batch:
                if config['clip_gradient']:
                    utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
                optim.step()
                optim.zero_grad()
                batch_idx = 0
            
            lrs = ','.join([format(param['lr'], ".2e") for param in optim.param_groups])
            if stage == 1:
                Bar.suffix = '{:4}/{:4} | loss: {:1.3f}, dfs: {:1.3f}, bac: {:1.3f}, mse: {:1.3f}, LRs: [{}], time: {:1.3f}.'.format(
                    i, num_iter, float(loss_count / i), float(adb_count / i), float(ac_count / i), float(mse_count / i), lrs, time.time() - st)
            else:
                Bar.suffix = '{:4}/{:4} | loss: {:1.3f}, LRs: [{}], time: {:1.3f}.'.format(i, num_iter, float(loss_count / i), lrs, time.time() - st)
            bar.next()
            
            if epoch > 1 and stage > 1 and config['olr']:
                lamda = config['resdual']
                for gt_path, pred, gt, flip in zip(gt_names, torch.sigmoid(Y['final'].detach()), gts, flips):
                    if flip:
                        pred = pred.flip(2)
                    new_gt = (pred * (1 - lamda)).cpu().numpy().transpose(1, 2, 0)
                    cv2.imwrite(gt_path, new_gt * 255)
                
        sche.step()
        bar.finish()
        torch.cuda.empty_cache()
        test_model(model, test_sets, config, epoch)

if __name__ == "__main__":
    main()