import sys
import importlib
from data import Test_Dataset
import torch
import time
from progress.bar import Bar
import os
from collections import OrderedDict
import cv2
from PIL import Image
from util import *
import numpy as np
import argparse

from data import *
from metric import *

# python3 eval.py  --data_path=../dataset/ --pre_path=maps/rgbt/ADF --mode=te
# python3 eval.py  --data_path=../dataset/ --pre_path=maps/vsod/LSD --mode=oe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../dataset/', help='The name of network')
    parser.add_argument('--vals', default='all', help='Set the testing sets')
    
    parser.add_argument('--pre_path', default='./maps', help='Weight path of network')
    parser.add_argument('--mode', default='ce', help='Weight path of network')
    
    params = parser.parse_args()
    config = vars(params)
    config['orig_size'] = True
    config['size'] = 320
    config['mode'] = config['mode'].split(',')
    config['stage'] = 1
    config['trset'] = 'tr'
    
    if config['vals'] == 'all':
        vals = ['SED2', 'PASCAL-S', 'ECSSD', 'HKU-IS', 'DUTS-TE', 'DUT-OMRON']
    else:
        vals = config['vals'].split(',')
    
    #print(config['mode'])
    test_sets = get_test_list(modes=config['mode'], config=config)
        
    #print(test_sets.items())
    for set_name, test_set in test_sets.items():
        #img_path = '{}/{}/'.format(config['pre_path'], val)
        set_sub = set_name.split('_')[-1]
        #print(set_sub)
        img_path = '{}/{}/'.format(config['pre_path'], set_sub)
        #img_path = config['pre_path']
        if not os.path.exists(img_path):
            print('{} not exists!!!!!'.format(img_path))
            continue
        #test_set = Test_Dataset(name=val, config=config)
        titer = test_set.size
        MR = MetricRecorder(titer)
        #MR = MetricRecorder()
        
        #print(titer)
        test_bar = Bar('Dataset {:10}:'.format(set_name), max=titer)
        pre_name = ''
        kk = 0
        for j in range(titer):
            sample_dict = test_set.load_data(j)
            gt = sample_dict['gt']
            name = sample_dict['name']
            #_, gt, name = test_set.load_data(j)
            name = name.split('.')[0]
            
            a,b = name.split('/')
            
            #if pre_name == a:
            #    kk += 1
            #else:
            #    kk = 0
            #name = '{}/{}'.format(a, kk)
            #name = '{}/{}_{}'.format(a, a, b)
            
            pred = Image.open(img_path + name + '.png').convert('L')
            #print(np.max(pred))
            out_shape = gt.shape
            
            #MR.update(pre=pred, gt=gt)
            pred = np.array(pred.resize((out_shape[::-1])))
            
            pred, gt = normalize_pil(pred, gt)
            MR.update(pre=pred, gt=gt)
            #print(np.max(pred), np.max(gt))
            #MR.update(pre=pred.astype(np.uint8), gt=(gt * 255).astype(np.uint8))
            
                
            Bar.suffix = '{}/{}'.format(j, titer)
            test_bar.next()
        
        #scores = MR.show(bit_num=3)
        mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
        print('  Max-F: {}, Maen-F: {}, Fbw: {}, MAE: {}, SM: {}, EM: {}.'.format(maxf, meanf, wfm, mae, sm, em))
        #print('  Max-F: {}, adp-F: {}, Fbw: {}, MAE: {}, SM: {}, EM: {}.'.format(scores['fm'], scores['adpFm'], scores['wFm'], scores['MAE'], scores['Sm'], scores['adpEm']))
        #mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
        #print('  MAE: {}, Max-F: {}, Maen-F: {}, SM: {}, EM: {}, Fbw: {}.'.format(mae, maxf, meanf, sm, em, wfm))

    
if __name__ == "__main__":
    main()