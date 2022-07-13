import sys
import argparse  
import os
from base.config import base_config, cfg_convert


def get_config():
    # Default configure
    ''' 
    # For ImageNet pretrain weights
    cfg_dict = {
        'optim': 'SGD', # 'Adam'
        'schedule': 'StepLR',
        'lr': 5e-2,  # '5e-5'
        'batch': 8,
        'ave_batch': 1,
        'epoch': 20,
        'step_size': '21',
        'gamma': 0.1,
        'clip_gradient': 0,
        'test_batch': 1,
    }
    '''
    cfg_dict = {
        'optim': 'SGD', # 'Adam'
        'schedule': 'StepLR',
        'lr': 0.1,  # '5e-2' - '1e-2'
        'batch': 8,
        'ave_batch': 1,
        'epoch': 20,
        'step_size': '12,16',
        'gamma': 0.1,
        'clip_gradient': 0,
        'test_batch': 1,
    }
    
    parser = base_config(cfg_dict)
    # Add custom params here
    # parser.add_argument('--size', default=320, type=int, help='Input size')
    parser.add_argument('--ac', default=0.05, type=float) #0.05
    parser.add_argument('--rgb', default=200, type=float) #200
    parser.add_argument('--lrw', nargs='+')
    parser.add_argument('--temp', default='temp1', help='Job name')
    parser.add_argument('--refine', action='store_true')

    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    
    if config['multi']:
      config['batch'] = 8
      config['ave_batch'] = 1
    else:
      config['batch'] = 8
      config['ave_batch'] = 1
    
    # Config post-process
    #if config['refine']:
    #  config['params'] = [['encoder', 0], ['decoder', config['lr']], ['enc2', config['lr'] * 0.001], ['refine', config['lr'] * 0.01]]
    #else:
    #  config['params'] = [['encoder', 0], ['decoder', config['lr']]]
    config['params'] = [['encoder', 0], ['decoder', config['lr']]]
    #config['params'] = [['encoder', config['lr'] * 0.1], ['decoder', config['lr']]]
    config['lr_decay'] = 0.9
    
    return config, None