import sys
import argparse  
import os
from base.config import base_config, cfg_convert


def get_config():
    # Default configure
    cfg_dict = {
        'optim': 'SGD',
        'schedule': 'StepLR',
        'lr': 0.005,
        'batch': 8,
        'ave_batch': 1,
        'epoch': 10,
        'step_size': '6,8',
        'gamma': 0.5,
        'clip_gradient': 0,
        'test_batch': 1
    }
    
    parser = base_config(cfg_dict)
    # Add custom params here
    parser.add_argument('--resdual', default=0.4, type=float)
    
    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    
    return config, None