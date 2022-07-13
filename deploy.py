import sys
import importlib
import torch
import os
from PIL import Image
from util import *
import numpy as np

from base.framework_factory import load_framework

mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])

def img_process(image, config):
    image = image.resize((config['size'], config['size']))
    image = np.array(image).astype(np.float32)
    image = ((image / 255.) - mean) / std
    image = image.transpose((2, 0, 1))
    im = torch.tensor(np.expand_dims(image, 0)).float()
    return im

def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    
    config, model, _, _, _, saver = load_framework(net_name)
    
    if config['weight'] != '':
        saved_model = torch.load(config['weight'], map_location='cpu')
        new_name = {}
        for k, v in saved_model.items():
            if k.startswith('model'):
                new_name[k[6:]] = v
            else:
                new_name[k] = v
        model.load_state_dict(new_name)
        model.eval()
        model = model.cuda()
    
    img_fold = '../dataset/McShip/images'
    #img_fold = './fewshot'
    img_list = os.listdir(img_fold)
    for img_name in img_list:
        img_path = os.path.join(img_fold, img_name)
        image = Image.open(img_path).convert('RGB')
        image = img_process(image, config)
        
        Y = model(image.cuda())
        pred = Y['final'].sigmoid_().cpu().data.numpy()

        im = Image.fromarray((pred[0, 0] * 255)).convert('L')
        im.save('./ship/' + img_name.split('.')[0] + '.png')
    
if __name__ == "__main__":
    main()