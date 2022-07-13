from collections import OrderedDict

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from PIL import Image
from util import *
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import manifold, datasets

def normalize(im):
    im = (im - torch.min(im)) / (torch.max(im) - torch.min(im))
    return im
    
def Saver(preds, gt, name, save_folder, config):
    pass
'''
def Saver(preds, gt, name, save_folder, config):
    loc_x, reg_x = preds['feat']
    feats = loc_x * reg_x
    feats = feats[0].cpu().detach().numpy()
    feats = feats.transpose((1, 2, 0))
    
    im = Image.fromarray(gt.astype(np.uint8)).resize((160, 160))
    im = (np.array(im) > 0.5).astype(np.float)
    
    print(feats.shape, im.shape)
    feats = np.reshape(feats, (-1, 64))
    y = np.reshape(im, (-1))
    print(feats.shape, y.shape)
    
    #feats = feats[:100]
    #y = y[:100]
    
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(feats)
    
    print("Org data dimension is {}. Embedded data dimension is {}".format(feats.shape[-1], X_tsne.shape[-1]))
    
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    #plt.show()
    plt.savefig("./filename.png")
'''
# Save intermediate prediction
'''
def Saver(preds, gt, name, save_folder, config):
    #print(save_folder)
    
    loc_x, reg_x = preds['feat']
    loc_x = loc_x[0]
    reg_x = reg_x[0]
    
    assert loc_x.size() == reg_x.size()
    channel = loc_x.size()[0]
    atten = loc_x * reg_x
    for i in range(channel):
        att_i_path = os.path.join(save_folder, 'atten', str(i))
        check_path(att_i_path)
        att_path = os.path.join(att_i_path, name + '.png')
        
        att_map = atten[i]
        att_map = normalize(att_map)
        
        Image.fromarray(att_map.cpu().detach().numpy() * 255).convert('L').save(att_path)
        
        loc_i_path = os.path.join(save_folder, 'loc', str(i))
        reg_i_path = os.path.join(save_folder, 'reg', str(i))
        check_path(loc_i_path)
        check_path(reg_i_path)
        
        loc_map = loc_x[i]
        reg_map = reg_x[i]
        
        loc_map = normalize(loc_map)
        reg_map = normalize(reg_map)
        
        loc_path = os.path.join(loc_i_path, name + '.png')
        reg_path = os.path.join(reg_i_path, name + '.png')
        
        Image.fromarray(loc_map.cpu().detach().numpy() * 255).convert('L').save(loc_path)
        Image.fromarray(reg_map.cpu().detach().numpy() * 255).convert('L').save(reg_path)
'''