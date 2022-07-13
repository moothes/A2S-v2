import os, random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
from shutil import copyfile, copy
from collections import OrderedDict

mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])

sub = 'split' # 'joint'

def copydirs(from_file, to_file):
    if not os.path.exists(to_file):
        os.makedirs(to_file)
    files = os.listdir(from_file)
    for f in files:
        if os.path.isdir(from_file + '/' + f):
            copydirs(from_file + '/' + f, to_file + '/' + f)
        else:
            copy(from_file + '/' + f, to_file + '/' + f)

def OLR(gt_root, name):
    gt_orig = gt_root
    gt_root = os.path.join('./pseudo', 'temp2', name)
    
    print("Copying labels to temp folder: {}.".format(gt_root))
    copydirs(gt_orig, gt_root)
    print('Using temp labels from {}'.format(gt_root))
    return gt_orig, gt_root
    

def get_color_list(name, config, phase):
    name_list = []
    image_root = os.path.join(config['data_path'], name, 'images')
    img_list = os.listdir(image_root)
    
    if config['stage'] > 1 and phase == 'train':
        gt_root = './pseudo/c/{}'.format(name)
        print('Using pseudo labels from {}'.format(gt_root))
    else:
        gt_root = os.path.join(config['data_path'], name, 'segmentations')
    
    for img_name in img_list:
        img_tag = img_name.split('.')[0]
        
        tag_dict = {}
        tag_dict['rgb'] = os.path.join(image_root, img_name)
        tag_dict['gt'] = os.path.join(gt_root, img_tag + '.png')
        name_list.append(tag_dict)
    
    return name_list

def get_rgbd_list(name, config, phase):
    name_list = []
    image_root = os.path.join(config['data_path'], 'RGBD/{}/image'.format(name))
    dep_root = os.path.join(config['data_path'], 'RGBD/{}/depth'.format(name))
    
    if config['stage'] > 1 and phase == 'train':
        gt_root = './pseudo/d-split/{}'.format(name)
        print('Using pseudo labels from {}'.format(gt_root))
        
        if config['olr']:
            gt_orig, gt_root = OLR(gt_root, name)
    else:
        gt_root = os.path.join(config['data_path'], 'RGBD/{}/mask'.format(name))
    
    img_list = os.listdir(image_root)
    for img_name in img_list:
        img_tag = img_name.split('.')[0]
        
        tag_dict = {}
        tag_dict['rgb'] = os.path.join(image_root, img_name)
        tag_dict['gt'] = os.path.join(gt_root, img_tag + '.png')
        tag_dict['dep'] = os.path.join(dep_root, img_tag + '.png')
        name_list.append(tag_dict)
    
    return name_list
    
def get_rgbt_list(name, config, phase):    
    name_list = []
    image_root = os.path.join(config['data_path'], 'RGBT/{}/RGB'.format(name))
    th_root = os.path.join(config['data_path'], 'RGBT/{}/T'.format(name))
    
    if config['stage'] > 1 and phase == 'train':
        gt_root = './pseudo/t-split/{}'.format(name)
        print('Using pseudo labels from {}'.format(gt_root))
        
        if config['olr']:
            gt_orig, gt_root = OLR(gt_root, name)
    else:
        gt_root = os.path.join(config['data_path'], 'RGBT/{}/GT'.format(name))
    
    img_list = os.listdir(image_root)
    for img_name in img_list:
        img_tag = img_name.split('.')[0]
        
        tag_dict = {}
        tag_dict['rgb'] = os.path.join(image_root, img_name)
        tag_dict['gt'] = os.path.join(gt_root, img_tag + '.png')
        tag_dict['th'] = os.path.join(th_root, img_tag + '.jpg')
        name_list.append(tag_dict)
    
    return name_list
    

def get_frame_list(name, config, phase):
    name_list = []
    
    base_path = os.path.join(config['data_path'], 'vsod', name)
    videos = os.listdir(os.path.join(base_path, 'JPEGImages'))
        
    if config['stage'] > 1 and phase == 'train':
        gt_base = './pseudo/o-joint/{}'.format(name)
        print('Using pseudo labels from {}'.format(gt_base))
    
        if config['olr']:
            gt_orig, gt_base = OLR(gt_base, name)
    else:
        gt_base = os.path.join(base_path, 'Annotations')
    
    
    for video in videos:
        image_root = os.path.join(base_path, 'JPEGImages', video)
        gt_root = os.path.join(gt_base, video)
        of_root = os.path.join(base_path, 'optical', video)
        
        img_list = os.listdir(image_root)
        img_list = sorted(img_list)
        if phase == 'train' and config['stage'] == 1 and 'select' in video:
            img_list = img_list[::5]
        
        for img_name in img_list:
            img_tag = img_name.split('.')[0]
            
            tag_dict = {}
            tag_dict['rgb'] = os.path.join(image_root, img_name)
            tag_dict['gt'] = os.path.join(gt_root, img_tag + '.png')
            tag_dict['of'] = os.path.join(of_root, img_tag + '.jpg')
            name_list.append(tag_dict)
         
    
    return name_list

def get_train_image_list(names, config):
    image_list = []
    phase = 'train'
    
    ccc, ddd, ooo, ttt = [], [], [], []
    if 'c' in names:
        ccc = get_color_list('DUTS-TR', config, phase)
        image_list += ccc
    if 'd' in names:
        ddd = get_rgbd_list('RGBD-TR', config, phase)
        image_list += ddd
    if 'o' in names:
        ooo = get_frame_list('VSOD-TR', config, phase)
        image_list += ooo
    if 't' in names:
        ttt = get_rgbt_list('VT5000-TR', config, phase)
        image_list += ttt

    print('Loading {} images for {}: RGB({}), RGBD({}), VSOD({}), RGBT({}).'.format(len(image_list), phase, len(ccc), len(ddd), len(ooo), len(ttt)))
    return image_list

def get_test_list(modes='cr', config=None):
    test_dataset = OrderedDict()
    
    for mode in modes:
        modal, subset = mode
        if subset == 'e':
            if modal == 'c':
                test_list = ['PASCAL-S', 'ECSSD', 'MSB-TE', 'DUTS-TE', 'HKU-IS', 'DUT-OMRON']
            elif modal == 'd':
                #test_list = ['DUT', 'LFSD', 'NJUD', 'NLPR', 'RGBD135', 'SIP', 'SSD', 'STERE1000']
                test_list = ['NJUD', 'NLPR', 'RGBD135', 'SIP']
            elif modal == 'o':
                test_list = ['FBMS', 'SegV2', 'DAVIS-TE', 'DAVSOD-TE']
            elif modal == 't':
                test_list = ['VT5000-TE', 'VT1000', 'VT821']
                    
            for test_set in test_list:
                set_name = '_'.join((modal, test_set))
                test_dataset[set_name] = Test_Dataset(test_set, modal, config)
        else:
            if modal == 'c':
                trset = 'DUTS-TR'
            elif modal == 'd':
                trset = 'RGBD-TR'
            elif modal == 'o':
                trset = 'VSOD-TR'
            elif modal == 't':
                trset = 'VT5000-TR'
            
            set_name = '_'.join((modal, trset))
            test_dataset[set_name] = Test_Dataset(trset, modal, config)
        
    return test_dataset
    

def get_loader(config):
    dataset = Train_Dataset(config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['batch'],
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True,
                                  drop_last=True)
    return data_loader

def read_modality(sub, sample_path, flip, img_size):
    if sub in sample_path.keys():
        m_name = sample_path[sub]
        modal = Image.open(m_name).convert('RGB')
        modal = modal.resize((img_size, img_size))
        modal = np.array(modal).astype(np.float32) / 255.
        if flip:
            modal = modal[:, ::-1].copy()
        modal = modal.transpose((2, 0, 1))
    else:
        modal = np.zeros((3, img_size, img_size)).astype(np.float32)
        
    return modal
    

class Train_Dataset(data.Dataset):
    def __init__(self, config):
        self.config = config
        self.modality = config['trset']
        self.image_list = get_train_image_list(self.modality, config)
        self.size = len(self.image_list)

    def __getitem__(self, index):
        sample_path = self.image_list[index]
        img_name = sample_path['rgb']
        gt_name = sample_path['gt']
        
        image = Image.open(img_name).convert('RGB')
        gt = Image.open(gt_name).convert('L')
        
        img_size = self.config['size']
        image = image.resize((img_size, img_size))
        gt = gt.resize((img_size, img_size), Image.NEAREST)
    
        image = np.array(image).astype(np.float32)
        gt = np.array(gt)
        
        flip = random.random() > 0.5
        if flip:
            image = image[:, ::-1].copy()
            gt = gt[:, ::-1].copy()
        
        image = ((image / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-5)
        
        gt = np.expand_dims(gt, axis=0)
        
        out_dict = {'image': image, 'gt': gt, 'name': gt_name, 'flip': flip}
        
        for modality in self.modality:
            if modality == 'c':
                continue
            elif modality == 'd':
                sub = 'dep'
                out_dict[sub] = read_modality(sub, sample_path, flip, img_size)
            elif modality == 'o':
                sub = 'of'
                out_dict[sub] = read_modality(sub, sample_path, flip, img_size)
            elif modality == 't':
                sub = 'th'
                out_dict[sub] = read_modality(sub, sample_path, flip, img_size)
        
        return out_dict

    def __len__(self):
        return self.size

class Test_Dataset:
    def __init__(self, name, mode, config=None):
        self.config = config
        
        read_list = None
        if mode == 'c':
            read_list = get_color_list
        elif mode == 'd':
            read_list = get_rgbd_list
        elif mode == 'o':
            read_list = get_frame_list
        elif mode == 't':
            read_list = get_rgbt_list
        self.image_list = read_list(name, config, 'test')
        
        self.set_name = name
        self.size = len(self.image_list)
        self.modality = mode

    def load_data(self, index):
        sample_path = self.image_list[index]
        img_name = sample_path['rgb']
        gt_name = sample_path['gt']
        
        image = Image.open(img_name).convert('RGB')
        image = image.resize((self.config['size'], self.config['size']))
        image = np.array(image).astype(np.float32)
        gt = Image.open(gt_name).convert('L')
        img_size = self.config['size']
        
        img_pads = img_name.split('/')
        name = '/'.join(img_pads[img_pads.index(self.set_name) + 2:])
        
        image = ((image / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-5)
        
        out_dict = {'image': image, 'gt': gt, 'name': name}
        
        for modality in self.modality:
            if modality == 'c':
                continue
            elif modality == 'd':
                sub = 'dep'
            elif modality == 'o':
                sub = 'of'
            elif modality == 't':
                sub = 'th'
            out_dict[sub] = read_modality(sub, sample_path, False, img_size)
        return out_dict

def test_data():
    config = {'orig_size': True, 'size': 288, 'data_path': '../dataset'}
    dataset = 'SOD'
    
    '''
    data_loader = Test_Dataset(dataset, config)
    #data_loader = Train_Dataset(dataset, config)
    data_size = data_loader.size
    
    for i in range(data_size):
        img, gt, name = data_loader.load_data(i)
        #img, gt = data_loader.__getitem__(i)
        new_img = (img * std + mean) * 255.
        #new_img = gt * 255
        print(np.min(new_img), np.max(new_img))
        new_img = (new_img).astype(np.uint8)
        #print(new_img.shape).astype(np.)
        im = Image.fromarray(new_img)
        #im.save('temp/' + name + '.jpg')
        im.save('temp/' + str(i) + '.jpg')
    
    '''
    
    data_loader = Val_Dataset(dataset, config)
    imgs, gts, names = data_loader.load_all_data()
    print(imgs.shape, gts.shape, len(names))
    

if __name__ == "__main__":
    test_data()