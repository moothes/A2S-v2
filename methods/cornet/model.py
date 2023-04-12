import torch
from torch import nn
from torch.nn import functional as F

mode = 'bilinear' # 'nearest' # 

def up_conv(cin, cout, up=True):
    yield nn.Conv2d(cin, cout, 3, padding=1)
    yield nn.GroupNorm(cout//2, cout)
    yield nn.ReLU(inplace=True)
    if up:
        yield nn.Upsample(scale_factor=2, mode='bilinear')

def local_conv(cin, cout):
    yield nn.Conv2d(cin, cout * 2, 3, padding=1)
    yield nn.GroupNorm(cout, cout * 2)
    yield nn.ReLU(inplace=True)
    yield nn.Upsample(scale_factor=2, mode='bilinear')
    yield nn.Conv2d(cout * 2, cout, 3, padding=1)
    yield nn.GroupNorm(cout//2, cout)
    yield nn.ReLU(inplace=True)
    yield nn.Upsample(scale_factor=2, mode='bilinear')

class info_block(nn.Module):
    def __init__(self, config, feat, tar_feat):
        super(info_block, self).__init__()
        self.gconv = nn.Sequential(*list(up_conv(tar_feat, tar_feat, False)))
        self.res_conv1 = nn.Conv2d(tar_feat, tar_feat, 3, padding=1)
        self.res_conv2 = nn.Conv2d(tar_feat, tar_feat, 3, padding=1)
        self.fuse = nn.Conv2d(tar_feat * 3, tar_feat, 3, padding=1)

    def forward(self, xs, glob_x):
        glob_x0 = nn.functional.interpolate(self.gconv(glob_x), size=xs[0].size()[2:], mode=mode)
        loc_x1 = xs[0]
        
        # RAM
        res_x1 = torch.sigmoid(self.res_conv1(loc_x1 - glob_x0)) #
        loc_x2 = nn.functional.interpolate(xs[1], size=xs[0].size()[2:], mode=mode)
        res_x2 = torch.sigmoid(self.res_conv2(loc_x2 - glob_x0)) #
        loc_x = self.fuse(torch.cat([loc_x1 * res_x1, loc_x2 * res_x2, glob_x0], dim=1))
        return loc_x, res_x1, res_x2

class decoder(nn.Module):
    def __init__(self, config, encoder, feat):
        super(decoder, self).__init__()
        
        self.adapter0 = nn.Sequential(*list(up_conv(feat[0], feat[0], False)))
        self.adapter1 = nn.Sequential(*list(up_conv(feat[1], feat[0], False)))
        self.adapter2 = nn.Sequential(*list(up_conv(feat[2], feat[0], False)))
        self.adapter3 = nn.Sequential(*list(up_conv(feat[3], feat[0], False)))
        self.adapter4 = nn.Sequential(*list(up_conv(feat[4], feat[0], False)))
        
        self.region = info_block(config, feat[2:4], feat[0])
        self.local = info_block(config, feat[0:2], feat[0])
        
        self.gb_conv = nn.Sequential(*list(local_conv(feat[0], feat[0])))
        
    def forward(self, xs, x_size):
        xs[0] = self.adapter0(xs[0])
        xs[1] = self.adapter1(xs[1])
        xs[2] = self.adapter2(xs[2])
        xs[3] = self.adapter3(xs[3])
        xs[4] = self.adapter4(xs[4])
        
        glob_x = xs[4]
        reg_x, r3, r4 = self.region(xs[2:4], glob_x)
        
        glob_x = self.gb_conv(glob_x)
        loc_x, r1, r2 = self.local(xs[0:2], glob_x)

        reg_x = nn.functional.interpolate(reg_x, size=xs[0].size()[2:], mode=mode)
        pred = torch.sum(loc_x * reg_x, dim=1, keepdim=True)
        pred = nn.functional.interpolate(pred, size=x_size, mode='bilinear')

        OutDict = {}
        OutDict['sal'] = [pred, ]
        OutDict['final'] = pred
        return OutDict

class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()
        self.encoder = encoder
        self.decoder = decoder(config, encoder, feat)

    def forward(self, x, phase='test'):
        x = x[0]
        x_size = x.size()[2:]
        xs = self.encoder(x)
        out = self.decoder(xs, x_size)
        return out
