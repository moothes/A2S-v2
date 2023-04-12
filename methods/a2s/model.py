import torch
from torch import nn
from torch.nn import functional as F


def up_conv(cin, cout):
    yield nn.Conv2d(cin, cout, 3, padding=1)
    yield nn.GroupNorm(1, cout)
    yield nn.ReLU(inplace=True)
    
def foreground_sign(pred):
    b, c, w, h = pred.size()
    p = pred.gt(0).float()
    num_pos = p[:, :, 0, 0] + p[:, :, w-1, 0] + p[:, :, w-1, h-1] + p[:, :, 0, h-1]
    sign = ((num_pos < 2).float() * 2 - 1).view(b, c, 1, 1)
    return sign


class SE_block(nn.Module):
    def __init__(self, feat):
        super(SE_block, self).__init__()
        self.conv = nn.Conv2d(feat, feat, 1)
        self.gn = nn.GroupNorm(feat // 2, feat)

    def forward(self, x):
        glob_x = F.adaptive_avg_pool2d(x, (1, 1))
        glob_x = torch.sigmoid(self.conv(glob_x))
        x = glob_x * x
        return x

class ada_block(nn.Module):
    def __init__(self, config, feat, out_feat=64):
        super(ada_block, self).__init__()
        
        self.ad0 = nn.Sequential(*list(up_conv(feat, out_feat)))
        self.se = SE_block(out_feat)

    def forward(self, x):
        x = self.ad0(x)
        x = self.se(x)
        return x

def normalize(x):
    center = torch.mean(x, dim=(2, 3), keepdim=True)
    x = x - center
    return x

class decoder(nn.Module):
    def __init__(self, config, encoder, feat):
        super(decoder, self).__init__()
        
        self.ad2 = ada_block(config, feat[2], feat[0])
        self.ad3 = ada_block(config, feat[3], feat[0])
        self.ad4 = ada_block(config, feat[4], feat[0])
        self.fusion = ada_block(config, feat[0] * 3, feat[0])
        
    def forward(self, xs, x_size, phase='test'):
        x2 = self.ad2(xs[2])
        x3 = self.ad3(xs[3])
        x4 = self.ad4(xs[4])
        
        x2u = nn.functional.interpolate(x2, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x3u = nn.functional.interpolate(x3, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x4u = nn.functional.interpolate(x4, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        
        fuse = torch.cat([x2u, x3u, x4u], dim=1)
        feat = self.fusion(fuse)
        feat = normalize(feat)
        
        pred = torch.sum(feat, dim=1, keepdim=True)
        
        # Sign function
        pred = pred * foreground_sign(pred)
        pred = nn.functional.interpolate(pred, size=x_size, mode='bilinear', align_corners=True)
        
        OutDict = {}
        OutDict['feat'] = [feat, ]
        OutDict['sal'] = [pred, ]
        OutDict['final'] = pred
        
        return OutDict

class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        # encoder: backbone, forward function output 5 encoder features. details in methods/base/model.py
        # feat: length of encoder features. e.g.: VGG:[64, 128, 256, 512, 512]; Resnet:[64, 256, 512, 1024, 2048]
        super(Network, self).__init__()
        self.encoder = encoder
        self.decoder = decoder(config, encoder, feat)

    def forward(self, x, phase='test'):
        x_size = x.size()[2:]
        xs = self.encoder(x)
        out = self.decoder(xs, x_size, phase)
        return out
