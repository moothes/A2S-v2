import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet
import torch
from torch import nn
import torch.nn.functional as F
from itertools import chain
from torchvision.models import resnet50

def convblock(in_,out_,ks,st,pad):
    return nn.Sequential(
        nn.Conv2d(in_,out_,ks,st,pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

class GFB(nn.Module):
    def __init__(self,in_1,in_2):
        super(GFB, self).__init__()
        self.ca1 = CA(2*in_1)
        self.conv1 = convblock(2*in_1,128, 3, 1, 1)
        self.conv_globalinfo = convblock(512,128,3, 1, 1)
        self.ca2 = CA(in_2)
        self.conv_curfeat =convblock(in_2,128,3,1,1)
        self.conv_out= convblock(128,in_2,3,1,1)

    def forward(self, pre1,pre2,cur,global_info):
        cur_size = cur.size()[2:]
        pre = self.ca1(torch.cat((pre1,pre2),1))
        pre =self.conv1(F.interpolate(pre,cur_size,mode='bilinear',align_corners=True))

        global_info = self.conv_globalinfo(F.interpolate(global_info,cur_size,mode='bilinear',align_corners=True))
        cur_feat =self.conv_curfeat(self.ca2(cur))
        fus = pre + cur_feat + global_info
        return self.conv_out(fus)

        
class GlobalInfo(nn.Module):
    def __init__(self):
        super(GlobalInfo, self).__init__()
        self.ca = CA(1024)
        self.de_chan = convblock(1024,256,3,1,1)
        
        self.b0 = nn.Sequential(
            nn.AdaptiveMaxPool2d(13),
            nn.Conv2d(256,128,1,1,0,bias=False),
            nn.ReLU(inplace=True)
        )

        self.b1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(9),
            nn.Conv2d(256,128,1,1,0,bias=False),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(5),
            nn.Conv2d(256, 128, 1, 1, 0,bias=False),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(256, 128, 1, 1, 0,bias=False),
            nn.ReLU(inplace=True)
        )
        self.fus = convblock(768,512,1,1,0)
        
    def forward(self, rgb,t):
        x_size=rgb.size()[2:]
        x=self.ca(torch.cat((rgb,t),1))
        x=self.de_chan(x)
        b0 = F.interpolate(self.b0(x),x_size,mode='bilinear',align_corners=True)
        b1 = F.interpolate(self.b1(x),x_size,mode='bilinear',align_corners=True)
        b2 = F.interpolate(self.b2(x),x_size,mode='bilinear',align_corners=True)
        b3 = F.interpolate(self.b3(x),x_size,mode='bilinear',align_corners=True)
        out = self.fus(torch.cat((b0,b1,b2,b3,x),1))
        return out
        
class CA(nn.Module):
    def __init__(self,in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()
    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)

class FinalScore(nn.Module):
    def __init__(self):
        super(FinalScore, self).__init__()
        self.ca =CA(128)
        self.score = nn.Conv2d(128, 1, 1, 1, 0)
    def forward(self,f1,f2):
        f1 = torch.cat((f1,f2),1)
        f1 = self.ca(f1)
        score = self.score(f1)
        return score

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.global_info =GlobalInfo()
        self.score_global = nn.Conv2d(512, 1, 1, 1, 0)

        self.gfb4_1 = GFB(512,512)
        self.gfb3_1= GFB(512,256)
        self.gfb2_1= GFB(256,128)
        self.gfb1_1 = GFB(128, 64)  # 1/2

        self.gfb4_2 = GFB(512, 512) #1/16
        self.gfb3_2 = GFB(512, 256)#1/8
        self.gfb2_2 = GFB(256, 128)#1/4
        self.gfb1_2 = GFB(128, 64)  # 1/2

        self.score_1=nn.Conv2d(64, 1, 1, 1, 0)
        self.score_2 = nn.Conv2d(64, 1, 1, 1, 0)

        self.refine =FinalScore()


    def forward(self,rgb,t):
        xsize=rgb[0].size()[2:]
        global_info =self.global_info(rgb[4],t[4]) # 512 1/16
        d1=self.gfb4_1(global_info,t[4],rgb[3],global_info)
        d2=self.gfb4_2(global_info, rgb[4], t[3], global_info)
        #print(d1.shape,d2.shape)
        d3= self.gfb3_1(d1, d2,rgb[2],global_info)
        d4 = self.gfb3_2(d2, d1, t[2], global_info)
        d5 = self.gfb2_1(d3, d4, rgb[1], global_info)
        d6 = self.gfb2_2(d4, d3, t[1], global_info) #1/2 128
        d7 = self.gfb1_1(d5, d6, rgb[0], global_info)
        d8 = self.gfb1_2(d6, d5, t[0], global_info)  # 1/2 128

        score_global = self.score_global(global_info)

        score1=self.score_1(d7)
        score2 = self.score_2(d8)
        score =self.refine(d7,d8)
        
        OutDict = {}
        OutDict['sal'] = [score,score1,score2,score_global]
        OutDict['final'] = score
        return OutDict


class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()

        self.encoder = encoder
        self.t_net = resnet(pretrained=config['pretrain'])
        
        trans_layers_mapping = [[256,128],[512,256],[1024,512],[2048,512]]
        self.trans_rgb = nn.ModuleList()
        self.trans_t = nn.ModuleList()
        for mapp in trans_layers_mapping:
            self.trans_rgb.append(convblock(mapp[0],mapp[1],1,1,0))
            self.trans_t.append(convblock(mapp[0], mapp[1], 1, 1, 0))
        self.decoder = Decoder()#decoder(config, encoder, feat)
        
        for m in chain(self.decoder.modules(),chain(self.trans_rgb.modules(),self.trans_t.modules())):
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, phase='test'):
        rgb, rgbt = x
        x_size = rgb.size()[2:]
        rgb_f = self.encoder(rgb)
        for i in range(1,5):
            rgb_f[i] = self.trans_rgb[i-1](rgb_f[i])
        t_f = self.t_net(rgbt)
        #t_f = self.encoder(rgbt)
        for i in range(1,5):
            t_f[i] = self.trans_t[i-1](t_f[i])
            
        out = self.decoder(rgb_f, t_f)
        return out
