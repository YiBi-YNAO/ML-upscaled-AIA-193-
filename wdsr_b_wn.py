import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


#def make_model(args, parent=False):
 #   return WDSR_B(args)


class Block(nn.Module):
    def __init__(
            self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2)))  # channels：64->64*6
        body.append(act)
        body.append(                                                  # channels：64*6->64*0.8
            wn(nn.Conv2d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(                                                  # channels：64*0.8->64
            wn(nn.Conv2d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


"""
    标准化输入
    head： 卷积×1
    body： resblock(卷积+relu+卷积×2) *16
    tail： 卷积×1
    skip： 卷积×1
    标准化输出
    
    上采样：一次卷积将channels提升到足够的倍数（feats×n×2），然后直接转换为所需图片
"""
class WDSR_B(nn.Module):
    def __init__(self, scale,n_resblocks,n_feats,n_colors,res_scale):
        super(WDSR_B, self).__init__()
        # hyper-params
        #self.args = args
        #scale = args.scale[0]
        #n_resblocks = args.n_resblocks
        #n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        #wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)                  # 激活函数：weight_norm?

        #self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
         #   [args.r_mean, args.g_mean, args.b_mean])).view([1, 3, 1, 1])

        # define head module
        head = []
        head.append(
            wn(nn.Conv2d(n_colors, n_feats, 3, padding=3//2)))   # channels：3->64

        # define body module
        body = []
        for i in range(n_resblocks):                                  # 16个resblock
            body.append(                                              # ## 参数：64, 3, relu, 1, wn
                Block(n_feats, kernel_size, act=act, res_scale=res_scale, wn=wn))

        # define tail module
        tail = []
        out_feats = scale*scale*n_colors  # 4*4*3
        tail.append(
            wn(nn.Conv2d(n_feats, out_feats, 3, padding=3//2)))       # channels:64->4*4*3
        tail.append(nn.PixelShuffle(scale))

        skip = []
        skip.append(
            wn(nn.Conv2d(n_colors, out_feats, 5, padding=5//2))  # channels：3->4*4*3
        )
        skip.append(nn.PixelShuffle(scale))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
       # x = (x - self.rgb_mean.cuda()*255)/127.5
        x=x/4000.
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        x=x*4000.
        #x = x*127.5 + self.rgb_mean.cuda()*255
        return x

