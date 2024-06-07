# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from argparse import Namespace

import torch
import torch.nn as nn
import numpy as np
from model.attention import cSELayer_3d,scSELayer

def make_rdn(arg, RDNkSize=3):
    
    args = Namespace()
    args.G0 = arg.growth
    args.RDNkSize = RDNkSize
    args.RDNconfig = arg.RDNconfig
    args.attn = arg.attention
    args.enc = arg.encoder

    args.n_colors = arg.in_chans
    return RDN(args)

class ConvBlock_3d(nn.Module):
    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_chans,affine=True),
            nn.LeakyReLU()
        )


                  
    def forward(self, input):

        out = self.layers(input)
        return torch.cat((input, out), 1)



class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers,attention_type = 'cSE',reduction = 16,attention = False):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(ConvBlock_3d(G0 + c*G, G))
        
        
        if attention:
            if attention_type == 'cSE':
                self.attention = cSELayer_3d(channel=G0,reduction=reduction)
            if attention_type == 'scSE':
                self.attention = scSELayer(channel=G0,reduction=reduction)
        else:
            self.attention = None
            
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv3d(G0 + C*G, G0, 1, padding=0, stride=1)  

    def forward(self, x):
        if self.attention != None:
            return self.attention(self.LFF(self.convs(x)) + x)
        else:
            return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self, args,encoder = 'rdb'):
        super(RDN, self).__init__()
        self.args = args
        self.G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
            'C': (5, 8, 32),
            'D': (5, 10, 64)
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv3d(args.n_colors, self.G0 , kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv3d(self.G0 , self.G0 , kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = self.G0 , growRate = G, nConvLayers = C,attention = args.attn)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv3d(self.D * self.G0 , self.G0 , 1, padding=0, stride=1),
            nn.Conv3d(self.G0 , self.G0 , kSize, padding=(kSize-1)//2, stride=1)
        ])

        self.out_dim = self.G0

    def forward(self, x):
        
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        return x
