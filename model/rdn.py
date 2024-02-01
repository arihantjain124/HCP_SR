# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from argparse import Namespace

import torch
import torch.nn as nn
import numpy as np
from model.models import ConvBlock_3d

def make_rdn(in_chans=7, RDNkSize=3, growth = 16, RDNconfig='C',enc = 'rdb'):
    args = Namespace()
    args.G0 = growth
    args.RDNkSize = RDNkSize
    args.RDNconfig = RDNconfig

    args.n_colors = in_chans
    return RDN(args,enc)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv3d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.LeakyReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers,encoder, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            if(encoder == 'rdb'):
                convs.append(RDB_Conv(G0 + c*G, G))
            else:
                convs.append(ConvBlock_3d(G0 + c*G, G))
        
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv3d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self, args,encoder = 'rdn'):
        super(RDN, self).__init__()
        self.args = args
        self.G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (3, 4, 16),
            'B': (4, 6, 32),
            'C': (5, 8, 64),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv3d(args.n_colors, self.G0 , kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv3d(self.G0 , self.G0 , kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = self.G0 , growRate = G, nConvLayers = C,encoder = encoder)
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