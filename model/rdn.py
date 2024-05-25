# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from argparse import Namespace

import torch
import torch.nn as nn
import numpy as np
from model.attention import cSELayer_3d,scSELayer

def make_rdn(in_chans=7, RDNkSize=3, growth = 16, RDNconfig='C',enc = 'rdb',drop_prob = 0,attn = False):
    args = Namespace()
    args.G0 = growth
    args.RDNkSize = RDNkSize
    args.RDNconfig = RDNconfig
    args.drop_prob = drop_prob
    args.attn = attn

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

class ConvBlock_3d(nn.Module):
    def __init__(self, in_chans, out_chans, drop_prob = 0):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(),
            nn.Dropout3d(drop_prob)
        )


                  
    def forward(self, input):

        out = self.layers(input)
        return torch.cat((input, out), 1)



class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers,encoder,drop_prob = 0,attention_type = 'cSE',reduction = 16,attention = False):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            if(encoder == 'rdb'):
                convs.append(RDB_Conv(G0 + c*G, G))
            else:
                convs.append(ConvBlock_3d(G0 + c*G, G,drop_prob = drop_prob))
        
        
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
            'A': (3, 5, 16),
            'B': (4, 7, 32),
            'C': (5, 8, 32),
            'D': (8, 8, 32)
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv3d(args.n_colors, self.G0 , kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv3d(self.G0 , self.G0 , kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = self.G0 , growRate = G, nConvLayers = C,encoder = encoder,drop_prob = self.args.drop_prob,attention = args.attn)
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
