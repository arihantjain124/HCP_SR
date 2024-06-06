import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

import math
from model.rdn import make_rdn
from model.rdn_2d import make_rdn as make_rdn_2d
from model.resblock import ResBlock
from model.arb_decoder import ImplicitDecoder_3d,ImplicitDecoder_2d
import numpy as np


class DMRI_arb(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.encoder = make_rdn(args)
        # print(self.encoder)
        self.decoder = ImplicitDecoder_3d(args)
        self.tv = args.tv


    def forward(self, inp,scale,rel_coor):

        B,C,H,W,D = inp.shape
        scale = np.asarray(scale)
        # print(scale)
        H_hr = round(H*float(scale[0]))
        W_hr = round(W*float(scale[1]))
        D_hr = round(D*float(scale[2]))
        
        size = [H_hr, W_hr,D_hr]
        
        feat = self.encoder(inp)
        
        return self.decoder(feat,size,rel_coor)

class DMRI_arb_2d(nn.Module):
    def __init__(self,inch = 7,growth = 16):
        super().__init__()
        self.encoder = make_rdn_2d(in_chans=inch,growth = growth)
        self.decoder = ImplicitDecoder_2d(in_channels= growth) 
    
    def set_scale(self, scale):
        self.scale = scale

    def forward(self, inp):
        
        B,C,H,W = inp.shape

        H_hr = round(H*self.scale[0])
        W_hr = round(W*self.scale[1])
        
        size = [H_hr, W_hr]
        
        feat = self.encoder(inp)
        
        pred = self.decoder(feat,size)
        
        return pred
