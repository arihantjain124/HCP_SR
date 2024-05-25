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
    def __init__(self,in_chans = 7,int_chans = 32,encoder_type = 'rdb',drop_prob = 0,tv = False,attn = False):
        super().__init__()
        self.encoder = make_rdn(in_chans=in_chans,growth = int_chans,enc= encoder_type,drop_prob=drop_prob,attn = attn)
        # print(self.encoder)
        self.decoder = ImplicitDecoder_3d(in_channels= int_chans,tv = tv)
        self.tv = tv


    def forward(self, inp,scale,rel_coor):

        B,C,H,W,D = inp.shape
        scale = np.asarray(scale)
        H_hr = round(H*scale[0])
        W_hr = round(W*scale[1])
        D_hr = round(D*scale[2])
        
        size = [H_hr, W_hr,D_hr]
        
        feat = self.encoder(inp)
        
        pred = self.decoder(feat,size,rel_coor)
        
        return pred
        # if self.tv:
        #     return (pred[0]*0.5+0.5),(pred[1]*0.5+0.5)
        # else:
        #     return (pred*0.5+0.5)
    
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