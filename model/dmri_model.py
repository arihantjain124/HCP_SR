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
    def __init__(self,args):
        super().__init__()
        self.encoder = make_rdn_2d(args)
        self.decoder = ImplicitDecoder_2d(args) 
    
    def forward(self, inp,scale,rel_coor):
        
        B,C,H,W = inp.shape
        
        size = rel_coor.shape[2:]   

        # H_hr = round(H*scale[0])
        # W_hr = round(W*scale[1])
        
        # size = [H_hr, W_hr]


        # print(rel_coor.shape,inp.shape,size,scale)
        feat = self.encoder(inp)
        
        pred = self.decoder(feat,size,rel_coor)
        
        return pred
