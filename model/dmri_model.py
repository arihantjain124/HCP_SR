import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
import random 
import math
from model.rdn import make_rdn
from model.rdn_2d import make_rdn as make_rdn_2d
from model.resblock import ResBlock
import unfoldNd
from model.models import CSEUnetModel_3d,CSEUnetModel
from model.arb_decoder import ImplicitDecoder_3d,ImplicitDecoder_2d



class DMRI_arb(nn.Module):
    def __init__(self,in_chans = 7,int_chans = 32):
        super().__init__()
        self.encoder = make_rdn(in_chans=in_chans,growth = int_chans,enc= 'conv')
        self.decoder = ImplicitDecoder_3d(in_channels= int_chans)
    
    def set_scale(self, scale):
        self.scale = scale

    def forward(self, inp):
        
        B,C,H,W,D = inp.shape
        
        H_hr = round(H*self.scale[0])
        W_hr = round(W*self.scale[1])
        D_hr = round(D*self.scale[2])
        
        size = [H_hr, W_hr,D_hr]
        
        feat = self.encoder(inp)
        
        pred = self.decoder(feat,size)
        
        return pred
    

class DMRI_RCAN_3d(nn.Module):
    def __init__(self,in_chans = 7,int_chans = 32):
        super().__init__()
        self.encoder = CSEUnetModel_3d(in_chans=in_chans,out_chans=32,chans=int_chans,num_pool_layers=3,drop_prob = 0,attention_type='cSE',reduction=16)
        
        self.decoder = ImplicitDecoder_3d(in_channels= 32)
    
    def set_scale(self, scale):
        self.scale = scale

    def forward(self, inp):
        
        B,C,H,W,D = inp.shape
        
        H_hr = round(H*self.scale[0])
        W_hr = round(W*self.scale[1])
        D_hr = round(D*self.scale[2])
        
        size = [H_hr, W_hr,D_hr]
        
        feat = self.encoder(inp)
        
        pred = self.decoder(feat,size)
        
        return pred
    

class DMRI_RCAN_2d(nn.Module):
    def __init__(self,in_chans = 7,int_chans = 32):
        super().__init__()
        self.encoder = CSEUnetModel(in_chans=in_chans,out_chans=64,chans=int_chans,num_pool_layers=3,drop_prob = 0,attention_type='cSE',reduction=16)
        self.decoder = ImplicitDecoder_2d(in_channels= 64) 
    
    def set_scale(self, scale):
        self.scale = scale

    def forward(self, inp):
        
        B,C,H,W = inp.shape
        # print(self.scale)
        H_hr = round(H*self.scale[0])
        W_hr = round(W*self.scale[1])
        
        size = [H_hr, W_hr]
        
        feat = self.encoder(inp)
        
        
        pred = self.decoder(feat,size)
        
        return pred


class DMRI_RDN_3d(nn.Module):
    def __init__(self,inch = 7,growth = 16):
        super().__init__()
        self.encoder = make_rdn(in_chans=inch,growth = growth)
        self.decoder = ImplicitDecoder_3d(in_channels= growth)
    
    def set_scale(self, scale):
        self.scale = scale

    def forward(self, inp):
        
        B,C,H,W,D = inp.shape
        
        H_hr = round(H*self.scale[0])
        W_hr = round(W*self.scale[1])
        D_hr = round(D*self.scale[2])
        
        size = [H_hr, W_hr,D_hr]
        
        feat = self.encoder(inp)
        
        pred = self.decoder(feat,size)
        
        return pred
      
    
class DMRI_RDN_2d(nn.Module):
    def __init__(self,inch = 7,growth = 16):
        super().__init__()
        self.encoder = make_rdn_2d(in_chans=inch,growth = growth)
        self.decoder = ImplicitDecoder_2d(in_channels= growth) 
    
    def set_scale(self, scale):
        self.scale = scale

    def forward(self, inp):
        
        B,C,H,W = inp.shape
        # print(self.scale)
        H_hr = round(H*self.scale[0])
        W_hr = round(W*self.scale[1])
        
        size = [H_hr, W_hr]
        
        feat = self.encoder(inp)
        
        pred = self.decoder(feat,size)
        
        return pred