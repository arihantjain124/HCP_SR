import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common
from argparse import Namespace
import random 
import math
from model.rdn import make_rdn
from model.resblock import ResBlock
import unfoldNd

class SineAct(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # return x
        return torch.sin(x)

class ImplicitDecoder(nn.Module):
    def __init__(self, in_channels=16, hidden_dims=[64,32,16,16,8], output_dim = 5):
        super().__init__()

        self.Q = nn.ModuleList()
        last_dim_Q = in_channels * 27
        for hidden_dim in hidden_dims:
            self.Q.append(nn.Sequential(nn.Conv3d(last_dim_Q, hidden_dim, 1),
                                        SineAct()))
            last_dim_Q = hidden_dim

        self.in_branch = nn.Sequential(nn.Conv3d(in_channels * 27, hidden_dims[-2], 1),
                            nn.ReLU(),
                            nn.Conv3d(hidden_dims[-2],hidden_dims[-1], 1),
                            nn.ReLU())
        
        self.mixer = nn.Sequential(nn.Conv3d(hidden_dims[-1] * 2, hidden_dims[-1], 1),
                                   nn.ReLU())

        self.last_layer = nn.Conv3d(hidden_dims[-1], output_dim, 1)

    def step(self, x):
        lat = x
        for i in range(len(self.Q)):
            lat = self.Q[i](lat)
        feat = self.in_branch(x)

        feat = torch.cat([lat,feat],dim=1)
        feat = self.mixer(feat)
        return self.last_layer(feat)

    def forward(self, x, size):
        B, C, H_in, W_in,D_in = x.shape
        # print(x.shape)
        ratio = (x.new_tensor([math.sqrt((H_in*W_in*D_in)/(size[0]*size[1]*size[2]))]).view(1, -1, 1, 1,1).expand(B, -1, *size))
        x = F.interpolate(unfoldNd.unfoldNd(x, 3, padding=1).view(B, C*27, H_in, W_in,D_in), size=ratio.shape[-3:],mode = "trilinear")
        
        # x = unfoldNd.unfoldNd(x, 3, padding=1).view(B, C*27, H_in, W_in,D_in)
        # print(x.shape,H_in, W_in,D_in,ratio.shape)
        return self.step(x)

class DMRI_SR(nn.Module):
    def __init__(self,inch = 7):
        super().__init__()
        self.encoder = make_rdn(inchannel=inch)
        self.decoder = ImplicitDecoder()
    
    def set_scale(self, scale):
        self.scale = scale

    def forward(self, inp):
        
        B,C,H,W,D = inp.shape
        # print(self.scale)
        H_hr = round(H*self.scale[0])
        W_hr = round(W*self.scale[1])
        D_hr = round(D*self.scale[2])
        
        size = [H_hr, W_hr,D_hr]
        
        feat = self.encoder(inp)
        # latent = self.latent_layer(feat)
        pred = self.decoder(feat,size)
        
        return pred

