import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common
from argparse import Namespace
import random 
import math
from model.rdn import make_rdn
from model.resblock import ResBlock


class SineAct(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)

def patch_norm_2d(x, kernel_size=3):
    mean = F.avg_pool2d(x, kernel_size=kernel_size, padding=kernel_size//2)
    mean_sq = F.avg_pool2d(x**2, kernel_size=kernel_size, padding=kernel_size//2)
    var = mean_sq - mean**2
    return (x-mean)/(var + 1e-6)

class ImplicitDecoder(nn.Module):
    def __init__(self, in_channels=64, hidden_dims=[16,16,16,16,16]):
        super().__init__()

        last_dim_K = in_channels * 9
        
        last_dim_Q = 4

        self.K = nn.ModuleList()
        self.Q = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim*2, 1),
                                        nn.ReLU(),
                                        ResBlock(channels = hidden_dim*2, nConvLayers = 4)
                                        ))    
            self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                        SineAct()))
            last_dim_K = hidden_dim*2
            last_dim_Q = hidden_dim
        self.last_layer = nn.Conv2d(hidden_dims[-1], 2, 1)
        self.in_branch = nn.Sequential(nn.Conv2d(in_channels * 9, hidden_dims[-2], 1),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-2],hidden_dims[-1], 1),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-1],2, 1),
                            nn.ReLU())
        
    def _make_pos_encoding(self, x, size): 
        B, C, H, W = x.shape
        H_up, W_up = size
       
        h_idx = -1 + 1/H + 2/H * torch.arange(H, device=x.device).float()
        w_idx = -1 + 1/W + 2/W * torch.arange(W, device=x.device).float()
        in_grid = torch.stack(torch.meshgrid(h_idx, w_idx), dim=0)

        h_idx_up = -1 + 1/H_up + 2/H_up * torch.arange(H_up, device=x.device).float()
        w_idx_up = -1 + 1/W_up + 2/W_up * torch.arange(W_up, device=x.device).float()
        up_grid = torch.stack(torch.meshgrid(h_idx_up, w_idx_up), dim=0)
        
        rel_grid = (up_grid - F.interpolate(in_grid.unsqueeze(0), size=(H_up, W_up), mode='nearest-exact'))
        rel_grid[:,0,:,:] *= H
        rel_grid[:,1,:,:] *= W

        return rel_grid.contiguous().detach()

    def step(self, x, syn_inp):
        q = syn_inp
        k = x
        kk = torch.cat([k],dim=1)
        for i in range(len(self.K)):
            kk = self.K[i](kk)
            dim = kk.shape[1]//2
            q = kk[:,:dim]*self.Q[i](q)
        q = self.last_layer(q)
        return q + self.in_branch(x)

    def forward(self, x, size):
        B, C, H_in, W_in = x.shape
        rel_coord = (self._make_pos_encoding(x, size).expand(B, -1, *size))
        ratio = (x.new_tensor([math.sqrt((H_in*W_in)/(size[0]*size[1]))]).view(1, -1, 1, 1).expand(B, -1, *size))
        syn_inp = torch.cat([rel_coord, ratio], dim=1)
        x = F.interpolate(F.unfold(x, 3, padding=1).view(B, C*9, H_in, W_in), size=syn_inp.shape[-2:], mode='bilinear')
        pred = self.step(x, syn_inp)
        return pred


class DMRI_SR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = make_rdn()
        self.decoder = ImplicitDecoder()
        # self.mixer = nn.Conv2d(64*2, 64, 1, padding=0, stride=1)
    
    def set_scale(self, scale, scale2):
        self.scale = scale
        self.scale2 = scale2

    def forward(self, inp):
        
        B,C,H,W = inp.shape

        H_hr = round(H*self.scale)
        W_hr = round(W*self.scale2)
        size = [H_hr, W_hr]
        
        feat = self.encoder(inp)
        # latent = self.latent_layer(feat)
        pred = self.decoder(feat,size)
        
        return pred

