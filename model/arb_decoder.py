import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
import random 
import math
from model.rdn import make_rdn
from model.resblock import ResBlock_3d
from model.resblock import ResBlock
import unfoldNd

class SineAct(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)

class ImplicitDecoder_3d(nn.Module):
    def __init__(self, in_channels=64, hidden_dims=[64, 64, 64, 64, 64, 64],out_chans= 5):
        super().__init__()

        last_dim_K = in_channels * 27
        
        last_dim_Q = 4
        
        self.K = nn.ModuleList()
        self.Q = nn.ModuleList()
        
        for hidden_dim in hidden_dims:
            self.K.append(nn.Sequential(nn.Conv3d(last_dim_K, hidden_dim, 1),
                                        nn.ReLU(),
                                        ResBlock_3d(channels = hidden_dim, nConvLayers = 3)
                                        ))    
            
            self.Q.append(nn.Sequential(nn.Conv3d(last_dim_Q, hidden_dim, 1),
                                        SineAct()))
            
            last_dim_K = hidden_dim
            last_dim_Q = hidden_dim
            
        self.last_layer = nn.Conv3d(hidden_dims[-1] , out_chans , 1)
        
        self.in_branch = nn.Sequential(nn.Conv3d(in_channels * 27, hidden_dims[-2], 1),
                            nn.LeakyReLU(),
                            nn.Conv3d(hidden_dims[-2],hidden_dims[-1], 1),
                            nn.LeakyReLU(),
                            nn.Conv3d(hidden_dims[-1],out_chans, 1),
                            nn.LeakyReLU())
        
        self.tensor_val = nn.Sequential(nn.Conv3d(hidden_dims[-1], hidden_dims[-2], 1),
                            nn.ReLU(),
                            nn.Conv3d(hidden_dims[-2],hidden_dims[-1], 1),
                            nn.ReLU(),
                            nn.Conv3d(hidden_dims[-1],6, 1),
                            nn.ReLU())
        
    def step(self,  x, syn_inp):
        
        q = syn_inp
        k = x
        
        for i in range(len(self.K)):
            
            k = self.K[i](k)
            q = k*self.Q[i](q)
            
        tv = self.tensor_val(q)
        
        out = self.last_layer(q)
        
        return out + self.in_branch(x) ,tv
    
    def _make_pos_encoding(self, x, size): 
        B, C, H, W, D = x.shape
        H_up, W_up, D_up = size 
        h_idx = -1 + 1/H + 2/H * torch.arange(H, device=x.device).float()
        w_idx = -1 + 1/W + 2/W * torch.arange(W, device=x.device).float()
        d_idx = -1 + 1/D + 2/D * torch.arange(D, device=x.device).float()
        in_grid = torch.stack(torch.meshgrid(h_idx, w_idx,d_idx), dim=0)

        h_idx_up = -1 + 1/H_up + 2/H_up * torch.arange(H_up, device=x.device).float()
        w_idx_up = -1 + 1/W_up + 2/W_up * torch.arange(W_up, device=x.device).float()
        d_idx_up = -1 + 1/D_up + 2/D_up * torch.arange(D_up, device=x.device).float()
        up_grid = torch.stack(torch.meshgrid(h_idx_up, w_idx_up,d_idx_up), dim=0)
        
        rel_grid = (up_grid - F.interpolate(in_grid.unsqueeze(0), size=(H_up, W_up,D_up), mode='nearest-exact'))
        rel_grid[:,0,:,:] *= H
        rel_grid[:,1,:,:] *= W
        rel_grid[:,2,:,:] *= D

        return rel_grid.contiguous().detach()
    
    def forward(self, x, size):
        B, C, H_in, W_in,D_in = x.shape
        
        
        rel_coord = (self._make_pos_encoding(x, size).expand(B, -1, *size))
        
        ratio = (x.new_tensor([math.sqrt((H_in*W_in*D_in)/(size[0]*size[1]*size[2]))]).view(1, -1, 1, 1).expand(B, -1, *size))
        
        syn_inp = torch.cat([rel_coord, ratio], dim=1)
        
        x = F.interpolate(unfoldNd.unfoldNd(x, 3, padding=1).view(B, C*27, H_in, W_in,D_in), size=ratio.shape[-3:],mode = "trilinear")
        
        # print(syn_inp.shape,x.shape)
        pred,tv = self.step(x, syn_inp)
        return pred,tv
    
class ImplicitDecoder_2d(nn.Module):
    def __init__(self, in_channels=64, hidden_dims=[64, 64, 64, 64, 64],out_chans= 5):
        super().__init__()

        last_dim_K = in_channels * 9
        
        last_dim_Q = 3

        self.K = nn.ModuleList()
        self.Q = nn.ModuleList()
        
        for hidden_dim in hidden_dims:
            self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                        nn.ReLU(),
                                        ResBlock(channels = hidden_dim, nConvLayers = 4)
                                        ))    
            self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                        SineAct()))
            last_dim_K = hidden_dim
            last_dim_Q = hidden_dim
            
        self.last_layer = nn.Conv2d(hidden_dims[-1], out_chans, 1)
        
        self.in_branch = nn.Sequential(nn.Conv2d(in_channels * 9, hidden_dims[-2], 1),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-2],hidden_dims[-1], 1),
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-1],out_chans, 1),
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
        
        for i in range(len(self.K)):
            
            k = self.K[i](k)
            q = k*self.Q[i](q)
            
        q = self.last_layer(q)
        
        return q + self.in_branch(x)


    def forward(self, x, size):
        B, C, H_in, W_in = x.shape
        
        rel_coord = (self._make_pos_encoding(x, size).expand(B, -1, *size))
        
        ratio = (x.new_tensor([math.sqrt((H_in*W_in)/(size[0]*size[1]))]).view(1, -1, 1, 1).expand(B, -1, *size))
        
        syn_inp = torch.cat([rel_coord, ratio], dim=1)
        
        x = F.interpolate(F.unfold(x, 3, padding=1).view(B, C*9, H_in, W_in), size=syn_inp.shape[-2:], mode='bilinear')
        
        # print(syn_inp.shape,x.shape)
        pred = self.step(x, syn_inp)
        return pred
    
    