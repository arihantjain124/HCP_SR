# +
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

class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

class ImplicitDecoder_3d(nn.Module):
    def __init__(self, args,hidden_dims=[64,64,64,64,64]):
        super().__init__()
        
        in_channels = args.growth
        self.tv = args.tv
        out_chans = args.out_chans
        last_dim_K = in_channels * 27
        last_dim_Q = 3
        self.pixel_shuffle = PixelShuffle3d(2)
        self.K = nn.ModuleList()
        self.Q = nn.ModuleList()
        
        for hidden_dim in hidden_dims:
            self.K.append(nn.Sequential(nn.Conv3d(last_dim_K, hidden_dim, 1),
                                        nn.LeakyReLU(),
                                        ResBlock_3d(channels = hidden_dim, nConvLayers = 5)
                                        ))    
            
            self.Q.append(nn.Sequential(nn.Conv3d(last_dim_Q, hidden_dim, 1),
                                        SineAct()))
                
            last_dim_K = hidden_dim
            last_dim_Q = hidden_dim
            
        self.last_layer = nn.Conv3d(hidden_dims[-1] , out_chans , 1)

        self.fa_adc = nn.Sequential(nn.Conv3d(2, hidden_dims[-2]//2, 1),
                                nn.LeakyReLU(),
                                nn.Conv3d(hidden_dims[-2]//2,hidden_dims[-1]//2, 1),
                                nn.LeakyReLU(),
                                nn.Conv3d(hidden_dims[-1]//2,2, 1),
                                nn.ReLU())

        self.rgb = nn.Sequential(nn.Conv3d(3, hidden_dims[-2]//2, 1),
                                nn.LeakyReLU(),
                                nn.Conv3d(hidden_dims[-2]//2,hidden_dims[-1]//2, 1),
                                nn.LeakyReLU(),
                                nn.Conv3d(hidden_dims[-1]//2,3, 1),
                                nn.ReLU())
        if self.tv:
            self.tensor_val = nn.Sequential(nn.Conv3d(hidden_dims[-1], hidden_dims[-2], 1),
                                nn.LeakyReLU(),
                                nn.Conv3d(hidden_dims[-2],hidden_dims[-1], 1),
                                nn.LeakyReLU(),
                                nn.Conv3d(hidden_dims[-1],6, 1),
                                nn.LeakyReLU())
            
    def step(self,  x, rel_coor):
        
        k = x
        q = rel_coor
        # print(q.shape)
        if(q == None):
            for i in range(len(self.K)):
                k = self.K[i](k)
            
            q = k
            out = self.last_layer(k)

        else:
            for i in range(len(self.K)):
                
                k = self.K[i](k)
                q = k*self.Q[i](q)
                
        
            out = self.last_layer(q)
        
        out = out + torch.cat([self.fa_adc(out[:,:2,:,:,:]), self.rgb(out[:,2:,:,:,:])], dim = 1)


        if self.tv:
            tv = self.tensor_val(q)
            return out ,tv
        else:
            return out     
        

    
    
    def forward(self, x, size,rel_coord):

        B, C, H_in, W_in,D_in = x.shape

        # x = self.pixel_shuffle(x)

        x = F.interpolate(unfoldNd.unfoldNd(x, 3, padding=1).view(B, C*27, H_in, W_in,D_in), size=size[-3:],mode = "trilinear")
        
        return self.step(x, rel_coord)  
    


class ImplicitDecoder_2d(nn.Module):
    def __init__(self,args,hidden_dims=[64, 64, 64, 64, 64]):
        super().__init__()

        in_channels = args.growth
        self.tv = args.tv
        out_chans = args.out_chans
        last_dim_K = in_channels * 9
        last_dim_Q = 2

        self.K = nn.ModuleList()
        self.Q = nn.ModuleList()
        
        for hidden_dim in hidden_dims:
            self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                        nn.LeakyReLU(),
                                        ResBlock(channels = hidden_dim, nConvLayers = 3)
                                        ))    


            self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                        SineAct()))
            last_dim_K = hidden_dim
            last_dim_Q = hidden_dim
            
        self.last_layer = nn.Conv2d(hidden_dims[-1], out_chans, 1)
        
    def step(self, x, rel_coor):
        
        
        k = x
        q = rel_coor
        # print(q.shape)

        if(q==None):
            for i in range(len(self.K)):
                k = self.K[i](k)
            
            
            out = self.last_layer(k)

        else:
            for i in range(len(self.K)):
                
                k = self.K[i](k)
                q = k*self.Q[i](q)
                
            out = self.last_layer(q)
        
        return out


    def forward(self, x, size,rel_coord):
        B, C, H_in, W_in = x.shape
        
        x = F.interpolate(F.unfold(x, 3, padding=1).view(B, C*9, H_in, W_in), size=size[-2:], mode='bilinear')
        # print(x.shape,size)
        # print(syn_inp.shape,x.shape)
        return self.step(x, rel_coord)

    

