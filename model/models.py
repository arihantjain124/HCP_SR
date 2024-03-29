import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable, grad
import numpy as np
import os 
import math
from torchvision import models

class cSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        
        super(cSELayer, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        
        y = self.avg_pool(x)
        y = self.conv_du(y)
        
        return x * y

class cSELayer_3d(nn.Module):
    def __init__(self, channel, reduction=16):
        
        super(cSELayer_3d, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.conv_du = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        
        y = self.avg_pool(x)
        y = self.conv_du(y)
        
        return x * y
    
class sSELayer(nn.Module):
    def __init__(self, channel):
        super(sSELayer, self).__init__()
        
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, 1, 1, padding=0, bias=True),
                nn.Sigmoid())


    def forward(self, x):
        
        y = self.conv_du(x)
        
        return x * y
    
class sSELayer_3d(nn.Module):
    def __init__(self, channel):
        super(sSELayer, self).__init__()
        
        self.conv_du = nn.Sequential(
                nn.Conv3d(channel, 1, 1, padding=0, bias=True),
                nn.Sigmoid())


    def forward(self, x):
        
        y = self.conv_du(x)
        
        return x * y

class scSELayer(nn.Module):
    def __init__(self, channel,reduction=16):
        super(scSELayer, self).__init__()
        
        self.cSElayer = cSELayer(channel,reduction)
        self.sSElayer = sSELayer(channel)

    def forward(self, x):
        
        y1 = self.cSElayer(x)
        y2 = self.sSElayer(x)
        
        y  = torch.max(y1,y2)
        
        return y

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob,attention,attention_type,reduction): # cSE,scSE
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.attention = attention
        self.reduction = reduction

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )
        
        if self.attention:
            if attention_type == 'cSE':
                self.attention_layer = cSELayer(channel=self.out_chans,reduction=reduction)
            if attention_type == 'scSE':
                self.attention_layer = scSELayer(channel=self.out_chans,reduction=reduction)
                  
    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]

        """
        out = self.layers(input)
        
        if self.attention:
            out = self.attention_layer(out)
        
        return out

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'
            
            
            
class ConvBlock_3d(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob = 0 ,attention = True,attention_type = 'cSE',reduction = 16): # cSE,scSE
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.attention = attention
        self.reduction = reduction

        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_chans),
            nn.LeakyReLU(),
            nn.Dropout3d(drop_prob)
        )
        
        if self.attention:
            if attention_type == 'cSE':
                self.attention_layer = cSELayer_3d(channel=self.out_chans,reduction=reduction)
            if attention_type == 'scSE':
                self.attention_layer = scSELayer(channel=self.out_chans,reduction=reduction)
                  
    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]

        """
        
        out = self.layers(input)
        
        if self.attention:
            out = self.attention_layer(out)
        
        return torch.cat((input, out), 1)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'

class CSEUnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,attention_type='cSE',reduction=16):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.reduction = reduction

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob,attention=False,attention_type=attention_type,reduction=reduction)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob,attention=False,attention_type=attention_type,reduction=reduction)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob,attention=True,attention_type=attention_type,reduction=reduction)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob,attention=True,attention_type=attention_type,reduction=reduction)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob,attention=True,attention_type=attention_type,reduction=reduction)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        return self.conv2(output)

class CSEUnetModel_3d(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,attention_type='cSE',reduction=16):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.reduction = reduction

        self.down_sample_layers = nn.ModuleList([ConvBlock_3d(in_chans, chans, drop_prob,attention=False,attention_type=attention_type,reduction=reduction)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock_3d(ch, ch * 2, drop_prob,attention=False,attention_type=attention_type,reduction=reduction)]
            ch *= 2
        self.conv = ConvBlock_3d(ch, ch, drop_prob,attention=True,attention_type=attention_type,reduction=reduction)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock_3d(ch * 2, ch // 2, drop_prob,attention=True,attention_type=attention_type,reduction=reduction)]
            ch //= 2
        self.up_sample_layers += [ConvBlock_3d(ch * 2, ch, drop_prob,attention=True,attention_type=attention_type,reduction=reduction)]
        self.conv2 = nn.Sequential(
            nn.Conv3d(ch, ch // 2, kernel_size=1),
            nn.Conv3d(ch // 2, out_chans, kernel_size=1),
            nn.Conv3d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        max_pool = nn.MaxPool3d(kernel_size = 2)
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            
            output = max_pool(output)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='trilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        return self.conv2(output)

class DataConsistencyLayer(nn.Module):

    def __init__(self,us_mask):
        
        super(DataConsistencyLayer,self).__init__()

        self.us_mask = us_mask 

    def forward(self,predicted_img,us_kspace):

        # us_kspace     = us_kspace[:,0,:,:]
        predicted_img = predicted_img[:,0,:,:]
        
        kspace_predicted_img = torch.rfft(predicted_img,2,True,False).double()
        # print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,self.mask.shape)
        
        updated_kspace1  = self.us_mask * us_kspace 
        updated_kspace2  = (1 - self.us_mask) * kspace_predicted_img

        updated_kspace   = updated_kspace1[:,0,:,:,:] + updated_kspace2
        
        
        updated_img    = torch.ifft(updated_kspace,2,True) 
        
        update_img_abs = torch.sqrt(updated_img[:,:,:,0]**2 + updated_img[:,:,:,1]**2)
        
        update_img_abs = update_img_abs.unsqueeze(1)
        
        return update_img_abs.float()

class DnCn(nn.Module):

    def __init__(self,args,n_channels=2, nc=5, nd=5,**kwargs):

        super(DnCn, self).__init__()

        self.nc = nc
        self.nd = nd

        us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
        us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(args.device)

        print('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        dcs = []

        for i in range(nc):
            conv_blocks.append(CSEUnetModel(in_chans=1,out_chans=1,chans=32,num_pool_layers=3,drop_prob = 0,attention_type='cSE',reduction=16))       
            dcs.append(DataConsistencyLayer(us_mask))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self,x,k):

        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            xcrop = x

            if x.shape[2]==160 and x.shape[3]==160:
                xcrop = x[:,:,5:x.shape[2]-5,5:x.shape[3]-5]
            x = self.dcs[i](xcrop,k)
            if x.shape[2]==150 and x.shape[3]==150:
                x = F.pad(x,(5,5,5,5),"constant",0)

        return x



