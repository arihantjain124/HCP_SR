import torch
import torch.nn as nn
import numpy as np
import math
from torch import linalg as LA
import torch.nn.functional as F
from .sobel import sobel_3d


class SOG(nn.Module):
    def __init__(self,chans,type):
        super(SOG,self).__init__()

        self.kernel = {}
        for i in sobel_3d.keys():
            self.kernel[i] = torch.FloatTensor(sobel_3d[i]).unsqueeze(0).unsqueeze(0).to("cuda")

        self.loss_fn = nn.L1Loss()



    def g2(self,inp):
    
        inp = torch.permute(inp, (0,4,1,2,3))
        res = []
        for i in range(inp.shape[1]):
            
            temp = inp[:,i,:,:,:].unsqueeze(1)
        
            x = F.conv3d(temp, self.kernel['x'])

            y = F.conv3d(temp, self.kernel['y'])

            z = F.conv3d(temp, self.kernel['z'])

            temp = torch.sqrt((x**2) + (y**2)+ (z**2))
            
            x = F.conv3d(temp, self.kernel['x'])

            y = F.conv3d(temp, self.kernel['y'])

            z = F.conv3d(temp, self.kernel['z'])

            res.append(torch.sqrt((x**2) + (y**2))+ (z**2))
            
        res = torch.cat(res,dim=1)
    
        return res

    def forward(self,pred,hr):

        return self.loss_fn(self.g2(pred),self.g2(hr))
