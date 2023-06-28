from torch.nn import Conv3d
import torch.nn as nn
import torch
from torchmetrics import PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure

class DeepDTI_torch(nn.Module):
    def __init__(self):
        super(DeepDTI_torch, self).__init__()

        self.model= nn.Sequential(
            nn.Conv3d(7, 128, 3,padding="same"),      
            nn.ReLU(),
            nn.Conv3d(128, 128, 3,padding="same"),     
            nn.BatchNorm3d(128),    
            nn.ReLU(),
            nn.Conv3d(128, 128, 3,padding="same"),     
            nn.BatchNorm3d(128),    
            nn.ReLU(),
            nn.Conv3d(128, 128, 3,padding="same"),     
            nn.BatchNorm3d(128),    
            nn.ReLU(),
            nn.Conv3d(128, 128, 3,padding="same"),     
            nn.BatchNorm3d(128),    
            nn.ReLU(),
            nn.Conv3d(128, 128, 3,padding="same"),     
            nn.BatchNorm3d(128),    
            nn.ReLU(),
            nn.Conv3d(128, 128, 3,padding="same"),     
            nn.BatchNorm3d(128),    
            nn.ReLU(),
            nn.Conv3d(128, 128, 3,padding="same"),     
            nn.BatchNorm3d(128),    
            nn.ReLU(),
            nn.Conv3d(128, 128, 3,padding="same"),     
            nn.BatchNorm3d(128),    
            nn.ReLU(),
            nn.Conv3d(128, 7, 3,padding="same"),  
            nn.ReLU()
        )
        

    def forward(self,x):
        # Set 1
        output = self.model(x)
        return output


        
    
class Loss_MSE(nn.Module):
    def __init__(self):
        super(Loss_MSE, self).__init__()

    def forward(self, pred, gt):
        mask = gt[:,:,:,:,-1]
        loss = (gt[:,:,:,:,:7] - pred) * mask.unsqueeze(dim = -1)
        loss = torch.mean(torch.square(loss))
        return loss


class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
        
        self.psnr = PeakSignalNoiseRatio().cuda()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

    def forward(self, pred, gt):
        mask = gt[:,:,:,:,-1].unsqueeze(dim = -1)
        gt = gt[...,0:7] * mask
        pred = pred * mask
        return self.psnr(preds = pred,target = gt),self.ssim(preds = pred,target = gt)