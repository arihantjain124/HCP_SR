import cupy as cp
import cucim.skimage.metrics as metrics
import torch
import torch.nn as nn

def compute_ssim(hr,pred,mask):
    mask = torch.permute(mask,(1,2,3,0))
    mask = cp.array(mask)
    hr = cp.array(hr.squeeze()) * mask
    pred = cp.array(pred.squeeze()) * mask
    # print(pred.shape,hr.shape,mask.shape)
    return abs(float(metrics.structural_similarity(hr,pred,channel_axis =3,data_range=1)))
def compute_psnr(hr,pred,mask):
    mask = torch.permute(mask,(1,2,3,0))
    mask = cp.array(mask)
    
    hr = cp.array(hr.squeeze()) * mask
    pred = cp.array(pred.squeeze()) * mask
    return float(metrics.peak_signal_noise_ratio(hr,pred,data_range=1))

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, input, target, mask):
        mask = mask.bool()
        # print(input.shape,mask.shape)
        mask = torch.unsqueeze(mask,4)
        masked_input = torch.masked_select(input, mask)
        masked_target = torch.masked_select(target, mask)
        
        loss = torch.mean(torch.abs(masked_input - masked_target))
        
        return loss
    
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        mask = mask.bool()
        mask = torch.unsqueeze(mask,4)
        masked_input = torch.masked_select(input, mask)
        masked_target = torch.masked_select(target, mask)
        
        loss = torch.mean((masked_input - masked_target) ** 2)
        
        return loss