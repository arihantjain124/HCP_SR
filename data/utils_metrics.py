import cupy as cp
import cucim.skimage.metrics as metrics
import torch
import torch.nn as nn

def compute_ssim(hr,pred):
    hr = cp.array(hr.squeeze())
    pred = cp.array(pred.squeeze())
    print(pred.shape,hr.shape)
    return abs(float(metrics.structural_similarity(hr,pred,channel_axis =3,data_range=1)))

def compute_psnr(hr,pred):
    hr = cp.array(hr.squeeze())
    pred = cp.array(pred.squeeze())
    return float(metrics.peak_signal_noise_ratio(hr,pred,data_range=1))

