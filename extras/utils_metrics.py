import cupy as cp
import cucim.skimage.metrics as metrics
import torch
import torch.nn as nn
import numpy as np

def recon(x,lr,vol_size):
    num_blk = x.shape[0]
    vol = np.empty(shape=vol_size)
    print(num_blk)
    for i in range(num_blk):
        vol[lr[i][0]:lr[i][1]+1,lr[i][2]:lr[i][3]+1,lr[i][4]:lr[i][5]+1,...] = x[i,...]
    return vol

def compute_ssim(hr,pred):
    hr = cp.array(hr.squeeze())
    pred = cp.array(pred.squeeze())
    return abs(float(metrics.structural_similarity(hr,pred,channel_axis =3,data_range=1,win_size=3)))

def compute_psnr(hr,pred):
    hr = cp.array(hr.squeeze())
    pred = cp.array(pred.squeeze())
    return float(metrics.peak_signal_noise_ratio(hr,pred,data_range=1))


def compute_psnr_ssim(hr,pred,pnts,mask):
    hr = cp.array(hr.squeeze())
    pred = cp.array(pred.squeeze())
    print("here")
    return float(metrics.peak_signal_noise_ratio(hr,pred,data_range=1))
