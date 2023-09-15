import torch
import os
import numpy as np
import torch
import random
import math
from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import data.utils_dataloader as utils
from itertools import islice
import h5py

np.seterr(all="ignore")

loaded = {}
loaded_gt ={}

path,tot = "",""

def load_path(base_dir,ids):
    base_dir_7t = [base_dir + "/HCP_7T/" + i   for i in ids]
    base_dir_3t = [base_dir + "/HCP_3T/" + i   for i in ids]
    path_7t = {}
    path_3t = {}
    for i in base_dir_7t:
        path_7t[i[-6:]] = {"h5" : i + "/" + i[-6:] + ".h5"
                        , "GT" : i + "/" + i[-6:] + "_GT.h5"}
    for i in base_dir_3t:
        path_3t[i[-6:]] = {"h5" : i + "/" + i[-6:] + ".h5"
                        , "upsampled" : i + "/" + i[-6:] + "_upsampled.h5"
                        , "GT" : i + "/" + i[-6:] + "_GT.h5"}
    path = {'3T': path_3t, "7T":  path_7t}
    p = list(path_7t.keys())
    q = list(path_3t.keys())
    common = list(set(p) & set(q))

    return path,len(common)

def load_data(base_dir,ids):
    ids.sort()
    path,tot = load_path(base_dir,ids)
    for i in ids:
        name = path['3T'][i]['h5']
        res_vol = h5py.File(name, 'r')
        # print(res_vol.keys())
        loaded[i] = {'vol0':res_vol.get('volumes0')[:]
                            ,'mask':res_vol.get('mask')[:] }
        
        name = path['7T'][i]['GT']
        res = h5py.File(name, 'r')
        # print(res.keys())
        loaded_gt[i] = {'ADC':res.get('ADC')[:]
                            ,'FA':res.get('FA')[:] 
                            ,'color_FA':res.get('color_FA')[:] }
        
        
        res_vol.close()
        res.close()

        name = path['7T'][i]['h5']
        res = h5py.File(name, 'r')
        loaded_gt[i]['vol0'] = res.get('volumes0')[:]
        loaded_gt[i]['mask'] = res.get('mask')[:]
        res_vol.close()
        res.close()



class hcp_data(torch.utils.data.Dataset):
    def __init__(self, opt,ids):
        super(hcp_data).__init__()

        self.blk_size = opt.block_size
        self.base_dir = opt.dir if opt.dir != None else "/storage/users/arihant"
        self.ids = ids
        self.debug = opt.debug
        self.scale = opt.scale
        if(opt.sort == True):
            self.ids.sort()
        self.preload_data()
        self.blk_indx = np.cumsum(self.blk_indx)


    def __len__(self):
        return self.blk_indx[-1]
    
    def set_scale(self, scale):
        self.scale = scale
        
    def __getitem__(self,indx):

        blk_idx = np.searchsorted(self.blk_indx, indx)
        vol_idx = self.ids[blk_idx]
        blk_idx = indx - self.blk_indx[blk_idx]
        return self.loaded_blk[vol_idx][blk_idx,...],self.loaded_fa[vol_idx][blk_idx,...],self.loaded_adc[vol_idx][blk_idx,...],self.loaded_rgb[vol_idx][blk_idx,...]
        
    def preload_data(self):
        
        self.blk_indx = []
        self.loaded_blk = {}
        self.loaded_adc = {}
        self.loaded_fa = {}
        self.loaded_rgb = {}
        for i in self.ids:
            self.loaded_blk[i],self.loaded_adc[i],self.loaded_fa[i],self.loaded_rgb[i] = self.pre_proc(i)
            if(self.debug == True):
                print(i,"loaded")

    
    def blk_points_pair(self,datalr,datahr,blk_size = [16,16,4],sca = (1,1,1)):
    
        shpind = np.nonzero(datalr)
        xmin,xmax = np.min(shpind[0]),np.max(shpind[0])
        ymin,ymax = np.min(shpind[1]),np.max(shpind[1])
        zmin,zmax = np.min(shpind[2]),np.max(shpind[2])
        
        len_lr = [xmax - xmin + 1,ymax - ymin + 1,zmax - zmin + 1]
        n = [int(round(len_lr[i] / blk_size[i])) for i in range(3)]
        
        # determine starting and ending indices of each block
        
        lr_start = [xmin,ymin,zmin]
        lr_end = [xmax - blk_size[0] + 1,ymax - blk_size[1] + 1,zmax - blk_size[2] + 1]
        
        shpind = np.nonzero(datahr)
        xmin,xmax = np.min(shpind[0]),np.max(shpind[0])
        ymin,ymax = np.min(shpind[1]),np.max(shpind[1])
        zmin,zmax = np.min(shpind[2]),np.max(shpind[2])
        
        blk_size_hr = [round(blk_size[i]*sca[i]) for i in range(3)]
        hr_start = [xmin,ymin,zmin]
        hr_end = [xmax - blk_size_hr[0] + 1,ymax - blk_size_hr[1] + 1,zmax - blk_size_hr[2] + 1]
        
        ranges_lr = [np.round(np.linspace(lr_start[i], lr_end[i], n[i])) for i in range(3)]
        ranges_hr = [np.round(np.linspace(hr_start[i], hr_end[i], n[i])) for i in range(3)]
        
        ind_block_lr = []
        ind_block_hr = []
        count = 0
        for ii in np.arange(0, ranges_lr[0].shape[0]):
            for jj in np.arange(0, ranges_lr[1].shape[0]):
                for kk in np.arange(0, ranges_lr[2].shape[0]):
                    temp_lr = np.array([ranges_lr[0][ii], ranges_lr[0][ii]+blk_size[0]-1, ranges_lr[1][jj], ranges_lr[1][jj]+blk_size[1]-1, ranges_lr[2][kk], ranges_lr[2][kk]+blk_size[2]-1]).astype(int)
                    temp_hr = np.array([ranges_hr[0][ii], ranges_hr[0][ii]+blk_size_hr[0]-1, ranges_hr[1][jj], ranges_hr[1][jj]+blk_size_hr[1]-1, ranges_hr[2][kk], ranges_hr[2][kk]+blk_size_hr[2]-1]).astype(int)
                    curr_blk = datalr[temp_lr[0]:temp_lr[1]+1, temp_lr[2]:temp_lr[3]+1, temp_lr[4]:temp_lr[5]+1, ...]
                    curr_blk_hr = datahr[temp_hr[0]:temp_hr[1]+1, temp_hr[2]:temp_hr[3]+1, temp_hr[4]:temp_hr[5]+1, ...]
                    if((np.count_nonzero(curr_blk)/curr_blk.size > 0.65) and (np.count_nonzero(curr_blk_hr)/curr_blk_hr.size > 0.65)):
                        ind_block_lr.append(temp_lr)
                        ind_block_hr.append(temp_hr)
                        count = count + 1

                        
        # print(blk_size_hr)
        ind_block_lr = np.stack(ind_block_lr)
        ind_block_lr = ind_block_lr.astype(int)
        ind_block_hr = np.stack(ind_block_hr)
        ind_block_hr = ind_block_hr.astype(int)
        # print(ind_block)
        return ind_block_lr,ind_block_hr,len(ind_block_lr)


    def extract_block(self,data, inds):
            blocks = []
            for ii in np.arange(inds.shape[0]):
                inds_this = inds[ii, :]
                curr_blk = data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, ...]
                # if(np.count_nonzero(curr_blk)/curr_blk.size > 0.6):
                # print(curr_blk.shape)
                blocks.append(curr_blk)
            return np.stack(blocks, axis=0)

    def pre_proc(self,idx):

        vol = loaded[idx]['vol0']
        mask_lr = loaded[idx]['mask']
        mask_hr = loaded_gt[idx]['mask']
        adc = loaded_gt[idx]['ADC']
        fa = loaded_gt[idx]['FA']
        rgb = loaded_gt[idx]['color_FA']


        vol_norm = (vol-np.min(vol))/(np.max(vol)-np.min(vol))
        curr_blk = self.blk_points_pair(mask_lr,mask_hr,self.blk_size,self.scale)
        
        self.blk_indx.append(curr_blk[2])
        
        mask_lr  = mask_lr[...,np.newaxis]

        vol_norm = np.concatenate((vol_norm,mask_lr),axis =3)
        
        blks_img = self.extract_block(vol_norm,curr_blk[0])

        blks_adc = self.extract_block(adc,curr_blk[1])
        blks_fa = self.extract_block(fa,curr_blk[1])
        blks_rgb = self.extract_block(rgb,curr_blk[1])
        
        return blks_img,blks_adc,blks_fa,blks_rgb
    



class hcp_data_test(torch.utils.data.Dataset):
    def __init__(self, opt,ids):
        super(hcp_data_test).__init__()
        self.crop_depth = opt.crop_depth
        self.base_dir = opt.dir if opt.dir != None else "/storage/users/arihant"
        self.ids = ids
        self.debug = opt.debug
        if(opt.sort == True):
            self.ids.sort()
        self.loaded_blk = {}
        self.loaded_adc = {}
        self.loaded_fa = {}
        self.loaded_rgb = {}
        self.preload_data()

    def __len__(self):
        return len(self.ids)
    
    def set_scale(self, scale):
        self.scale = scale
    
    def __getitem__(self,indx):
        vol_idx = self.ids[indx]
        return self.loaded_blk[vol_idx],self.loaded_fa[vol_idx],self.loaded_adc[vol_idx],self.loaded_rgb[vol_idx]
    
    def preload_data(self,rebuild = False):

        shp_loaded = False
        
        for i in self.ids:
                
            if(shp_loaded == False):
                shp = np.array(loaded[i]['vol0']).shape
                shp_loaded = True

            self.loaded_blk[i],self.loaded_adc[i],self.loaded_fa[i],self.loaded_rgb[i] = self.pre_proc(i)

            if(self.debug == True):
                print(i,"loaded")
    

    def pre_proc(self,idx):

        vol = loaded[idx]['vol0']
        mask = loaded[idx]['mask']
        adc = loaded_gt[idx]['ADC']
        fa = loaded_gt[idx]['FA']
        rgb = loaded_gt[idx]['color_FA']

        # print("raw",vol.shape)
        # print("ADC",adc.shape)
        # print("FA",fa.shape)
        # print("RGB",rgb.shape)
        
        vol_norm = (vol-np.min(vol))/(np.max(vol)-np.min(vol))
        mask  = mask[...,np.newaxis]
        # curr_blk = self.verify_blk(vol_norm,adc,fa,rgb,curr_blk)
        vol_norm = np.concatenate((vol_norm,mask),axis =3)

        return vol_norm,adc,fa,rgb













