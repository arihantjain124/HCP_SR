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
        name = path['3T'][i]['upsampled']
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


class hcp_data(torch.utils.data.Dataset):
    def __init__(self, opt,ids):
        super(hcp_data).__init__()

        self.blk_size = opt.block_size
        self.base_dir = opt.dir if opt.dir != None else "/storage/users/arihant"
        self.ids = ids
        self.debug = opt.debug
        if(opt.sort == True):
            self.ids.sort()
        self.blk_per_vol = 0
        self.blk_indx = []
        self.loaded_blk = {}
        self.loaded_adc = {}
        self.loaded_fa = {}
        self.loaded_rgb = {}
        self.preload_data()
        self.blk_indx = np.cumsum(self.blk_indx)


    def __len__(self):
        return self.blk_indx[-1]
    
    def __getitem__(self,indx):
        # print(indx)
        blk_idx = np.searchsorted(self.blk_indx, indx)
        vol_idx = self.ids[blk_idx]
        blk_idx = indx - self.blk_indx[blk_idx]
        return self.loaded_blk[vol_idx][blk_idx,...],self.loaded_fa[vol_idx][blk_idx,...],self.loaded_adc[vol_idx][blk_idx,...],self.loaded_rgb[vol_idx][blk_idx,...]
        
    def preload_data(self,rebuild = False):

        shp_loaded = False
        for i in self.ids:
            if(shp_loaded == False):
                shp = np.array(loaded[i]['vol0']).shape
                self.num_blks = (np.floor(shp[0]/self.blk_size[0]),np.floor(shp[1]/self.blk_size[1]),np.floor(shp[2]/self.blk_size[2]))
                shp_loaded = True

            self.loaded_blk[i],self.loaded_adc[i],self.loaded_fa[i],self.loaded_rgb[i] = self.pre_proc(i)
            if(self.debug == True):
                print(i,"loaded")

    
    def blocks(self,base_mask,base_vol):
        # %% divide brain volume to blocks
        xind,yind,zind = np.nonzero(base_mask)
        xmin,xmax = np.min(xind),np.max(xind)
        ymin,ymax = np.min(yind),np.max(yind)
        zmin,zmax = np.min(zind),np.max(zind)

        ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax] 
        # print(ind_brain)
        # calculate number of blocks along each dimension
        xlen = xmax - xmin + 1
        ylen = ymax - ymin + 1
        zlen = zmax - zmin + 1

        nx = int(np.ceil(xlen / self.blk_size[0]))
        ny = int(np.ceil(ylen / self.blk_size[1]))
        nz = int(np.ceil(zlen / self.blk_size[2]))

        # determine starting and ending indices of each block
        xstart = xmin
        ystart = ymin
        zstart = zmin

        xend = xmax - self.blk_size[0] + 1
        yend = ymax - self.blk_size[1] + 1
        zend = zmax - self.blk_size[2] + 1

        xind_block = np.round(np.linspace(xstart, xend, nx))
        yind_block = np.round(np.linspace(ystart, yend, ny))
        zind_block = np.round(np.linspace(zstart, zend, nz))

        ind_block = []
        count = 0
        for ii in np.arange(0, xind_block.shape[0]):
            for jj in np.arange(0, yind_block.shape[0]):
                for kk in np.arange(0, zind_block.shape[0]):
                    temp = np.array([xind_block[ii], xind_block[ii]+self.blk_size[0]-1, yind_block[jj], yind_block[jj]+self.blk_size[1]-1, zind_block[kk], zind_block[kk]+self.blk_size[2]-1]).astype(int)
                    curr_blk = base_vol[temp[0]:temp[1]+1, temp[2]:temp[3]+1, temp[4]:temp[5]+1, ...]
                    if(np.count_nonzero(curr_blk)/curr_blk.size > 0.65):
                        ind_block.append(temp)
                        count = count + 1

        ind_block = np.stack(ind_block)
        ind_block = ind_block.astype(int)
        # print(ind_block)
        return ind_block,len(ind_block)

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
        mask = loaded[idx]['mask']
        adc = loaded_gt[idx]['ADC']
        fa = loaded_gt[idx]['FA']
        rgb = loaded_gt[idx]['color_FA']

        # print("raw",vol.shape)
        # print("ADC",adc.shape)
        # print("FA",fa.shape)
        # print("RGB",rgb.shape)

        vol_norm = (vol-np.min(vol))/(np.max(vol)-np.min(vol))
        curr_blk = self.blocks(mask,vol_norm)
        # self.blk_per_vol = curr_blk[1]
        self.blk_indx.append(curr_blk[1])
        mask  = mask[...,np.newaxis]
        vol_norm = np.concatenate((vol_norm,mask),axis =3)
        # curr_blk = self.verify_blk(vol_norm,adc,fa,rgb,curr_blk)

        blks_img = self.extract_block(vol_norm,curr_blk[0])
        blks_adc = self.extract_block(adc,curr_blk[0])
        blks_fa = self.extract_block(fa,curr_blk[0])
        blks_rgb = self.extract_block(rgb,curr_blk[0])

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













