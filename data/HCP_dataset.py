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


class hcp_data(torch.utils.data.Dataset):
    def __init__(self, opt,ids):
        super(hcp_data).__init__()

        self.blk_size = opt.block_size
        self.crop_depth = opt.crop_depth
        self.base_dir = opt.dir if opt.dir != None else "/storage/users/arihant"
        self.typ = opt.typ
        self.path,self.tot = self.load_path(self.base_dir,ids)
        self.ids = ids
        self.debug = opt.debug
        if(opt.sort == True):
            self.ids.sort()
        self.preload = opt.preload
        self.blk_per_vol = 0
        self.blk_indx = []
        if(opt.preload == True):
            self.loaded = {}
            self.loaded_gt = {}
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
            if(rebuild == True):
                self.loaded_blk[i],self.loaded_adc[i],self.loaded_fa[i],self.loaded_rgb[i] = self.pre_proc(i)
            if(self.typ == 'upsampled'):
                name = self.path['3T'][i]['upsampled']
                res_vol = h5py.File(name, 'r')
                self.loaded[i] = res_vol
                if(shp_loaded == False):
                    shp = np.array(res_vol.get('volumes0')).shape
                    self.num_blks = (np.floor(shp[0]/self.blk_size[0]),np.floor(shp[1]/self.blk_size[1]),np.floor(shp[2]/self.blk_size[2]))
                    shp_loaded = True
            else:
                name = self.path['3T'][i]['downsampled']
                res_vol = h5py.File(name, 'r')
                self.loaded[i] = res_vol
                if(shp_loaded == False):
                    shp = np.array(res_vol.get('volumes0')).shape
                    self.num_blks = (np.floor(shp[0]/self.blk_size[0]),np.floor(shp[1]/self.blk_size[1]),np.floor(shp[2]/self.blk_size[2]))
                    shp_loaded = True

            name = self.path['7T'][i]['GT']
            res = h5py.File(name, 'r')
            self.loaded_gt[i] = res
            self.loaded_blk[i],self.loaded_adc[i],self.loaded_fa[i],self.loaded_rgb[i] = self.pre_proc(i)
            
            res_vol.close()
            res.close()
            if(self.debug == True):
                print(i,"loaded")
        

    def load_path(self,base_dir,ids):
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
    
    def blocks(self,base_mask):
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

        ind_block = np.zeros([xind_block.shape[0]*yind_block.shape[0]*zind_block.shape[0], 6])
        count = 0
        for ii in np.arange(0, xind_block.shape[0]):
            for jj in np.arange(0, yind_block.shape[0]):
                for kk in np.arange(0, zind_block.shape[0]):
                    ind_block[count, :] = np.array([xind_block[ii], xind_block[ii]+self.blk_size[0]-1, yind_block[jj], yind_block[jj]+self.blk_size[1]-1, zind_block[kk], zind_block[kk]+self.blk_size[2]-1])
                    count = count + 1

        ind_block = ind_block.astype(int)
        # print(ind_block)
        return ind_block,len(ind_block)

    def extract_block(self,data, inds):
        blocks = []
        for ii in np.arange(inds.shape[0]):
            inds_this = inds[ii, :]
            curr_blk = data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, ...]
            # if(np.count_nonzero(curr_blk)/curr_blk.size > 0.6):
            blocks.append(curr_blk)
        return np.stack(blocks, axis=0)
    
    # def verify_blk(self,vol_norm,adc,fa,rgb,inds):
    #     for ii in np.arange(inds.shape[0]):
    #         inds_this = inds[ii, :]
    #         curr_blk = data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, ...]
    #         # if(np.count_nonzero(curr_blk)/curr_blk.size > 0.6):
    #         blocks.append(curr_blk)

    def rebuild(self,args):
        self.blk_size = args.block_size
        self.loaded_blk = {}
        self.loaded_adc = {}
        self.loaded_fa = {}
        self.loaded_rgb = {}
        self.preload_data(rebuild=True)
        self.blk_indx = np.cumsum(self.blk_indx)

    def pre_proc(self,idx):

        vol = np.array(self.loaded[idx].get('volumes0'))
        mask = np.array(self.loaded[idx].get('mask'))
        adc = np.array(self.loaded_gt[idx].get('ADC'))
        fa = np.array(self.loaded_gt[idx].get('FA'))
        rgb = np.array(self.loaded_gt[idx].get('color_FA'))
        # print("raw",vol.shape)
        # print("ADC",adc.shape)
        # print("FA",fa.shape)
        # print("RGB",rgb.shape)
        curr_blk = self.blocks(mask)
        # self.blk_per_vol = curr_blk[1]
        self.blk_indx.append(curr_blk[1])
        vol_norm = (vol-np.min(vol))/(np.max(vol)-np.min(vol))
        mask  = mask[...,np.newaxis]
        # curr_blk = self.verify_blk(vol_norm,adc,fa,rgb,curr_blk)
        vol_norm = np.concatenate((vol_norm,mask),axis =3)

        blks_img = self.extract_block(vol_norm,curr_blk[0])
        blks_adc = self.extract_block(adc,curr_blk[0])
        blks_fa = self.extract_block(fa,curr_blk[0])
        blks_rgb = self.extract_block(rgb,curr_blk[0])

        return blks_img,blks_adc,blks_fa,blks_rgb