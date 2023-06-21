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
        self.crop_depth = opt.crop_depth
        self.base_dir = opt.dir if opt.dir != None else "/storage/users/arihant"
        self.typ = opt.typ
        self.path,self.tot = self.load_path(self.base_dir,ids)
        self.ids = ids
        self.debug = opt.debug
        if(opt.sort == True):
            self.ids.sort()
        self.preload = opt.preload

        if(opt.preload == True):
            self.loaded = {}
            self.loaded_gt = {}
            self.loaded_blk = {}
            self.loaded_adc = {}
            self.loaded_fa = {}
            self.loaded_rgb = {}
            self.preload_data()

    def __len__(self):
        return self.tot
    
    def __getitem__(self,indx):
        vol_idx = self.ids[indx]
        return self.loaded_blk[vol_idx],self.loaded_fa[vol_idx],self.loaded_adc[vol_idx],self.loaded_rgb[vol_idx]
    
    def preload_data(self,rebuild = False):

        shp_loaded = False
        
        for i in self.ids:
            
            if(rebuild == True):
                self.loaded_blk[i],self.loaded_adc[i],self.loaded_fa[i],self.loaded_rgb[i] = self.pre_proc(i)
            
            if(self.typ == 'upsampled'):
                name = self.path['3T'][i]['upsampled']
                res_vol = h5py.File(name, 'r')

                
                if(shp_loaded == False):
                    shp = np.array(res_vol.get('volumes0')).shape
                    shp_loaded = True

                self.loaded[i] = {'vol0':res_vol.get('volumes0')[:]
                                  ,'mask':res_vol.get('mask')[:] }
                
            else:
                name = self.path['3T'][i]['downsampled']
                res_vol = h5py.File(name, 'r')

                self.loaded[i] = {'vol0':res_vol.get('volumes0')[:]
                                  ,'mask':res_vol.get('mask')[:] }
                
                if(shp_loaded == False):
                    shp = np.array(res_vol.get('volumes0')).shape
                    shp_loaded = True

            name = self.path['7T'][i]['GT']
            res = h5py.File(name, 'r')

            self.loaded_gt[i] = {'ADC':res.get('ADC')[:]
                                ,'FA':res.get('FA')[:] 
                                ,'color_FA':res.get('color_FA')[:] }
            
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
    

    def pre_proc(self,idx):

        vol = self.loaded[idx]['vol0']
        mask = self.loaded[idx]['mask']
        adc = self.loaded_gt[idx]['ADC']
        fa = self.loaded_gt[idx]['FA']
        rgb = self.loaded_gt[idx]['color_FA']

        print("raw",vol.shape)
        print("ADC",adc.shape)
        print("FA",fa.shape)
        print("RGB",rgb.shape)
        
        vol_norm = (vol-np.min(vol))/(np.max(vol)-np.min(vol))
        mask  = mask[...,np.newaxis]
        # curr_blk = self.verify_blk(vol_norm,adc,fa,rgb,curr_blk)
        vol_norm = np.concatenate((vol_norm,mask),axis =3)

        return vol_norm,adc,fa,rgb