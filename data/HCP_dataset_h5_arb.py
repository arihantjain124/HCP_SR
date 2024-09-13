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
import os
import torchio as tio
from itertools import permutations 
import skimage.metrics as metrics

from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
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
                        , "GT" : i + "/" + i[-6:] + "_GT.h5",
                        "bvals" : i + "/T1w/Diffusion_7T/bvals" , "bvecs" : i + "/T1w/Diffusion_7T/bvecs"}
    for i in base_dir_3t:
        path_3t[i[-6:]] = {"h5" : i + "/" + i[-6:] + ".h5"
                        , "GT" : i + "/" + i[-6:] + "_GT.h5",
                        "bvals" : i + "/T1w/Diffusion/bvals" , "bvecs" : i + "/T1w/Diffusion/bvecs"}
    path = {'3T': path_3t, "7T":  path_7t}
    p = list(path_7t.keys())
    q = list(path_3t.keys())
    common = list(set(p) & set(q))

    return path,len(common)

def load_data(base_dir,ids):
    ids.sort()
    path,tot = load_path(base_dir,ids)
    act_ids = []
    for i in ids:
        name = path['3T'][i]['h5']
        if(not os.path.isfile(name)):
            continue
        
        res_vol = h5py.File(name, 'r')
        
        name = path['3T'][i]['GT']
        res = h5py.File(name, 'r')
        
        loaded[i] = {'vol0':res_vol.get('volumes0')[:]
                            ,'mask':res_vol.get('mask')[:],
                    'ADC':res.get('ADC')[:],
                    'FA':res.get('FA')[:] ,
                    'color_FA':res.get('color_FA')[:]}
        
        res_vol.close()
        res.close()

        name = path['7T'][i]['h5']
        if(not os.path.isfile(name)):
            continue
        
        res_vol = h5py.File(name, 'r')
        
        name = path['7T'][i]['GT']
        res = h5py.File(name, 'r')
        
        
        loaded_gt[i] = {'vol0':res_vol.get('volumes0')[:]
                            ,'mask':res_vol.get('mask')[:],
                    'ADC':res.get('ADC')[:],
                    'FA':res.get('FA')[:] ,
                    'color_FA':res.get('color_FA')[:],
                    'tensor_vals':res.get('tensor_vals')[:]}
                
        res_vol.close()
        res.close()

        act_ids.append(i)
    return act_ids




def interpolate(data,size):
    
    if(len(data.shape)==3):
        inp = torch.unsqueeze(data, 0)
    else:
        inp = torch.permute(data, (3,0,1,2))
    inp = torch.unsqueeze(inp, 0)
    interpolated = torch.nn.functional.interpolate(inp,size = torch.Size(size))
    interpolated = torch.permute(interpolated, (2,3,4,1,0))
    
    if(len(data.shape)==3):
        interpolated = torch.squeeze(interpolated)
    return torch.squeeze(interpolated)

class hcp_data(torch.utils.data.Dataset):
    def __init__(self, opt,ids):
        super(hcp_data).__init__()
        
        self.blk_size = opt.block_size
        
        self.thres = opt.thres
        self.base_dir = opt.dir if opt.dir != None else "/storage/users/arihant"
        self.ids = ids
        self.debug = opt.debug

        self.enable_thres = opt.enable_thres
        self.batch_size = opt.batch_size
        
        if(opt.sort == True):
            self.ids.sort()
        
        self.preload_data()


    def __len__(self):
        return self.blk_indx[-1]
        
    def __getitem__(self,indx):

        blk_idx = np.searchsorted(self.blk_indx, indx)
        vol_idx = self.ids[blk_idx]
        if(blk_idx == 0):
            blk_idx = indx
        else:
            blk_idx = indx - self.blk_indx[blk_idx-1] - 1

        return self.collate(vol_idx,blk_idx)        

    def collate(self,vol_idx,blk_idx):
        
        data = self.loaded_blk[vol_idx][blk_idx],self.loaded_adc[vol_idx][blk_idx],self.loaded_fa[vol_idx][blk_idx],self.loaded_rgb[vol_idx][blk_idx]

        inp = torch.from_numpy(np.stack(data[0]))
        
        hr = torch.from_numpy(np.concatenate([np.expand_dims(data[1],axis = 3),np.expand_dims(data[2],axis = 3),data[3]], axis = 3))
        
        return inp,hr

    def preload_data(self):
            
        self.blk_indx = []
        self.loaded_blk = {}
        self.loaded_blk_hr = {}
        self.loaded_adc = {}
        self.loaded_fa = {}
        self.loaded_rgb = {}
        self.loaded_lr_maps = {}
            
        for i in self.ids:
            self.loaded_blk[i],self.loaded_adc[i],self.loaded_fa[i],self.loaded_rgb[i],self.loaded_blk_hr[i],self.loaded_lr_maps[i] = self.pre_proc(i)
            
        self.blk_indx = np.cumsum(self.blk_indx)

    def permute_dir(self,data,x):
        if(len(data.shape) == 3):
            
            if x == 2:
                data = torch.permute(data,(1,2,0))
            if x == 1:
                data =torch.permute(data,(0,2,1))
        else:
            if x == 2:
                data = torch.permute(data,(1,2,0,3))
            if x == 1:
                data =torch.permute(data,(0,2,1,3))
        return data

    def blk_points_pair(self,datalr,datahr,adc,fa,rgb,blk_size = (32,32,5)):
        
        shpind = torch.nonzero(datalr)
        xmin,xmax = torch.min(shpind[:,0]).item(),torch.max(shpind[:,0]).item()
        ymin,ymax = torch.min(shpind[:,1]).item(),torch.max(shpind[:,1]).item()
        zmin,zmax = torch.min(shpind[:,1]).item(),torch.max(shpind[:,2]).item()

        lr_start = [xmin,ymin,zmin]

        ranges = {}
        ranges_key = blk_size,(blk_size[0],blk_size[2],blk_size[1]),(blk_size[2],blk_size[1],blk_size[0])

        lr_end = [xmax - ranges_key[0][0] + 1,ymax - ranges_key[0][1] + 1,zmax - ranges_key[0][2] + 1]
        ranges[ranges_key[0]] = [np.arange(lr_start[i], lr_end[i], ranges_key[0][i]) for i in range(3)]


        lr_end = [xmax - ranges_key[1][0] + 1,ymax - ranges_key[1][1] + 1,zmax - ranges_key[1][2] + 1]
        ranges[ranges_key[1]] = [np.arange(lr_start[i], lr_end[i], ranges_key[1][i]) for i in range(3)]

        lr_end = [xmax - ranges_key[2][0] + 1,ymax - ranges_key[2][1] + 1,zmax - ranges_key[2][2] + 1]
        ranges[ranges_key[2]] = [np.arange(lr_start[i], lr_end[i], ranges_key[2][i]) for i in range(3)]
        
        sample_req = 50
        samples = np.random.randint(3, size=sample_req)
        blks = {"lr":[],"adc":[],"fa":[],"rgb":[],"hr":[]}
        for curr_dir in samples:    
            temp = [len(i) for i in ranges[ranges_key[curr_dir]]]
            ii,jj,kk = [np.random.randint(i,size=1)[0] for i in temp]
            curr_blk = ranges_key[curr_dir]
            curr_ranges = ranges[ranges_key[curr_dir]]
            x,y,z = curr_ranges[0][ii],curr_ranges[1][jj],curr_ranges[2][kk]
            temp_lr = np.array([x, x + curr_blk[0]-1, 
                                y, y + curr_blk[1]-1, 
                                z, z + curr_blk[2]-1]).astype(int)
            
            blks["lr"].append(self.permute_dir(datalr[temp_lr[0]:temp_lr[1]+1, temp_lr[2]:temp_lr[3]+1, temp_lr[4]:temp_lr[5]+1, ...],curr_dir))
            blks["hr"].append(self.permute_dir(datahr[temp_lr[0]:temp_lr[1]+1, temp_lr[2]:temp_lr[3]+1, temp_lr[4]:temp_lr[5]+1, ...],curr_dir))
            blks["adc"].append(self.permute_dir(adc[temp_lr[0]:temp_lr[1]+1, temp_lr[2]:temp_lr[3]+1, temp_lr[4]:temp_lr[5]+1, ...],curr_dir))
            blks["fa"].append(self.permute_dir(fa[temp_lr[0]:temp_lr[1]+1, temp_lr[2]:temp_lr[3]+1, temp_lr[4]:temp_lr[5]+1, ...],curr_dir))
            blks["rgb"].append(self.permute_dir(rgb[temp_lr[0]:temp_lr[1]+1, temp_lr[2]:temp_lr[3]+1, temp_lr[4]:temp_lr[5]+1, ...],curr_dir))

        return blks,len(blks["lr"])


    def pre_proc(self,idx):

        vol_lr = torch.from_numpy(loaded[idx]['vol0'])

        vol_hr = torch.from_numpy(loaded_gt[idx]['vol0'])

        size = vol_lr.shape[:3]
        vol_hr = interpolate(vol_hr,size)
        
        adc = interpolate(torch.from_numpy(loaded_gt[idx]['ADC']),size)
        fa = interpolate(torch.from_numpy(loaded_gt[idx]['FA']),size)
        rgb = interpolate(torch.from_numpy(loaded_gt[idx]['color_FA']),size)


        adc_lr = torch.from_numpy(loaded[idx]['ADC'])
        fa_lr = torch.from_numpy(loaded[idx]['FA'])
        rgb_lr = torch.from_numpy(loaded[idx]['color_FA'])


        res,num = self.blk_points_pair(vol_lr,vol_hr,adc,fa,rgb)

        self.blk_indx.append(num)
        
        res_lr,num = self.blk_points_pair(vol_lr,vol_hr,adc_lr,fa_lr,rgb_lr)

        return res['lr'],res['adc'],res['fa'],res['rgb'],res['hr'],res_lr