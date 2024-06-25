import torch
import torch.nn.functional as F
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
        
        loaded[i] = {'vol0':res_vol.get('volumes0')[:],
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
                    'color_FA':res.get('color_FA')[:]}
                
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
    def __init__(self, opt,ids,test=False):
        super(hcp_data).__init__()
        
        self.blk_size = opt.block_size
        self.var_blk_size = opt.start_var
        
        self.thres = opt.thres
        self.base_dir = opt.dir if opt.dir != None else "/storage/users/arihant"
        self.ids = ids
        self.debug = opt.debug

        self.sca = 0
        self.asy = 0
        self.var = 0

        self.enable_thres = opt.enable_thres
        self.type = opt.type

        self.tv = opt.tv

        self.transform = tio.transforms.RescaleIntensity(masking_method=lambda x: x > 0)

        self.batch_size = opt.batch_size
        self.scale_const = None
        
        if(opt.sort == True):
            self.ids.sort()
            
        self.test = test
        
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
        
        

    def _make_pos_encoding(self,blk): 
        
        blk = [ [i.item() for i in list(blk[j]) ] for j in range(len(blk))]
        res = []
        for n in range(len(blk)):   
            blk_x1,blk_x2,blk_y1,blk_y2,blk_z1,blk_z2 = blk[n]

            if(self.type == '2d'):

                if(blk_x1 - blk_x2 == 0):
                    blk_x1,blk_x2 = blk_y1,blk_y2
                    blk_y1,blk_y2 = blk_z1,blk_z2
                elif(blk_y1 - blk_y2 == 0):
                    blk_y1,blk_y2 = blk_z1,blk_z2

                l = []
                for i in range(blk_x1,blk_x2+1):
                    q = []
                    for j in range(blk_y1,blk_y2+1):
                        q.append((i,j))   
                    l.append(q)
                res.append(l)

            else:

                t = []
                for i in range(blk_x1,blk_x2+1):
                    l = []
                    for j in range(blk_y1,blk_y2+1):
                        q = []
                        for k in range(blk_z1,blk_z2+1):
                            q.append((i,j,k))   
                        l.append(q)
                    t.append(l)
                res.append(t)
            

        if (self.type == '2d'):
            
            res = torch.from_numpy(np.asarray(res))
            res = torch.permute(res, (0,3,1,2))
        else:
            res = torch.from_numpy(np.asarray(res))
            res = torch.permute(res, (0,4,1,2,3))
        return res

    def collate(self,vol_idx,blk_idx):
        # print(vol_idx,blk_idx)
        
        data = self.loaded_blk[vol_idx][blk_idx],self.loaded_adc[vol_idx][blk_idx],self.loaded_fa[vol_idx][blk_idx],self.loaded_rgb[vol_idx][blk_idx]

        coor = self.blks_coor[vol_idx][blk_idx]
        # print(coor)
        coor = self._make_pos_encoding(coor)
        inp = torch.from_numpy(np.stack(data[0]))
        if (self.type == '2d'):
            dims = 3
        else:
            dims = 4
            
        hr = np.concatenate([np.expand_dims(data[1],axis = dims),np.expand_dims(data[2],axis = dims),data[3]], axis = dims)
        
        if(self.test):
            data = self.loaded_adc_lr[vol_idx][blk_idx],self.loaded_fa_lr[vol_idx][blk_idx],self.loaded_rgb_lr[vol_idx][blk_idx]
            out = np.concatenate([np.expand_dims(data[0],axis = dims),np.expand_dims(data[1],axis = dims),data[2]], axis = dims)
            return inp,hr,self.size[vol_idx],coor,out
            
        return inp,hr,self.size[vol_idx],coor
    
    def preload_data(self,test = None,args = None):
        
        if args is not None:
            self.sca,self.var,self.asy = args['sca'],args['var'],args['asy']
            if(self.var > 0 ):
                self.var_blk_size = True
        if test is not None:
            self.test = True
            
        self.blk_indx = []
        self.loaded_blk = {}
        self.loaded_adc = {}
        self.loaded_fa = {}
        self.loaded_rgb = {}
        self.loaded_adc_lr = {}
        self.loaded_fa_lr = {}
        self.loaded_rgb_lr = {}
        self.blks_coor = {}
        self.size = {}
            
                
        for i in self.ids:
            if(self.test):
                self.loaded_blk[i],self.loaded_adc[i],self.loaded_fa[i],self.loaded_rgb[i],self.blks_coor[i],self.size[i],self.loaded_adc_lr[i],self.loaded_fa_lr[i],self.loaded_rgb_lr[i] = self.pre_proc_test(i)
            else:
                self.loaded_blk[i],self.loaded_adc[i],self.loaded_fa[i],self.loaded_rgb[i],self.blks_coor[i],self.size[i] = self.pre_proc_train(i)
        
        self.blk_indx = np.cumsum(self.blk_indx)


    def extract_block(self,data, inds):
        blocks = []
        for ii in np.arange(inds.shape[0]):
            inds_this = inds[ii, :]
            curr_blk = data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, ...]
            blocks.append(curr_blk.squeeze())
        return torch.from_numpy(np.stack(blocks, axis=0))

    
    def blocks(self,data,fa,adc,rgb,blk_size = [16,16,4]):
    
        shpind = torch.nonzero(data)
        xmin,xmax = torch.min(shpind[:,0]).item(),torch.max(shpind[:,0]).item()
        ymin,ymax = torch.min(shpind[:,1]).item(),torch.max(shpind[:,1]).item()
        zmin,zmax = torch.min(shpind[:,1]).item(),torch.max(shpind[:,2]).item()

        lr_start = [xmin,ymin,zmin]
        lr_end = [xmax - blk_size[0] + 1,ymax - blk_size[1] + 1,zmax - blk_size[2] + 1]

        ranges_lr = [np.arange(lr_start[i], lr_end[i], blk_size[i]) for i in range(3)]
        
        blocks = []
        fa_blks = []
        adc_blks = []
        rgb_blks = []
        coor = []

        count = 0

        for ii in np.arange(0, ranges_lr[0].shape[0]):
            for jj in np.arange(0, ranges_lr[1].shape[0]):
                for kk in np.arange(0, ranges_lr[2].shape[0]):
                    x,y,z = ranges_lr[0][ii],ranges_lr[1][jj],ranges_lr[2][kk]
                    temp_lr = np.array([x, x + blk_size[0]-1, 
                                        y, y + blk_size[1]-1, 
                                        z, z + blk_size[2]-1]).astype(int)
                    
                    curr_blk_hr = data[temp_lr[0]:temp_lr[1]+1, temp_lr[2]:temp_lr[3]+1, temp_lr[4]:temp_lr[5]+1, ...]
                    if(self.enable_thres):
                        if((torch.numel(curr_blk_hr) != 0 and torch.count_nonzero(curr_blk_hr)/torch.numel(curr_blk_hr) > self.thres)):
                            blocks.append(data[temp_lr[0]:temp_lr[1]+1, temp_lr[2]:temp_lr[3]+1, temp_lr[4]:temp_lr[5]+1, ...])
                            fa_blks.append(fa[temp_lr[0]:temp_lr[1]+1, temp_lr[2]:temp_lr[3]+1, temp_lr[4]:temp_lr[5]+1, ...])
                            adc_blks.append(adc[temp_lr[0]:temp_lr[1]+1, temp_lr[2]:temp_lr[3]+1, temp_lr[4]:temp_lr[5]+1, ...])
                            rgb_blks.append(rgb[temp_lr[0]:temp_lr[1]+1, temp_lr[2]:temp_lr[3]+1, temp_lr[4]:temp_lr[5]+1, ...])
                            
                            coor.append(temp_lr)

                    # if(self.enable_thres):
                    #     if((torch.numel(curr_blk_hr) != 0 and torch.count_nonzero(curr_blk_hr)/torch.numel(curr_blk_hr) > self.thres)):
                    #         ind_block_lr.append(temp_lr)
                    #         ind_block_hr.append(temp_hr)
                    #         count = count + 1
                    
        blocks = torch.from_numpy(np.stack(blocks, axis=0)).squeeze()
        fa_blks = torch.from_numpy(np.stack(fa_blks, axis=0)).squeeze()
        adc_blks = torch.from_numpy(np.stack(adc_blks, axis=0)).squeeze()
        rgb_blks = torch.from_numpy(np.stack(rgb_blks, axis=0)).squeeze()
        coor = torch.from_numpy(np.stack(coor, axis=0)).squeeze()

        return blocks,adc_blks,fa_blks,rgb_blks,coor

    def norm(self,data):
        if(len(data.size())<4):
            temp = self.transform(torch.unsqueeze(data,0))
            return torch.squeeze(temp)
        return self.transform(data)
    
    def size_scale_set(self,idx):

        if self.var_blk_size:
            
            x = np.around(np.random.uniform(1,1+self.sca),decimals=1)
            
            curr_scale = np.around(np.random.uniform(x,x+self.asy,3),decimals=1)

            curr_blk_size = [int(np.random.uniform(32-self.var,32+self.var)) for i in range(2)]
            if(self.type == '2d'):
                curr_blk_size.append(1)
            else:    
                z_var = int(self.var//4)
                curr_blk_size.append(int(np.random.uniform(8-z_var,8+z_var)))
            ### Randomizing AXES
            curr_blk_size = list(set(permutations(curr_blk_size)))[np.random.randint(0,3)]
            ###

            if(self.debug):
                print(idx,curr_blk_size)
            
        else:
            if self.scale_const is None:
                x = np.around(np.random.uniform(1,1+self.sca),decimals=1)
                asy = self.asy
                curr_scale = np.around(np.random.uniform(x-asy,x+asy,3),decimals=1)
            else:
                curr_scale = self.scale_const
            
            curr_blk_size = list(self.blk_size)
            if(self.type == '2d'):
                curr_blk_size[-1] = 1

            curr_blk_size = list(set(permutations(curr_blk_size)))[np.random.randint(0,3)]
            
        
        if(min(curr_blk_size) == 1):
            curr_scale[np.where(np.asarray(curr_blk_size) == 1)[0][0]] = 1


        return curr_scale,curr_blk_size
    
    def pre_proc_train(self,idx):

        vol = torch.from_numpy(loaded_gt[idx]['vol0'])
        adc = torch.from_numpy(loaded_gt[idx]['ADC'])
        fa  = torch.from_numpy(loaded_gt[idx]['FA'])
        rgb = torch.from_numpy(loaded_gt[idx]['color_FA'])

        curr_scale,curr_blk_size = self.size_scale_set(idx)
        
        curr_blk = self.blocks(vol,adc,fa,rgb,blk_size=curr_blk_size)
        
        drop_last = (len(curr_blk[0])//self.batch_size)*self.batch_size

        blks_img = torch.split(curr_blk[0][:drop_last,...],self.batch_size)
        blks_adc = torch.split(curr_blk[1][:drop_last,...],self.batch_size)
        blks_fa = torch.split(curr_blk[2] [:drop_last,...],self.batch_size)
        blks_rgb = torch.split(curr_blk[3][:drop_last,...],self.batch_size)
        blks_coor = torch.split(curr_blk[4][:drop_last,...],self.batch_size)
        
        size = [int(curr_blk_size[i]//curr_scale[i]) for i in range(len(curr_blk_size))]

        if(min(size) == 1):
            size.remove(1)

        # print(len(blks_img),idx)
        # print(type(size),size,blks_img[0].shape)
        
        
        downsample = []
        for i in range(len(blks_img)):

            if(len(blks_img[i].shape) == 5):
                blk = blks_img[i].permute(0,4,1,2,3)# blk.shape
                blk = F.interpolate(blk,size = size)
                blk = blk.permute(0,2,3,4,1)
                
            else:
                blk = blks_img[i].permute(0,3,1,2)# blk.shape
                blk = F.interpolate(blk,size = size)
                blk = blk.permute(0,2,3,1)

            downsample.append(blk)

        self.blk_indx.append(len(blks_img)-1)


        return downsample,blks_adc,blks_fa,blks_rgb,blks_coor,size

        
    def extract_block(self,data, inds):
            blocks = []
            for ii in np.arange(inds.shape[0]):
                inds_this = inds[ii, :]
                curr_blk = data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, ...]
                blocks.append(curr_blk.squeeze())
            return torch.from_numpy(np.stack(blocks, axis=0))

    def pre_proc_test(self,idx):
        
        curr_scale,curr_blk_size = self.size_scale_set(idx)
        
        vol = torch.from_numpy(loaded[idx]['vol0'])
        adc = torch.from_numpy(loaded_gt[idx]['ADC'])
        fa  = torch.from_numpy(loaded_gt[idx]['FA'])
        rgb = torch.from_numpy(loaded_gt[idx]['color_FA'])
        
        adc_lr = interpolate(torch.from_numpy(loaded[idx]['ADC']),vol.shape[:3])
        fa_lr = interpolate(torch.from_numpy(loaded[idx]['FA']),vol.shape[:3])
        rgb_lr = interpolate(torch.from_numpy(loaded[idx]['color_FA']),vol.shape[:3])

        curr_blk = self.blocks(vol,adc,fa,rgb,blk_size=curr_blk_size)
        
        adc_lr = self.extract_block(adc_lr,curr_blk[4])
        fa_lr  = self.extract_block(fa_lr,curr_blk[4])
        rgb_lr = self.extract_block(rgb_lr,curr_blk[4])

        drop_last = (len(curr_blk[0])//self.batch_size)*self.batch_size

        blks_img = torch.split(curr_blk[0][:drop_last,...],self.batch_size)
        blks_adc = torch.split(curr_blk[1][:drop_last,...],self.batch_size)
        blks_fa = torch.split(curr_blk[2] [:drop_last,...],self.batch_size)
        blks_rgb = torch.split(curr_blk[3][:drop_last,...],self.batch_size)
        blks_coor = torch.split(curr_blk[4][:drop_last,...],self.batch_size)

        
        blks_adc_lr = torch.split(adc_lr[:drop_last,...],self.batch_size)
        blks_fa_lr  = torch.split(fa_lr[:drop_last,...],self.batch_size)
        blks_rgb_lr = torch.split(rgb_lr[:drop_last,...],self.batch_size)


        size = [int(curr_blk_size[i]//curr_scale[i]) for i in range(len(curr_blk_size))]

        if(min(size) == 1):
            size.remove(1)
            
        downsample = []
        for i in range(len(blks_img)):

            if(len(blks_img[i].shape) == 5):
                blk = blks_img[i].permute(0,4,1,2,3)# blk.shape
                blk = F.interpolate(blk,size = size)
                blk = blk.permute(0,2,3,4,1)
                
            else:
                blk = blks_img[i].permute(0,3,1,2)# blk.shape
                blk = F.interpolate(blk,size = size)
                blk = blk.permute(0,2,3,1)

            downsample.append(blk)

        self.blk_indx.append(len(blks_img)-1)


        return downsample,blks_adc,blks_fa,blks_rgb,blks_coor,size,blks_adc_lr,blks_fa_lr,blks_rgb_lr

        
        



