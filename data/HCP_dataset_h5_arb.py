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
#     print(data.shape)
    if(len(data.shape)==3):
        inp = torch.unsqueeze(data, 0)
#         print(inp.shape)
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
        # self.psnr_sim = opt.psnr_sim

        self.transform = tio.transforms.RescaleIntensity(masking_method=lambda x: x > 0)
        self.tv_transform = tio.transforms.RescaleIntensity(out_min_max = (-1,1))
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

#         print(vol_idx,blk_idx)
        if(self.debug):
            return self.collate(vol_idx,blk_idx),(self.blks_ret_lr[vol_idx][blk_idx],self.blks_ret_hr[vol_idx][blk_idx])
        else:    
            return self.collate(vol_idx,blk_idx)
        

    def _make_pos_encoding(self,blk): 

        blk = [ [i.item() for i in list(blk[j]) ] for j in range(len(blk))]
        # print(blk)
        # return 0
        res = []
        for n in range(len(blk)):   
            blk_x1,blk_x2,blk_y1,blk_y2,blk_z1,blk_z2 = blk[n]

            if(self.type == '2d'):

                if(blk_x1 - blk_x2 == 0):
                    blk_x1,blk_x2 = blk_z1,blk_z2
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
        # print(blk,res.shape)
        return res

    def collate(self,vol_idx,blk_idx):
        
        data = self.loaded_blk[vol_idx][blk_idx],self.loaded_adc[vol_idx][blk_idx],self.loaded_fa[vol_idx][blk_idx],self.loaded_rgb[vol_idx][blk_idx]

        coor = self.blks_ret_lr[vol_idx][blk_idx],self.blks_ret_hr[vol_idx][blk_idx]

        coor_hr = self._make_pos_encoding(coor[1])

        inp = torch.from_numpy(np.stack(data[0]))
        if (self.type == '2d'):
            dims = 3
        else:
            dims = 4
            
        hr = np.concatenate([np.expand_dims(data[1],axis = dims),np.expand_dims(data[2],axis = dims),data[3]], axis = dims)
        
        if(self.test):
            data = self.loaded_adc_lr[vol_idx][blk_idx],self.loaded_fa_lr[vol_idx][blk_idx],self.loaded_rgb_lr[vol_idx][blk_idx]
            out = np.concatenate([np.expand_dims(data[0],axis = dims),np.expand_dims(data[1],axis = dims),data[2]], axis = dims)
            return inp,hr,self.scale[vol_idx],coor_hr,out
        
        elif(self.tv):
            
            tv = self.loaded_tv[vol_idx][blk_idx]
            return inp,hr,self.scale[vol_idx],(coor_hr,tv)

        else:
            # tv = torch.from_numpy(np.stack(self.loaded_tv[vol_idx][blk_idx]))
            return inp,hr,self.scale[vol_idx],coor_hr
    
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
        self.scale = {}
        self.blks_ret_lr = {}
        self.blks_ret_hr = {}
        
        
        if(self.test):
            self.loaded_adc_lr = {}
            self.loaded_fa_lr = {}
            self.loaded_rgb_lr = {}
        else:
            self.loaded_tv = {}
            
                
        for i in self.ids:
            if(self.test):
                self.loaded_blk[i],self.loaded_adc[i],self.loaded_fa[i],self.loaded_rgb[i],self.scale[i],self.blks_ret_lr[i],self.blks_ret_hr[i],self.loaded_adc_lr[i],self.loaded_fa_lr[i],self.loaded_rgb_lr[i] = self.pre_proc(i)
            else:
                self.loaded_blk[i],self.loaded_adc[i],self.loaded_fa[i],self.loaded_rgb[i],self.scale[i],self.blks_ret_lr[i],self.blks_ret_hr[i],self.loaded_tv[i] = self.pre_proc(i)
            if(self.debug == True):
                print(i,"loaded")
        self.blk_indx = np.cumsum(self.blk_indx)

    
    def blk_points_pair(self,datalr,datahr,blk_size = [16,16,4],stride = (0,0,0),scale = (1,1,1),vol_lr = None):
    
        shpind = torch.nonzero(datalr)
        xmin,xmax = torch.min(shpind[:,0]).item(),torch.max(shpind[:,0]).item()
        ymin,ymax = torch.min(shpind[:,1]).item(),torch.max(shpind[:,1]).item()
        zmin,zmax = torch.min(shpind[:,1]).item(),torch.max(shpind[:,2]).item()

        lr_start = [xmin,ymin,zmin]
        lr_end = [xmax - blk_size[0] + 1,ymax - blk_size[1] + 1,zmax - blk_size[2] + 1]

        shpind = torch.nonzero(datahr)
        xmin,xmax = torch.min(shpind[:,0]).item(),torch.max(shpind[:,0]).item()
        ymin,ymax = torch.min(shpind[:,1]).item(),torch.max(shpind[:,1]).item()
        zmin,zmax = torch.min(shpind[:,1]).item(),torch.max(shpind[:,2]).item()

        blk_size_hr = [round(blk_size[i]*scale[i]) for i in range(3)]
        hr_start = [xmin,ymin,zmin]
        hr_end = [xmax - blk_size_hr[0] + 1,ymax - blk_size_hr[1] + 1,zmax - blk_size_hr[2] + 1]

        a,b = [lr_end[i] - lr_start[i] for i in range(3)],[hr_end[i] - hr_start[i] for i in range(3)]
        offset = [round(b[i]/a[i],1) for i in range(3)]
        ranges_lr = [np.arange(lr_start[i], lr_end[i], blk_size[i] - stride[i]) for i in range(3)]
        ranges_hr = [np.round(ranges_lr[i]*offset[i]) for i in range(3)]
        
        # misc = {'offset': offset,
        #         'hr_pts': (hr_start,hr_end)}
        
        ind_block_lr = []
        ind_block_hr = []
        count = 0

        for ii in np.arange(0, ranges_lr[0].shape[0]):
            for jj in np.arange(0, ranges_lr[1].shape[0]):
                for kk in np.arange(0, ranges_lr[2].shape[0]):
                    x,y,z = ranges_lr[0][ii],ranges_lr[1][jj],ranges_lr[2][kk]
                    temp_lr = np.array([x, x + blk_size[0]-1, 
                                        y, y + blk_size[1]-1, 
                                        z, z + blk_size[2]-1]).astype(int)
                    
                    x,y,z = ranges_hr[0][ii],ranges_hr[1][jj],ranges_hr[2][kk]
                    temp_hr = np.array([x, x + blk_size_hr[0]-1,
                                        y, y + blk_size_hr[1]-1,
                                        z, z + blk_size_hr[2]-1]).astype(int)
                    
                    
                    curr_blk = datalr[temp_lr[0]:temp_lr[1]+1, temp_lr[2]:temp_lr[3]+1, temp_lr[4]:temp_lr[5]+1, ...]
                    curr_blk_lr = vol_lr[temp_lr[0]:temp_lr[1]+1, temp_lr[2]:temp_lr[3]+1, temp_lr[4]:temp_lr[5]+1, ...]

                    curr_blk_hr = datahr[temp_hr[0]:temp_hr[1]+1, temp_hr[2]:temp_hr[3]+1, temp_hr[4]:temp_hr[5]+1, ...]
                    # psnr_sim = float(metrics.peak_signal_noise_ratio(curr_blk.numpy(),curr_blk_lr.numpy(),data_range=1))


                    if(self.enable_thres):
                        if((torch.numel(curr_blk_hr) != 0 and torch.count_nonzero(curr_blk_hr)/torch.numel(curr_blk_hr) > self.thres)):
                            ind_block_lr.append(temp_lr)
                            ind_block_hr.append(temp_hr)
                            count = count + 1
                    else:
                        ind_block_lr.append(temp_lr)
                        ind_block_hr.append(temp_hr)
                        count = count + 1
                    
        ind_block_lr = np.stack(ind_block_lr)
        ind_block_lr = ind_block_lr.astype(int)
        ind_block_hr = np.stack(ind_block_hr)
        ind_block_hr = ind_block_hr.astype(int)
        
        return ind_block_lr,ind_block_hr,len(ind_block_lr)


    def extract_block(self,data, inds):
            blocks = []
            for ii in np.arange(inds.shape[0]):
                inds_this = inds[ii, :]
                curr_blk = data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, ...]
                blocks.append(curr_blk.squeeze())
            return torch.from_numpy(np.stack(blocks, axis=0))

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
    
    def pre_proc(self,idx):

        vol = torch.from_numpy(loaded[idx]['vol0'])
        vol_lr = interpolate(torch.from_numpy(loaded_gt[idx]['vol0']),vol.shape[:3])

        curr_scale,curr_blk_size = self.size_scale_set(idx)

        size = [int(curr_scale[i] * vol.shape[i]) for i in range(3)]
        

        # print(curr_blk_size,curr_scale)

        vol_hr = interpolate(torch.from_numpy(loaded_gt[idx]['vol0']),size)
        adc = interpolate(torch.from_numpy(loaded_gt[idx]['ADC']),size)
        fa = interpolate(torch.from_numpy(loaded_gt[idx]['FA']),size)
        rgb = interpolate(torch.from_numpy(loaded_gt[idx]['color_FA']),size)


        ## testing
        tv = interpolate(torch.from_numpy(loaded_gt[idx]['tensor_vals']),size)
        tv = self.tv_transform(tv)
        # tv = torch.from_numpy(loaded_gt[idx]['tensor_vals'])
        

        curr_blk = self.blk_points_pair(vol,vol_hr,blk_size=curr_blk_size,scale=curr_scale,vol_lr = vol_lr)
        
        drop_last = (curr_blk[2]//self.batch_size)*self.batch_size
        
        
        blks_img = torch.split(self.extract_block(vol,curr_blk[0])[:drop_last,...],self.batch_size)
        blks_adc = torch.split(self.extract_block(adc,curr_blk[1])[:drop_last,...],self.batch_size)
        blks_fa = torch.split(self.extract_block(fa,curr_blk[1])[:drop_last,...],self.batch_size)
        blks_rgb = torch.split(self.extract_block(rgb,curr_blk[1])[:drop_last,...],self.batch_size)
        blks_tv = torch.split(self.extract_block(tv,curr_blk[1])[:drop_last,...],self.batch_size)
        # print(len(blks_rgb))

        ## Co - ordinates for the cropping 
        curr_blk_lr = torch.split(torch.from_numpy(curr_blk[0])[:drop_last,...],self.batch_size)
        curr_blk_hr = torch.split(torch.from_numpy(curr_blk[1])[:drop_last,...],self.batch_size)
        
        self.blk_indx.append((curr_blk[2]//self.batch_size)-1)

#         print(blks_rgb.shape)

        if(self.test):
            blks_lr_adc = torch.split(self.extract_block(torch.from_numpy(loaded[idx]['ADC']),curr_blk[0])[:drop_last,...],self.batch_size)
            blks_lr_fa = torch.split(self.extract_block(torch.from_numpy(loaded[idx]['FA']),curr_blk[0])[:drop_last,...],self.batch_size)
            blks_lr_rgb = torch.split(self.extract_block(torch.from_numpy(loaded[idx]['color_FA']),curr_blk[0])[:drop_last,...],self.batch_size)
            # print(blks_lr_adc.shape,blks_lr_fa.shape,blks_lr_rgb.shape)
            return blks_img,blks_adc,blks_fa,blks_rgb,curr_scale,curr_blk_lr,curr_blk_hr,blks_lr_adc,blks_lr_fa,blks_lr_rgb
        else:
            # blks_tv = torch.split(self.extract_block(tv,curr_blk[1])[:drop_last,...],self.batch_size)
            return blks_img,blks_adc,blks_fa,blks_rgb,curr_scale,curr_blk_lr,curr_blk_hr,blks_tv
        
        # return blks_img,blks_adc,blks_fa,blks_rgb,blks_tv,curr_scale,curr_blk_lr,curr_blk_hr

