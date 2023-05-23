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

class hcp_data(torch.utils.data.IterableDataset):
    def __init__(self, opt,ids):
        super(hcp_data).__init__()
        
        self.blk_size = opt.block_size
        self.crop_depth = opt.crop_depth
        self.base_dir = opt.dir if opt.dir != None else "/storage/users/arihant"
        self.path,self.tot = self.load_path(self.base_dir,ids)
        self.ids = ids
        if(opt.sort == True):
            self.ids.sort()
        self.curr_id = -1
        self.batch_size = opt.batch_size
        # self.path,self.tot_vol,self.rand_sample = self.load_path(self.base_dir)
        # print(self.rand_sample[0])
        # self.data_3t = self.load_volume(self.rand_sample[0],'3T',self.crop_depth)
        # self.data_7t = self.load_volume(self.rand_sample[0],'7T',self.crop_depth)
        # base_mask = self.data_3t[1]
        # self.mul = np.array(self.data_7t[0].shape)/np.array(self.data_3t[0].shape)
        # self.blk_per_vols = self.blocks(base_mask)


    def __iter__(self):
        print(self.curr_id)
        self.pre_proc()
        # if(self.curr_id == -1 or self.curr_len_blk - self.curr_indx_blk < self.batch_size):
        #     self.pre_proc()
        #     self.curr_indx_blk = 0
        #     self.iterating = self.batcher(np.concatenate((self.block_img_gt,self.block_img_pred),axis = -1),batch_size = self.batch_size)
        # else:
        #     self.curr_indx_blk+=self.batch_size
        # # print(len(np.concatenate((self.block_img_gt,self.block_img_pred),axis = -1)))
        return iter(np.concatenate((self.block_img_gt,self.block_img_pred),axis = -1))


    # def batcher(self,iterable, batch_size):
    #     # self.curr_indx_blk+=1
    #     # yield iterable[self.curr_indx_blk,...]
    #     iterator = iter(iterable)
    #     while batch := list(islice(iterator, batch_size)):
    #         yield batch

    # def block_iterator(self):
    #     yield np.concatenate((self.block_img_gt,self.block_img_pred),axis = -1)


    def load_path(self,base_dir,ids):
        base_dir_7t = [base_dir + "/HCP_7T/" + i   for i in ids]
        base_dir_3t = [base_dir + "/HCP_3T/" + i   for i in ids]
        path_7t = {}
        path_3t = {}
        for i in base_dir_7t:
            path_7t[i[-6:]] = {"3d_scan" : i + "/T1w/T1w_acpc_dc_restore_1.05.nii.gz" ,"data" : i + "/T1w/Diffusion_7T/data.nii.gz" 
                            , "bvals" : i + "/T1w/Diffusion_7T/bvals" , "bvecs" : i + "/T1w/Diffusion_7T/bvecs"
                            , "brain_mask" : i + "/T1w/Diffusion_7T/nodif_brain_mask.nii.gz"
                            , "grad_dev" : i + "/T1w/Diffusion_7T/grad_dev.nii.gz"}
        for i in base_dir_3t:
            path_3t[i[-6:]] = {"3d_scan" : i + "/T1w/T1w_acpc_dc_restore_1.25.nii.gz" , "data" : i + "/T1w/Diffusion/data.nii.gz" 
                            , "bvals" : i + "/T1w/Diffusion/bvals" , "bvecs" : i + "/T1w/Diffusion/bvecs"
                            , "brain_mask" : i + "/T1w/Diffusion/nodif_brain_mask.nii.gz"
                            , "grad_dev" : i + "/T1w/Diffusion/grad_dev.nii.gz"}
            
            
        path = {'3T': path_3t, "7T":  path_7t}
        p = list(path_7t.keys())
        q = list(path_3t.keys())
        common = list(set(p) & set(q))

        # print("Number of Common data_id ",common)
        rand_sample = random.sample(common,1)
        return path,len(common)

    def load_volume(self,id_load,res,crop = 10):
        # print(self.path[res][id_load])
        load_from = self.path[res][id_load]
        data , affine= load_nifti(load_from["data"])
        mask,affine = load_nifti(load_from["brain_mask"])
        # scan, affine = load_nifti(load_from["3d_scan"])
        # grad_dev, affine = load_nifti(load_from["grad_dev"])
        bvals, bvecs = read_bvals_bvecs(load_from['bvals'], load_from['bvecs'])
        gtab = gradient_table(bvals, bvecs)
        if(res == '7T'):
            vol = {'data' : data[:,:,self.crop_depth:-self.crop_depth,:],
                    'mask': mask[:,:,self.crop_depth:-self.crop_depth],
                    'gtab': gtab}
        else:
            vol = {'data' : data[:,:,self.crop_depth:-self.crop_depth,:],
                    'mask': mask[:,:,self.crop_depth:-self.crop_depth],
                    'gtab': gtab}
        return vol
    
    def blocks(self,base_mask):
        # %% divide brain volume to blocks
        xind,yind,zind = np.nonzero(base_mask)
        xmin,xmax = np.min(xind),np.max(xind)
        ymin,ymax = np.min(yind),np.max(yind)
        zmin,zmax = np.min(zind),np.max(zind)

        ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax] 
        # print(ind_brain)
        # calculate number of blocks along each dimension
        sz_block = self.blk_size
        xlen = xmax - xmin + 1
        ylen = ymax - ymin + 1
        zlen = zmax - zmin + 1

        nx = int(np.ceil(xlen / sz_block))
        ny = int(np.ceil(ylen / sz_block))
        nz = int(np.ceil(zlen / sz_block))

        # determine starting and ending indices of each block
        xstart = xmin
        ystart = ymin
        zstart = zmin

        xend = xmax - sz_block + 1
        yend = ymax - sz_block + 1
        zend = zmax - sz_block + 1

        xind_block = np.round(np.linspace(xstart, xend, nx))
        yind_block = np.round(np.linspace(ystart, yend, ny))
        zind_block = np.round(np.linspace(zstart, zend, nz))

        ind_block = np.zeros([xind_block.shape[0]*yind_block.shape[0]*zind_block.shape[0], 6])
        count = 0
        for ii in np.arange(0, xind_block.shape[0]):
            for jj in np.arange(0, yind_block.shape[0]):
                for kk in np.arange(0, zind_block.shape[0]):
                    ind_block[count, :] = np.array([xind_block[ii], xind_block[ii]+sz_block-1, yind_block[jj], yind_block[jj]+sz_block-1, zind_block[kk], zind_block[kk]+sz_block-1])
                    count = count + 1

        ind_block = ind_block.astype(int)
        # print(ind_block)
        return ind_block,len(ind_block)

    def extract_block(self,data, inds):

        xsz_block = inds[0, 1] - inds[0, 0] + 1
        ysz_block = inds[0, 3] - inds[0, 2] + 1
        zsz_block = inds[0, 5] - inds[0, 4] + 1
        ch_block = data.shape[-1]
        
        blocks = np.zeros((inds.shape[0], xsz_block, ysz_block, zsz_block, ch_block))
        
        for ii in np.arange(inds.shape[0]):
            inds_this = inds[ii, :]
            blocks[ii, :, :, :, :] = data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :]
        return blocks

    def pre_proc(self):
        self.curr_id+=1
        vol = self.load_volume(self.ids[self.curr_id],'3T')
        self.curr_blk = self.blocks(vol['mask'])
        # print(vol['mask'].shape)
        min_bval = min(vol['gtab'].bvals)
        print(self.ids[self.curr_id],self.curr_id)
        # print("volume loaded")

        dwis = vol['data'][:,:,:,np.where(vol['gtab'].bvals>min_bval)].squeeze()
        b0 = utils.mean_volume(vol['data'],vol['gtab'],min_bval)
        bvals = vol['gtab'].bvals[np.where(vol['gtab'].bvals>min_bval)]
        bvecs = vol['gtab'].bvecs[np.where(vol['gtab'].bvals>min_bval)]
        dwis_gt = utils.diff_coefficent(dwis,b0,bvecs,bvals,shp = vol['data'].shape,bval_synth = 1000)

        # print("gt loaded")

        idx = utils.optimal_dirs(vol['gtab'],10000,num_dirs = 5,debug = False,base_bval = min_bval)
        bval_synth = 1000
        b0s = vol['data'][:,:,:,np.where(vol['gtab'].bvals==min_bval)].squeeze()

        img_gt = np.zeros((dwis_gt.shape + (5,)))
        img_pred = np.zeros((dwis_gt.shape + (5,)))

        
        self.block_img_gt = np.zeros(( (self.curr_blk[1] * 5,) + (64,64,64) + (dwis_gt.shape[-1]+1,))) 
        self.block_img_pred = np.zeros(( (self.curr_blk[1] * 5,) + (64,64,64) + (dwis_gt.shape[-1],)))
        # print(self.block_img_gt.shape)
        # print("indexes Found")
        mask_expand = np.expand_dims(vol['mask'], 3)


        mask_block = self.extract_block(mask_expand, self.curr_blk[0]); 
        for i in range(5):
            b0 = b0s[:,:,:,i]
            dwis6 =  dwis[:,:,:,idx[i]]
            bvals6 = bvals[idx[i]]
            bvecs6 = bvecs[idx[i]]
            dwis_pred = utils.diff_coefficent(dwis6,b0,bvecs6,bvals6,shp = vol['data'].shape,bval_synth = 1000)
            
            # print(f"index {i} done")
            for jj in np.arange(0, dwis_pred.shape[-1]):  
                
                img = dwis_pred[:, :, :, jj]
                imgmean = np.mean(img[np.nonzero(vol['mask'])])
                imgstd = np.std(img[np.nonzero(vol['mask'])])

                img_pred[:,:,:,jj,i]  = (img - imgmean) / imgstd * vol['mask'] # normalize by substracting mean then dividing the std dev of brain voxels in input images
                
                img_gt[:,:,:,jj,i]  = (dwis_gt[:, :, :, jj] - imgmean) / imgstd * vol['mask']
            
            self.block_img_gt[self.curr_blk[1] * i:self.curr_blk[1] * (i+1),:,:,:,:dwis_gt.shape[-1]] = self.extract_block(img_gt[...,i],self.curr_blk[0])
            self.block_img_pred[self.curr_blk[1] * i:self.curr_blk[1] * (i+1),...] = self.extract_block(img_pred[...,i],self.curr_blk[0])
            self.block_img_gt[self.curr_blk[1] * i:self.curr_blk[1] * (i+1),:,:,:,dwis_gt.shape[-1]:dwis_gt.shape[-1]+1] = mask_block
            # last channel is brain mask, which is used to weigth loss from each voxel
        self.curr_len_blk = self.block_img_gt.shape[0]