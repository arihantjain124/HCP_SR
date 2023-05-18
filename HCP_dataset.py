import torch.utils.data as data
import os
import numpy as np
import torch
import random
import math
from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
    
class hcp_data(data.Dataset):
    def __init__(self, opt):
        
        self.blk_size = opt.size
        self.crop_depth = opt.crop_depth
        self.base_dir = opt.dir if opt.dir != None else "/workspace/data" 
        self.path,self.tot_vol,self.rand_sample = self.load_data(self.base_dir)
        self.data_3t = self.load_volume(self.rand_sample,'3T',self.crop_depth)
        self.data_7t = self.load_volume(self.rand_sample,'7T',self.crop_depth)
        base_mask = self.data_3t[1]
        self.mul = np.array(self.data_7t[0].shape)/np.array(self.data_3t[0].shape)
        self.blk_per_vols = self.blocks(base_mask)



    def load_data(self,base_dir):
        base_dir_7t = [base_dir + "/HCP_7T/" + i   for i in os.listdir(base_dir + "/HCP_7T") if len(i) == 6]
        base_dir_3t = [base_dir + "/HCP_3T/" + i   for i in os.listdir(base_dir + "/HCP_3T") if len(i) == 6]
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

        print("Number of Common data_id ",common)
        rand_sample = random.sample(common,1)
        return path,common,rand_sample

    def load_volume(self,id_load,res,crop = 10):

        load_from = self.path[res][id_load]
        data , affine= load_nifti(load_from["data"])
        mask,affine = load_nifti(load_from["brain_mask"])
        scan, affine = load_nifti(load_from["3d_scan"])
        grad_dev, affine = load_nifti(load_from["grad_dev"])
        bvals, bvecs = read_bvals_bvecs(load_from['bvals'], load_from['bvecs'])
        gtab = gradient_table(bvals, bvecs)
        if(res == '7T'):
            return data[:,:,crop*2:-crop*2,:],mask[:,:,crop*2:-crop*2],scan,gtab,grad_dev
        else:
            return data[:,:,crop:-crop,:],mask[:,:,crop:-crop],scan,gtab,grad_dev
    
    def blocks(self,base_mask):
        # %% divide brain volume to blocks
        xind,yind,zind = np.nonzero(base_mask)
        xmin,xmax = np.min(xind),np.max(xind)
        ymin,ymax = np.min(yind),np.max(yind)
        zmin,zmax = np.min(zind),np.max(zind)

        ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax] 

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

        return ind_block,len(ind_block)

    def __len__(self):
        return self.blk_per_vols[1] * len(self.tot_vol)
        
    def _get_patch(self,img_indx,patch_indx,axs,typ = 'single'):
        pass
    
    def __getitem__(self, idx):
        
        pass