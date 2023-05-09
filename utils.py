from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.linalg import eig

base_dir = "/storage/users/arihant"
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
    
    
path = {'3T': path_3t, "7T": path_7t}
p = list(path_7t.keys())
q = list(path_3t.keys())
common = list(set(p) & set(q))

print("number of common Subjects ",len(common))


def load_hcp(id_load,res,ret_img = False,crop = 10):
    load_from = path[res][id_load]
    if ret_img:
        data , affine, img = load_nifti(load_from["data"], return_img=ret_img)
    else:
        data , affine= load_nifti(load_from["data"], return_img=ret_img)
    mask,affine = load_nifti(load_from["brain_mask"], return_img=ret_img)
    scan, affine = load_nifti(load_from["3d_scan"], return_img=False)
    
    bvals, bvecs = read_bvals_bvecs(load_from['bvals'], load_from['bvecs'])
    gtab = gradient_table(bvals, bvecs)
    
    return data[:,:,crop:-crop,:],mask[:,:,crop:-crop],scan,gtab

def mean_volume(data,gtab,b):
    if b not in gtab.bvals:
        print("invalid b value")
        return None
    else:
        return np.mean(data , axis = 3 ,where = gtab.bvals == b)
    
    
def rot3d(arg):
    x,y,z = arg[0],arg[1],arg[2]
    Rx = np.array([[1 ,0 ,0 ],[0,np.cos(x),-np.sin(x)],[0 ,np.sin(x) ,np.cos(x)]])
    Ry = np.array([[np.cos(y),0 ,np.sin(y) ],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    Rz = np.array([[np.cos(z) ,-np.sin(z) ,0 ],[np.sin(z),np.cos(z),0],[0 ,0,1]])
    R = Rx @ Ry @ Rz
    return R

def amatrix(mat):
    
    a = [mat[:,0] * mat[:,0],2 * mat[:,0] * mat[:,1], 2* mat[:,0] * mat[:,2],
        mat[:,1] * mat[:,1],2 * mat[:,1] * mat[:,2], mat[:,2] * mat[:,2]]
    return np.array(a).T


def dtimetric(tensor,mask):
    ret = {}
    mask = mask >0.1
    sz = mask.shape
    v1 = np.zeros((sz[0],sz[1],sz[2],3))
    v2 = np.zeros((sz[0],sz[1],sz[2],3))
    v3 = np.zeros((sz[0],sz[1],sz[2],3))
    l1 = np.zeros((sz))
    l2 = np.zeros((sz))
    l3 = np.zeros((sz))
    md = np.zeros((sz))
    rd = np.zeros((sz))
    fa = np.zeros((sz))
    
    for i in range(sz[0]):
        for j in range(sz[1]):
            for k in range(sz[2]):
                if (mask[i,j,k]):
                    
                    
                    tensor_vox = tensor[i,j,k,:].squeeze()
                    tensor_mtx = np.zeros((3,3))
                    tensor_mtx[0,0] = tensor_vox[0]
                    tensor_mtx[0,1] = tensor_vox[1];tensor_mtx[1,0] = tensor_vox[1]
                    tensor_mtx[0,2] = tensor_vox[2];tensor_mtx[2,0] = tensor_vox[2]
                    tensor_mtx[1,1] = tensor_vox[3]
                    tensor_mtx[1,2] = tensor_vox[4];tensor_mtx[2,1] = tensor_vox[4]
                    tensor_mtx[2,2] = tensor_vox[5]
                    
                    D,V = eig(tensor_mtx)
                    MD = np.mean(D)
                    FA = np.sqrt(sum((D-MD) ** 2)) / np.sqrt(sum(D**2)) * np.sqrt(1.5)
                    
                    v1[i, j, k, :] = V[:, 2]
                    v2[i, j, k, :] = V[:, 1]
                    v3[i, j, k, :] = V[:, 0]
                    l1[i,j,k] = D[2]
                    l2[i,j,k] = D[1]
                    l3[i,j,k] = D[0]
                    fa[i,j,k] = FA
                    md[i,j,k] = MD
                    rd[i,j,k] = np.mean(D[0:1])
           
    ret['v1'] = v1
    ret['v2'] = v2
    ret['v3'] = v3
    ret['l1'] = l1
    ret['l2'] = l2
    ret['l3'] = l3
    ret['fa'] = fa
    ret['md'] = md
    ret['rd'] = rd                
    return ret
                    
    