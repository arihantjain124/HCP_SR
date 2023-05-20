from dipy.io.image import load_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.linalg import eig
from numpy.linalg import inv
from numpy.linalg import pinv
# from numpy.linalg import lstsq
from scipy.linalg import lstsq
from numpy.linalg import solve
from numpy import inf
import numpy as np

dsm = np.array([0.91, 0.416, 0,0, 0.91, 0.416,0.416, 0, 0.91,0.91, -0.416, 0,0, 0.91, -0.416,-0.416, 0, 0.91])
dsm = dsm.reshape(6,3)
dsm_norm = np.copy(dsm)
dsm_mag = np.sqrt(dsm[:,0]**2 + dsm[:,1]**2 + dsm[:,2]**2)
for i in range(3):
    dsm_norm[:,i] = dsm[:,i] / dsm_mag

def get_ids():
    return common

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
                    
def diff_coefficent(dwis,b0_img,bvecs,bvals,shp,bval_synth,base_bval = 5):
    # compute apparent diffusion coefficients
    # meanb0 = mean_volume(data,gtab,base_bval)[...,np.newaxis]
    b0_img = b0_img[...,np.newaxis]
    c = np.log(1e-3+(dwis / (1e-9+b0_img))); 
    for i in range(c.shape[3]):
        c[:,:,:,i] = c[:,:,:,i] / (-bvals[i]) # c = -In(Si/S0)/b

        
    c_vec = c.reshape(shp[0]*shp[1]*shp[2],c.shape[3]) # tx volume data to vectors
    # c_vec.dropna(inplace=True)
    c_vec = overflow_fix(c_vec)
    A = overflow_fix(amatrix(bvecs)) # Diffusion Tensor Transformation Matrix
    # print(A.shape)
    D_vec = lstsq(A,c_vec.T,cond=None)[0] # Solving for D = inv(A) * C
    # D_vec = pinv(amatrix(bvecs)) @ c_vec.T # solve tensors

    # D_img = D_vec.T.reshape(shp[0],shp[1],shp[2],6)
    # D_img = overflow_fix(D_img)
    # print(D_img.max(),D_img.min())


    # # synthesize dwis along DSM6 dirs
    D_synth = np.exp(-bval_synth * (amatrix(dsm_norm) @ D_vec))
    # print(D_synth.shape)
    dwis = b0_img * D_synth.T.reshape(shp[0],shp[1],shp[2], D_synth.shape[0]);    
    dwis = overflow_fix(dwis)
    diff_img = np.concatenate((b0_img,dwis),axis =3 )
    return diff_img


def optimal_dirs(gtab,num_iter = 10000,num_dirs = 5,debug = False,base_bval = 5):
    rotang_all = []
    angerr_all  = []
    condnum_all = []
    ind_all = []
    dirs = np.array(gtab.bvecs[np.where(gtab.bvals != base_bval)[0]])
    for i in range(0,num_iter):
        
        d = np.random.rand(1,3) * 2 * np.pi
        rotang = d[0]
        R = rot3d(rotang)
        dsm_rot = (rot3d(d[0]) @ dsm_norm.T).T
        
        ang_error = np.degrees(np.arccos(abs(dsm_rot @ dirs.T)))
        minerrors,idx = np.amin(ang_error,1),np.argmin(ang_error,1)

        mean_ang_err = np.mean(np.amin(ang_error,1))
        condnum = np.linalg.cond(amatrix(dirs[idx]))
        
        idx.sort()
        if (mean_ang_err < 5 and condnum < 1.6):
            if ((len(ind_all) == 0 ) or  len(np.where((ind_all == idx).all(axis=1))[0]) == 0 ):
                angerr_all.append(mean_ang_err)
                condnum_all.append(condnum)
                ind_all.append(idx)
                rotang_all.append(rotang)
    condnum_all = np.array(condnum_all)
    indx  = condnum_all.argsort()[:num_dirs]
    if (debug):
        print("Lowest Condition Number : ",condnum_all[indx])
    ind_use = np.array(ind_all)[indx]
    condnum_use = condnum_all[condnum_all.argsort()[:5]]
    angerr_use = np.array(angerr_all)[indx]
    rotang_use = np.array(rotang_all)[indx]
    return ind_use

def overflow_fix(data):
    data = np.nan_to_num(data)
    data[data == np.inf] = 0
    return data