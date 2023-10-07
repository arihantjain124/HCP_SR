from importlib import import_module
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data.HCP_dataset_h5_arb as HCP_dataset

class Data:
    def __init__(self, args,ids):
        
        self.dataset_hcp = HCP_dataset
        self.pin_mem = args.pin_mem
        self.batch_size = args.batch_size
        self.ids = self.dataset_hcp.load_data(args.dir,ids)
        self.train_vols = int(len(self.ids) * (args.train_set))
        self.testing_dataset = self.dataset_hcp.hcp_data_test_recon(args,self.ids[self.train_vols:],test = True)
        self.training_dataset = self.dataset_hcp.hcp_data(args,self.ids[:self.train_vols])
        scale = (1,1,1)
        self.training_dataset.set_scale(scale = scale,blk_size = args.block_size)
        self.testing_dataset.set_scale(scale = scale,blk_size = args.test_block_size)
        self.testing_data = DataLoader(dataset=self.testing_dataset, batch_size=1,pin_memory=self.pin_mem,collate_fn=self.resize_test)
        self.training_data = DataLoader(dataset=self.training_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=self.pin_mem, drop_last=True,collate_fn=self.resize)
        print("Loading Done")



    def rebuild(self,scale,blk_size,test_blk_size,train=False,test=False):
        if train:
            self.training_dataset.set_scale(scale = scale,blk_size = blk_size)
            self.training_data = DataLoader(dataset=self.training_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_mem, drop_last=True,collate_fn=self.resize)
        if test:
            self.testing_dataset.set_scale(scale = scale,blk_size = test_blk_size)
            self.testing_data = DataLoader(dataset=self.testing_dataset, batch_size=1,pin_memory=self.pin_mem,collate_fn=self.resize_test)

    def resize(self,data):
        x,y = [],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(np.concatenate([np.expand_dims(data[i][1],axis = 3),np.expand_dims(data[i][2],axis = 3),data[i][3]], axis=3))
            
        lr = torch.from_numpy(np.stack(x))
        pred = torch.from_numpy(np.stack(y))
        return lr,pred


    def resize_test(self,data):
        x,y,z,mask = [],[],[],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(np.concatenate([np.expand_dims(data[i][1],axis = 3),np.expand_dims(data[i][2],axis = 3),data[i][3]], axis=3))
            z.append(data[i][4])
            mask.append(data[i][5])
        lr = torch.from_numpy(np.stack(x)).squeeze()
        pred = torch.from_numpy(np.stack(y)).squeeze()
        pnt = torch.from_numpy(np.stack(z)).squeeze()
        mask = torch.from_numpy(np.stack(mask)).squeeze()
        return lr,pred,pnt,mask


