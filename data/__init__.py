from importlib import import_module
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data.HCP_dataset_h5_arb as HCP_dataset

class Data:
    def __init__(self, args,ids,debug = False):
        
        self.dataset_hcp = HCP_dataset
        self.pin_mem = args.pin_mem
        self.batch_size = 1
        self.ids = self.dataset_hcp.load_data(args.dir,ids)
        self.train_vols = args.no_vols
        self.test_vols = args.test_vols

        self.training_dataset = self.dataset_hcp.hcp_data(args,self.ids[:self.train_vols])
        self.training_data = DataLoader(dataset=self.training_dataset, batch_size=1, drop_last=True, pin_memory=self.pin_mem,collate_fn=self.resize)
        
        self.testing_dataset = self.dataset_hcp.hcp_data(args,self.ids[self.train_vols:self.train_vols+args.test_vols],test = True)
        self.testing_data = DataLoader(dataset=self.testing_dataset, batch_size=1,pin_memory=self.pin_mem,drop_last=True,collate_fn=self.resize_test)
        
        if(args.debug):
            print("train",self.ids[:self.train_vols],len(self.ids[:self.train_vols]))
            print("test",self.ids[self.train_vols:self.train_vols+args.test_vols],len(self.ids[self.train_vols:self.train_vols+args.test_vols]))
        
        print("Loading Done")

    def rebuild(self,type,blk_size = None,train_var = False):
        if type == "train":
            self.training_dataset.preload_data(blk_size = blk_size,var = train_var)
            self.training_data = DataLoader(dataset=self.training_dataset, batch_size=1,drop_last=True, pin_memory=self.pin_mem,collate_fn=self.resize)
        if type == "test":
            self.testing_dataset.preload_data(test=True)
            self.testing_data = DataLoader(dataset=self.testing_dataset, batch_size=1,drop_last=True,pin_memory=self.pin_mem,collate_fn=self.resize_test)

    def resize(self,data):
        x,y = [],[]
        dims = len(data[0][1].shape)
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(np.concatenate([np.expand_dims(data[i][1],axis = dims),np.expand_dims(data[i][2],axis = dims),data[i][3]], axis = dims))
        lr = torch.from_numpy(np.stack(x))
        hr = torch.from_numpy(np.stack(y))
        scale = data[i][4]
        return lr,hr,scale


    def resize_test(self,data):
        x,y,z,s = [],[],[],[]
        dims = len(data[0][1].shape)
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(np.concatenate([np.expand_dims(data[i][1],axis = dims),np.expand_dims(data[i][2],axis = dims),data[i][3]], axis = dims))
            z.append(np.concatenate([np.expand_dims(data[i][5],axis = dims),np.expand_dims(data[i][6],axis = dims),data[i][7]], axis = dims))
        lr = torch.from_numpy(np.stack(x))
        hr = torch.from_numpy(np.stack(y))
        out = torch.from_numpy(np.stack(z))
        scale = data[i][4]
        return lr,hr,out,scale


