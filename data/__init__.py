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
        self.train_vols = int(len(self.ids) * (args.train_set))

        self.testing_dataset = self.dataset_hcp.hcp_data_test_recon(args,self.ids[self.train_vols:],debug=debug)
        self.training_dataset = self.dataset_hcp.hcp_data(args,self.ids[:self.train_vols])

        self.testing_data = DataLoader(dataset=self.testing_dataset, batch_size=1,shuffle=True,pin_memory=self.pin_mem,drop_last=True,collate_fn=self.resize_test)
        self.training_data = DataLoader(dataset=self.training_dataset, batch_size=16, shuffle=True, pin_memory=self.pin_mem, drop_last=True,collate_fn=self.resize)
        
        print("Loading Done")

    def rebuild(self,blk_size,type,stable = False):
        if type == "train":
            self.training_dataset.preload_data(stable,blk_size = blk_size)
            if(stable):
                self.training_data = DataLoader(dataset=self.training_dataset, batch_size=16, shuffle=True, drop_last=True, pin_memory=self.pin_mem,collate_fn=self.resize)
            else:
                self.training_data = DataLoader(dataset=self.training_dataset, batch_size=1, shuffle=True, drop_last=True, pin_memory=self.pin_mem,collate_fn=self.resize)
        if type == "test":
            self.testing_dataset.preload_data(blk_size = blk_size)
            self.testing_data = DataLoader(dataset=self.testing_dataset, batch_size=1,shuffle=True,drop_last=True,pin_memory=self.pin_mem,collate_fn=self.resize_test)

    def resize(self,data):
        x,y,z = [],[],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(np.concatenate([np.expand_dims(data[i][1],axis = 3),np.expand_dims(data[i][2],axis = 3),data[i][3]], axis=3))
        lr = torch.from_numpy(np.stack(x))
        pred = torch.from_numpy(np.stack(y))
        scale = data[i][4]
        return lr,pred,scale


    def resize_test(self,data):
        x,y,z,s = [],[],[],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(np.concatenate([np.expand_dims(data[i][1],axis = 3),np.expand_dims(data[i][2],axis = 3),data[i][3]], axis=3))
            z.append(data[i][4])
            s.append(data[i][5])
        lr = torch.from_numpy(np.stack(x)).squeeze()
        pred = torch.from_numpy(np.stack(y)).squeeze()
        pnt = torch.from_numpy(np.stack(z)).squeeze()
        scale = data[i][5]
        return lr,pred,pnt,scale


