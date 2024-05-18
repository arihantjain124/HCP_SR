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
        self.ids = self.dataset_hcp.load_data(args.dir,ids)
        self.train_vols = args.no_vols
        self.test_vols = args.test_vols
        self.flag_range = True
        self.flag_asy = True

        self.training_dataset = self.dataset_hcp.hcp_data(args,self.ids[:self.train_vols],start_var = args.start_var,batch_size=args.batch_size)
        self.testing_dataset = self.dataset_hcp.hcp_data(args,self.ids[self.train_vols:self.train_vols+args.test_vols],test = True,batch_size=args.test_batch_size)    
        
        self.training_data = DataLoader(dataset=self.training_dataset, batch_size=1,shuffle = True)
        self.testing_data = DataLoader(dataset=self.testing_dataset, batch_size=1,shuffle = True)
        
        if(args.debug):
            print("train",self.ids[:self.train_vols],len(self.ids[:self.train_vols]))
            print("test",self.ids[self.train_vols:self.train_vols+args.test_vols],len(self.ids[self.train_vols:self.train_vols+args.test_vols]))
        
        print("Loading Done")

    def rebuild(self,blk_size = None,var = False):
        t = {}
        t['range'] = self.training_dataset.range 
        t['asy'] = self.training_dataset.asy

        if(np.random.randint(2) == 0):
            if(t['range'] == 2):
                self.flag_range = False
            elif(t['range'] == 1):
                self.flag_range = True
                
            if(self.flag_range):
                self.training_dataset.scale_range(t['range'] + 0.1)
                self.testing_dataset.scale_range(t['range'] + 0.1)
            else:
                self.training_dataset.scale_range(t['range'] - 0.1)
                self.testing_dataset.scale_range(t['range'] - 0.1)
            
        else:
            if(t['asy'] == 0.8):
                self.flag_asy = False
            elif(t['asy'] == 0.1):
                self.flag_asy = True
                
            if(self.flag_asy):
                self.training_dataset.set_asy(t['asy'] - 0.1)
                self.testing_dataset.set_asy(t['asy'] - 0.1)
            else:
                self.training_dataset.set_asy(t['asy'] + 0.1)
                self.testing_dataset.set_asy(t['asy'] + 0.1)
            
        self.training_dataset.preload_data(blk_size = blk_size,var = var)
        self.testing_dataset.preload_data(test=True,var = var)
        
        self.training_data = DataLoader(dataset=self.training_dataset, batch_size=1,drop_last=True,shuffle = True, pin_memory=self.pin_mem)
        self.testing_data = DataLoader(dataset=self.testing_dataset, batch_size=1,drop_last=True,shuffle = True,pin_memory=self.pin_mem)

        return t

