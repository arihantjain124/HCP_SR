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
        self.flag_sca = True
        self.flag_asy = True
        self.flag_var = True

        self.training_dataset = self.dataset_hcp.hcp_data(args,self.ids[:self.train_vols])

        self.testing_dataset = self.dataset_hcp.hcp_data(args,self.ids[self.train_vols:self.train_vols+args.test_vols],test = True)    
        
        self.training_data = DataLoader(dataset=self.training_dataset, batch_size=1,shuffle = True)
        self.testing_data = DataLoader(dataset=self.testing_dataset, batch_size=1,shuffle = True)
        
        if(args.debug):
            print("train",self.ids[:self.train_vols],len(self.ids[:self.train_vols]))
            print("test",self.ids[self.train_vols:self.train_vols+args.test_vols],len(self.ids[self.train_vols:self.train_vols+args.test_vols]))
        
        print("Loading Done")

    def rebuild(self,blk_size = None,var = False):
        t = {}
        t['sca'] = float(self.training_dataset.sca) 
        t['asy'] = float(self.training_dataset.asy)
        t['var'] = float(self.training_dataset.var)

        if(np.random.randint(5) <= 1):
            
            if(t['sca'] > 1):
                self.flag_sca = False
            elif(t['sca'] < 0.1):
                self.flag_sca = True
                
            if(self.flag_sca):
                t['sca'] += 0.1
            else:
                t['sca'] -= 0.1
            
        elif(np.random.randint(5) <4):
            if(t['var'] > 7):
                self.flag_var = False
            elif(t['var'] < 2):
                self.flag_var = True

            if(self.flag_var):
                t['var'] += 2
            else:
                t['var'] -= 2
            
        else:
            if(t['asy'] > 0.3):
                self.flag_asy = False
            elif(t['asy'] < 0.1):
                self.flag_asy = True
                
            if(self.flag_asy):
                t['asy'] += 0.1
                
            else:
                t['asy'] -= 0.1
                
            
        self.training_dataset.preload_data(args = t)
        self.testing_dataset.preload_data(test=True,args = t)
        
        self.training_data = DataLoader(dataset=self.training_dataset, batch_size=1,drop_last=True,shuffle = True, pin_memory=self.pin_mem)
        self.testing_data = DataLoader(dataset=self.testing_dataset, batch_size=1,drop_last=True,shuffle = True,pin_memory=self.pin_mem)

        return t

