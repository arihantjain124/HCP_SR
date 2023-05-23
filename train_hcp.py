import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np




parser = argparse.ArgumentParser(description="IMDN")
parser.add_argument("--block_size", type=int, default=64,
                    help="Block Size")
parser.add_argument("--crop_depth", type=int, default=30,
                    help="crop across z-axis")
parser.add_argument("--dir", type=str,
                    help="dataset_directory")
parser.add_argument("--batch_size", type=int,
                    help="dataset_directory")

args = list(parser.parse_known_args())[0]

args.dir = "/storage/users/arihant"
args.batch_size = 4

import data.HCP_dataset

import utils
ids = utils.get_ids()


training_dataset = data.HCP_dataset.hcp_data(args,ids)

print("Number of data images ",training_dataset.tot)



def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    tot_len = len(dataset.ids)
    per_worker = int(tot_len / float(worker_info.num_workers))
    worker_id = worker_info.id
    dataset.ids = dataset.ids[worker_id * per_worker:max(((worker_id+1) * per_worker),tot_len )]
    dataset.curr_indx_blk = 0

    
def custom_collate(data):
    data = np.stack([data]).squeeze()
    data_gt = data[...,:8]
    data_pred = data[...,8:]
    return data_gt,data_pred

training_dataloader = torch.utils.data.DataLoader(training_dataset, num_workers=0,batch_size = 60,worker_init_fn=worker_init_fn,collate_fn=custom_collate)
print("Dataloader Created")

def train():
    for i in range(n):
        data = iter(training_dataloader)
        for j in data:
            gt,pred = j
            pass
        pass


def main():
    print("Training Starting")
    

if __name__ == "__main__":
    main()