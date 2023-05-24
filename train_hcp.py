import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import data.HCP_dataset
from torch.utils.tensorboard import SummaryWriter
import torchcnn
import utils

from tqdm import tqdm


ids = utils.get_ids()
ids = ids[40:50]
torch.backends.cudnn.benchmark = True
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
    data = torch.from_numpy(data).float()
    data_gt = data[...,:8]
    data_pred = data[...,8:]
    return data_gt,data_pred


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description="HCP")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--block_size", type=int, default=32,
                    help="Block Size")
parser.add_argument("--crop_depth", type=int, default=30,
                    help="crop across z-axis")
parser.add_argument("--dir", type=str,
                    help="dataset_directory")
parser.add_argument("--batch_size", type=int,
                    help="dataset_directory")
parser.add_argument("--sort", type=bool,
                    help="dataset_directory")
parser.add_argument("--debug", type=bool,
                    help="dataset_directory")

args = list(parser.parse_known_args())[0]

args.dir = "/storage"
args.batch_size = 8
args.debug = False
cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

training_dataset = data.HCP_dataset.hcp_data(args,ids)

print("Number of data images ",training_dataset.tot)

training_dataloader = torch.utils.data.DataLoader(training_dataset, num_workers=0,batch_size = args.batch_size,worker_init_fn=worker_init_fn,collate_fn=custom_collate)
print("Dataloader Created")


def train(model,criterion,optimizer,tb):
    model.train()
    pbar = tqdm(total = (len(ids) * 600))
    k=0
    for i in range(len(ids)):
        data = iter(training_dataloader)
        for j in data:
            pbar.update(1)
            k+=1
            gt,inp = j
            if args.cuda:
                gt = gt.to(device) 
                inp = inp.to(device)

            optimizer.zero_grad()
            inp = torch.permute(inp, (0,4,1,2,3)).cuda()
            pred = model(inp)
            pred = torch.permute(pred, (0,2,3,4,1)).cuda()
            loss = criterion(pred,gt)
            loss.backward()
            optimizer.step()
            if(k%10 == 0):
                time_stamp = k/10
                tb.add_scalar("Loss", loss.item(), time_stamp)
                pbar.set_description("Loss_l1: {:.10f}".format(loss.item()))
    pbar.close()

def main():
    print("Training Starting")
    

if __name__ == "__main__":
    comment = f'runs/model = DeepDTI_torch,batch_size ={args.batch_size},block_size={args.block_size}'
    print(comment)
    tb = SummaryWriter(comment=comment)
    network = torchcnn.DeepDTI_torch().cuda()
    optimizer = optim.Adam(network.parameters(), lr=0.0005)
    criterion = torchcnn.Loss_MSE()
    for i in range(1):
        train(network,criterion,optimizer,tb)
    main()
    tb.close()