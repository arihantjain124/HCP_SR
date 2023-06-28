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

time_stamp = 0
ids = utils.get_ids()
ids_test = ids[40:41]
ids_train = ids[30:34]
torch.backends.cudnn.benchmark = True
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    tot_len = len(dataset.ids)
    per_worker = int(tot_len / float(worker_info.num_workers))
    worker_id = worker_info.id
    dataset.ids = dataset.ids[worker_id * per_worker:max(((worker_id+1) * per_worker),tot_len )]
    dataset.curr_id = -1

    
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
parser.add_argument("--preload", type=bool,
                    help="dataset_directory")

args = list(parser.parse_known_args())[0]

args.dir = "/storage"
args.batch_size = 16
args.debug = True
args.preload = True
cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

training_dataset = data.HCP_dataset.hcp_data(args,ids_train)
testing_dataset = data.HCP_dataset.hcp_data(args,ids_test)

print("Number of data images ",training_dataset.tot)

training_dataloader = torch.utils.data.DataLoader(training_dataset, num_workers=0,batch_size = args.batch_size,worker_init_fn=worker_init_fn,collate_fn=custom_collate)
testing_dataloader = torch.utils.data.DataLoader(testing_dataset, num_workers=0,batch_size = args.batch_size,worker_init_fn=worker_init_fn,collate_fn=custom_collate)
print("Dataloader Created")


def train(model,criterion,optimizer,tb,time_stamp):
    model.train()
    # pbar = tqdm(total = (len(ids_train) * 5))
    pbar = tqdm()
    for i in range(len(ids_train)):
        for j in iter(training_dataloader):
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
            if(time_stamp%10 == 0):
                tb.add_scalar("Loss", loss.item(), time_stamp/10)
                pbar.set_description("Loss_l1: {:.10f} {:.2f}".format(loss.item(),time_stamp))
            
            time_stamp+=1
            pbar.update(1)
    pbar.close()
    return time_stamp

def valid(model,scores,tb,epoch):
    model.eval()
    print("validation Start")
    # pbar = tqdm(total = (len(ids_train) * 5))
    pbar = tqdm()
    tot_psnr,tot_ssim = [],[]
    for i in range(len(ids_test)):
        for j in iter(testing_dataloader):
            gt,inp = j
            if args.cuda:
                gt = gt.to(device) 
                inp = inp.to(device)
            
            inp = torch.permute(inp, (0,4,1,2,3)).cuda()
            pred = model(inp)
            pred = torch.permute(pred, (0,2,3,4,1)).cuda()
            psnr,ssim = scores(pred,gt)
            tot_psnr.append(psnr.item())
            tot_ssim.append(ssim.item())
            pbar.update(1)
    
    pbar.close()

    tb.add_scalar("PSNR", sum(tot_psnr)/len(tot_psnr), epoch)
    tb.add_scalar("SSIM", sum(tot_ssim)/len(tot_ssim), epoch)
    
    print(f"psnr:{sum(tot_psnr)/len(tot_psnr)},ssim:{sum(tot_ssim)/len(tot_ssim)}")
    



network = torchcnn.DeepDTI_torch().cuda()
optimizer = optim.Adam(network.parameters(), lr=0.003)
criterion = torchcnn.Loss_MSE()
scores = torchcnn.PSNR()
# if __name__ == "__main__":
comment = f'runs/model = DeepDTI_torch,batch_size ={args.batch_size},block_size={args.block_size}'
print(comment)
tb = SummaryWriter(comment=comment)
for i in range(5):
    time_stamp = train(network,criterion,optimizer,tb,time_stamp)
    valid(network,scores,tb,i)
tb.close()