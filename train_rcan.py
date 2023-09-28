import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

### multi processing
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
####

import data.HCP_dataset_h5
import data.utils_metrics as utils_met
from itertools import product
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils
import data.HCP_dataset_h5 as HCP_dataset

import math
import torch.nn.functional as F


def pad(x):
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        #print(w_pad,h_pad)
        # # TODO: fix this type when PyTorch fixes theirs
        # # the documentation lies - this actually takes a list
        # # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

def unpad(x,h_pad,w_pad,h_mult,w_mult):
    return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

def resize(data):
    x,y = [],[]
    for i in range(len(data)):
        x.append(data[i][0].reshape((data[i][0].shape[0]*data[i][0].shape[1],data[i][0].shape[2],data[i][0].shape[3])))
        y.append(np.concatenate([np.expand_dims(data[i][1],axis = 3),np.expand_dims(data[i][2],axis = 3),data[i][3]], axis=3))
    return torch.from_numpy(np.stack(x)),torch.from_numpy(np.stack(y))


def resize_mask(data):
    x,y = [],[]
    mask = []
    for i in range(len(data)):
        mask.append(data[i][0][:,:,:,7])
        x.append(data[i][0].reshape((data[i][0].shape[0]*data[i][0].shape[1],data[i][0].shape[2],data[i][0].shape[3])))
        y.append(np.concatenate([np.expand_dims(data[i][1],axis = 3),np.expand_dims(data[i][2],axis = 3),data[i][3]], axis=3))
    return torch.from_numpy(np.stack(x)),torch.from_numpy(np.stack(y)),torch.from_numpy(np.stack(mask))

parser = argparse.ArgumentParser(description="RCAN")
parser.add_argument("--block_size", type=tuple, default=(64,64,64),
                    help="Block Size")
parser.add_argument("--crop_depth", type=int, default=30,
                    help="crop across z-axis")
parser.add_argument("--dir", type=str,
                    help="dataset_directory")
parser.add_argument("--batch_size", type=int,
                    help="")
parser.add_argument("--sort", type=bool,
                    help="")
parser.add_argument("--debug", type=bool,
                    help="")
parser.add_argument("--preload", type=bool,
                    help="")
args = list(parser.parse_known_args())[0]
args.preload = True
args.debug = False
args.dir = "/storage"
args.batch_size = 4
args.sort = True
args.typ = 'upsampled'
args.block_size = (64,64,64)
args.epochs = 100
print(args)


args.cuda = True
cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')


#####
ids = utils.get_ids()
ids.sort()

####
total_vols = 10
train_vols = 8

####
ids = ids[:total_vols]
dataset_hcp = HCP_dataset
dataset_hcp.load_data(args.dir,ids)
testing_dataset = dataset_hcp.hcp_data_test(args,ids[train_vols:])
testing_data_loader = DataLoader(dataset=testing_dataset, batch_size=1,pin_memory=True,collate_fn=resize_mask)
############







def save_checkpoint(modelname,psnr,ssim):
    model_folder = "checkpoint_skip_blank/"
    model_out_path = model_folder + "{}_ssim{:.2f},psnr{:.2f}.pth".format(modelname,ssim*100,psnr)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    return num_params


def train(model,l1_criterion,optimizer,epoch,training_data_loader,tb):
    model.train()
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    # print("len of dataloader ",len(training_data_loader))
    pbar = tqdm(total = len(training_data_loader))
    for iteration, (lr_tensor, hr_tensor,mask) in enumerate(training_data_loader, 1):
        pbar.update(1)
        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]
            mask = mask.to(device)  # ranges from [0, 1]
        
        optimizer.zero_grad()
        lr_tensor = torch.permute(lr_tensor,(0,3,1,2))
        pred_tensor = model(lr_tensor)
        pred_tensor = torch.permute(pred_tensor,(0,2,3,1))
        pred_tensor = pred_tensor.reshape(args.batch_size,args.block_size[0],args.block_size[1],args.block_size[2],5)

        # print(sr_tensor.shape,hr_tensor.shape)
        loss_l1 = l1_criterion(pred_tensor, hr_tensor,mask)

        loss_l1.backward()
        optimizer.step()
        
        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]
        # print(curr_lr)
        if iteration % 10 == 0:
            time_stamp = (len(training_data_loader)//10 * (epoch-1)) + iteration/10
            tb.add_scalar("Loss", loss_l1.item(), time_stamp)
            # print("===> Epoch[{}]({}/{}): Loss_l1: {:.10f}".format(epoch, iteration, len(training_data_loader),
            #                                                       loss_l1.item()))
            # tb.add_scalar("lr", scheduler.get_last_lr()[0], time_stamp)
            tb.add_scalar("lr", curr_lr, time_stamp)
            pbar.set_description("Loss_l1: {:.10f}, LR {:.5f}".format(loss_l1.item(),curr_lr))
            # print("===> Epoch[{}]({}/{}): Loss_l1: {:.10f}, LR {:.5f}".format(epoch, iteration, len(training_data_loader),
            #                                                       loss_l1.item(),curr_lr))
    pbar.close()
    
def valid(model,epoch,tb,out_size = (1, 173, 207, 173, 5)):
    model.eval()

    avg_psnr, avg_ssim = 0, 0
    for batch in testing_data_loader:
        lr_tensor, hr_tensor, mask = batch[0], batch[1], batch[2]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)
            mask = mask.to(device)  # ranges from [0, 1]
        
        lr_tensor = torch.permute(lr_tensor,(0,3,1,2))
        temp = pad(lr_tensor)
        
        with torch.no_grad():
            # print(lr_tensor.shape)
            pre = model(temp[0])

        pred = unpad(pre,temp[1][0],temp[1][1],temp[1][2],temp[1][3])
        pred = pred.reshape(out_size)
        avg_psnr += utils_met.compute_psnr(hr_tensor, pred,mask)
        avg_ssim += utils_met.compute_ssim(hr_tensor, pred,mask)
        
    tb.add_scalar("avg_psnr", avg_psnr / len(testing_data_loader), epoch)
    tb.add_scalar("avg_ssim", avg_ssim / len(testing_data_loader), epoch)
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader)))
    return avg_psnr / len(testing_data_loader),avg_ssim / len(testing_data_loader)


parameters = dict(
    models = ['CSEUnetModel'],
    lr = [0.008],
    batch_size = [32],
    block_size = [(64,64,64),(32,32,32),(64,64,8),(64,8,64)]
)

param_values = [v for v in parameters.values()]

best_psnr,best_ssim = 0,0

for run_id, (models,lr,batch_size,block_size) in enumerate(product(*param_values)):
    
    best_psnr,best_ssim = 0,0
    args.block_size = block_size
    args.batch_size = batch_size
    print("run id:", run_id + 1)
    if models == 'CSEUnetModel':
        model = CSEUnetModel(in_chans = 8,out_chans = 5,chans = 4,num_pool_layers = 2,drop_prob=0.2,reduction=4)
        # model = nn.DataParallel(model)
    
    training_dataset = dataset_hcp.hcp_data(args,ids[:train_vols])
    training_data_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True,collate_fn=resize_mask)
    print(f' model name {models} , num_params = {print_network(model)}')
    # args.lr = lr
    
    model = model.to(device)
    
    l1_criterion = utils_met.MaskedL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(training_data_loader)*5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(training_data_loader), epochs=args.epochs)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr = lr, max_lr=0.08, step_size_up  = len(training_data_loader) * 4 , mode  = 'exp_range',cycle_momentum=False)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor = 0.5)

    comment = f'model = {models} batch_size = {batch_size} lr = {lr} block_size = {block_size}'
    print(comment)
    tb = SummaryWriter(comment=comment)

    curr_psnr,curr_ssim = valid(model,-1,tb)
    for epoch in range(0, args.epochs):
        train(model,l1_criterion,optimizer,epoch,training_data_loader,tb)
        curr_psnr,curr_ssim = valid(model,epoch,tb)
        if(best_psnr<curr_psnr or best_ssim<curr_ssim):
            if(best_psnr<curr_psnr):
                best_psnr = curr_psnr
            if(best_ssim<curr_ssim):
                best_ssim = curr_ssim
            save_checkpoint(models,best_psnr,best_ssim)
tb.close()
