import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import data.HCP_dataset_h5
import utils
from itertools import product
from deep_cascade_caunet.models import CSEUnetModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import data.HCP_dataset_h5_test


parser = argparse.ArgumentParser(description="IMDN")
parser.add_argument("--block_size", type=tuple, default=(64,64,64),
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
args.preload = True
args.debug = False
args.dir = "/storage"
args.batch_size = 4
args.sort = True
args.typ = 'upsampled'
args.block_size = (64,64,64)
print(args)


cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

ids = utils.get_ids()
ids.sort()
ids = ids[:2]
training_dataset = data.HCP_dataset_h5.hcp_data(args,ids)
print("dataset Loaded",len(training_dataset))

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

training_dataset = data.HCP_dataset_h5.hcp_data(args,ids)

testing_dataset = data.HCP_dataset_h5_test.hcp_data(args,ids)
testing_data_loader = DataLoader(dataset=testing_dataset, batch_size=1,pin_memory=True,collate_fn=resize)
# training_data_loader = DataLoader(dataset=training_dataset, batch_size=40, shuffle=True, pin_memory=True, drop_last=True,collate_fn=resize)


parameters = dict(
    models = ['CSEUnetModel'],
    lr = [0.0005],
    batch_size = [32],
    block_size = [(64,64,64),(32,32,32),(64,64,8),(64,8,64)]
)

param_values = [v for v in parameters.values()]
# print(param_values)

# for lr,batch_size, shuffle in product(*param_values):
#     print(lr, batch_size, shuffle)

def save_checkpoint(modelname,psnr,ssim,typ,data):
    model_folder = "checkpoint_x{}/".format(args.scale)
    model_out_path = model_folder + "{}_ssim{:.2f},psnr{:.2f}_{}_{}.pth".format(modelname,ssim*100,psnr,typ,data)
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


def train(model,l1_criterion,optimizer,epoch,tb):
    model.train()
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    pbar = tqdm(total = len(training_data_loader))

    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):
        pbar.update(1)
        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]
        
        optimizer.zero_grad()

        lr_tensor = torch.permute(lr_tensor,(0,3,1,2))
        pred_tensor = model(lr_tensor)
        pred_tensor = torch.permute(pred_tensor,(0,2,3,1))
        pred_tensor = pred_tensor.reshape(args.batch_size,args.block_size[0],args.block_size[0],args.block_size[0],5)

        # print(sr_tensor.shape,hr_tensor.shape)
        loss_l1 = l1_criterion(pred_tensor, hr_tensor)

        loss_l1.backward()
        optimizer.step()
        
        scheduler.step()
        curr_lr = scheduler.get_last_lr()
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
    
# def valid(model,epoch,tb):
#     model.eval()

#     avg_psnr, avg_ssim = 0, 0
#     for batch in testing_data_loader:
#         lr_tensor, hr_tensor = batch[0], batch[1]
#         if args.cuda:
#             lr_tensor = lr_tensor.to(device)
#             hr_tensor = hr_tensor.to(device)

#         temp = pad(lr_tensor[0])
        
#         with torch.no_grad():
#             # print(lr_tensor.shape)
#             pre = model(temp)

#         pred = unpad(pre,temp[1][0],temp[1][1],temp[1][2],temp[1][3])

#         avg_psnr += utils.compute_psnr(im_pre, im_label)
#         avg_ssim += utils.compute_ssim(im_pre, im_label)
        
#     tb.add_scalar("avg_psnr", avg_psnr / len(testing_data_loader), epoch)
#     tb.add_scalar("avg_ssim", avg_ssim / len(testing_data_loader), epoch)
#     print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader)))
#     return avg_psnr / len(testing_data_loader),avg_ssim / len(testing_data_loader)



best_psnr,best_ssim = 0,0

for run_id, (models,lr,batch_size,block_size) in enumerate(product(*param_values)):
    
    best_psnr,best_ssim = 0,0
    print("run id:", run_id + 1)
    if models == 'CSEUnetModel':
        model = CSEUnetModel(in_chans = 8,out_chans = 5,chans = 4,num_pool_layers = 2,drop_prob=0.2,reduction=4)
        training_dataset.rebuild(args)
        training_data_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True,collate_fn=resize)
    print(f' model name {models} , num_params = {print_network(model)}')
    # args.lr = lr
    args.block_size = block_size
    args.batch_size = batch_size
    
    model = model.to(device)
    l1_criterion = nn.L1Loss().to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(training_data_loader), epochs=args.nEpochs)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')

    comment = f'model = {models} batch_size = {batch_size} lr = {lr} patch_size = {patch_size} typ = {typ} data = {data}'
    print(comment)
    tb = SummaryWriter(comment=comment)

    curr_psnr,curr_ssim = valid(model,0,tb)
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        train(model,l1_criterion,optimizer,epoch,tb)
        curr_psnr,curr_ssim = valid(model,epoch,tb)
        if(best_psnr<curr_psnr or best_ssim<curr_ssim):
            best_psnr,best_ssim = curr_psnr,curr_ssim
            save_checkpoint(models,best_psnr,best_ssim,typ,data)
tb.close()
