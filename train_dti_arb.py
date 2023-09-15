import argparse, os
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import cucim
import utils
import data.HCP_dataset_h5_arb as HCP_dataset

from torch.utils.tensorboard import SummaryWriter
from itertools import product
from tqdm import tqdm

##### Model Imports ####
import torch
import torch.nn as nn
import data.utils_metrics as utils_met
import cucim.skimage.metrics as met
import torch.optim as optim
from torch.utils.data import DataLoader
from model import dmri_arb
#########################

parser = argparse.ArgumentParser(description="DTI_ARB")
parser.add_argument("--block_size", type=tuple, default=(16,16,4),
                    help="Block Size")
parser.add_argument("--test_size", type=tuple, default=(173,207,173),
                    help="test_size")
parser.add_argument("--crop_depth", type=int, default=15,
                    help="crop across z-axis")
parser.add_argument("--dir", type=str,
                    help="dataset_directory")
parser.add_argument("--batch_size", type=int,
                    help="Batch_size")
parser.add_argument("--sort", type=bool,
                    help="Sort Subject Ids")
parser.add_argument("--debug", type=bool,
                    help="Print additional input")
parser.add_argument("--preload", type=bool,
                    help="Preload data into memory")

args = list(parser.parse_known_args())[0]
args.preload = True
args.debug = False
args.dir = "/storage"
args.batch_size = 4
args.sort = True
args.cuda = True
cuda = args.cuda
device = torch.device('cuda:2' if cuda else 'cpu')
# torch.cuda.set_device(1)
args.epochs = 100
print(args)

ids = utils.get_ids()
ids.sort()


def resize(data):
    x,y = [],[]
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(np.concatenate([np.expand_dims(data[i][1],axis = 3),np.expand_dims(data[i][2],axis = 3),data[i][3]], axis=3))
    return torch.from_numpy(np.stack(x)),torch.from_numpy(np.stack(y))
####
total_vols = 2
train_vols = 1
####

ids = ids[:total_vols]
dataset_hcp = HCP_dataset
dataset_hcp.load_data(args.dir,ids)
testing_dataset = dataset_hcp.hcp_data_test(args,ids[train_vols:])
testing_data_loader = DataLoader(dataset=testing_dataset, batch_size=1,pin_memory=True,collate_fn=resize)



def save_checkpoint(modelname,psnr,ssim):
    model_folder = "model_ckp_dti/"
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
    if(epoch<5):
        model.set_scale((1,1,1))
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    
    pbar = tqdm(total = len(training_data_loader))
    
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):
        pbar.update(1)
        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]
        
        optimizer.zero_grad()
        lr_tensor = torch.permute(lr_tensor, (0,4,1,2,3))
        pred_tensor = model(lr_tensor)
        pred_tensor = torch.permute(pred_tensor, (0,2,3,4,1))

        loss_l1 = l1_criterion(pred_tensor, hr_tensor)

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

    
def valid(model,epoch,tb):
    model.eval()

    avg_psnr, avg_ssim = 0, 0
    for iteration, (lr_tensor, hr_tensor) in enumerate(testing_data_loader, 1):
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)
        
        
        t3_vol = lr_tensor.shape
        t7_vol = hr_tensor.shape
        sca = [t7_vol[i]/t3_vol[i] for i in range(1,4)]
        model.set_scale(sca)
        lr_tensor = torch.permute(lr_tensor, (0,4,1,2,3))
        
        with torch.no_grad():
            pred_tensor = model(lr_tensor)

        pred_tensor = torch.permute(pred_tensor, (0,2,3,4,1))
       
        pred_tensor = torch.squeeze(pred_tensor)
        hr_tensor = torch.squeeze(hr_tensor)
        # print(pred_tensor.shape,hr_tensor.shape)
        avg_psnr += utils_met.compute_psnr(hr_tensor, pred_tensor)
        avg_ssim += utils_met.compute_ssim(hr_tensor, pred_tensor)
        
    tb.add_scalar("avg_psnr", avg_psnr / len(testing_data_loader), epoch)
    tb.add_scalar("avg_ssim", avg_ssim / len(testing_data_loader), epoch)
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader)))
    return avg_psnr / len(testing_data_loader),avg_ssim / len(testing_data_loader)


parameters = dict(
    models = ['arb'],
    lr = [0.008],
    batch_size = [32],
    block_size = [(16,16,4),(16,16,8),(16,16,16)]
)

param_values = [v for v in parameters.values()]

best_psnr,best_ssim = 0,0

for run_id, (models,lr,batch_size,block_size) in enumerate(product(*param_values)):
    
    best_psnr,best_ssim = 0,0
    args.block_size = block_size
    args.batch_size = batch_size
    # args.scale = scale
    print("run id:", run_id + 1)
    if models == 'arb':
        model = dmri_arb.DMRI_SR()
        model.set_scale((1,1,1))
    
    training_dataset = dataset_hcp.hcp_data(args,ids[:train_vols])
    training_data_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True,collate_fn=resize)
    print(f' model name {models} , num_params = {print_network(model)}')
    # args.lr = lr
    
    model = model.to(device)
    
    l1_criterion = torch.nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(training_data_loader))
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(training_data_loader), epochs=args.epochs)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr = lr, max_lr=0.08, step_size_up  = len(training_data_loader) * 4 , mode  = 'exp_range',cycle_momentum=False)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor = 0.5)

    comment = f'model = {models} batch_size = {batch_size} lr = {lr} block_size = {block_size}'
    print(comment)
    tb = SummaryWriter(comment=comment)
    torch.cuda.empty_cache()
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
