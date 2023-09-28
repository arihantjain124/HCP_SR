import argparse, os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import cucim
import utils
import data.HCP_dataset_h5_arb as HCP_dataset
from tensorboardX import SummaryWriter
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
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled = False
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "heuristic"

print("PyTorch Version {}" .format(torch.__version__))
print("Cuda Version {}" .format(torch.version.cuda))
print("CUDNN Version {}" .format(torch.backends.cudnn.version()))

parser = argparse.ArgumentParser(description="DTI_ARB")
parser.add_argument("--block_size", type=tuple, default=(16,16,16),
                    help="Block Size")
parser.add_argument("--test_block_size", type=tuple, default=(16,16,16),
                    help="Block Size")
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
parser.add_argument("--ret_points", type=bool, default=False,
                    help="return box point of crops")
parser.add_argument("--thres", type=float, default=0.6,
                    help="threshold for blk emptiness")
parser.add_argument("--offset", type=int, default=20,
                    help="epoch with scale (1,1,1)")
parser.add_argument("--gaps", type=int, default=20,
                    help="number of epochs of gap between each scale change")

args = list(parser.parse_known_args())[0]
args.preload = True
args.debug = False
args.dir = "/storage"
args.batch_size = 16
args.sort = True
args.cuda = True
cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')
args.scale = (1,1,1)
# torch.cuda.set_device(7)
args.epochs = 400
args.gaps = 20
args.offset = 10
print(args)



def random_scale(seed = 0):
    np.random.seed(seed)
    sections = [0]
    sections.extend([i+args.offset for i in range(0,args.epochs-10,args.gaps)])
    scales = {i:np.around(np.random.uniform(1,2,3),decimals=2) for i in sections}
    scales[0] = (1,1,1)
    scales_txt = {j: ','.join([str(x) for x in scales[j]])  for j in scales.keys()}
    return sections,scales,scales_txt

def resize(data):
    x,y = [],[]
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(np.concatenate([np.expand_dims(data[i][1],axis = 3),np.expand_dims(data[i][2],axis = 3),data[i][3]], axis=3))
        
    lr = torch.from_numpy(np.stack(x))
    pred = torch.from_numpy(np.stack(y))
    return lr,pred


def resize_test(data):
    x,y,z,mask = [],[],[],[]
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(np.concatenate([np.expand_dims(data[i][1],axis = 3),np.expand_dims(data[i][2],axis = 3),data[i][3]], axis=3))
        z.append(data[i][4])
        mask.append(data[i][5])
    lr = torch.from_numpy(np.stack(x)).squeeze()
    pred = torch.from_numpy(np.stack(y)).squeeze()
    pnt = torch.from_numpy(np.stack(z)).squeeze()
    mask = torch.from_numpy(np.stack(mask)).squeeze()
    return lr,pred,pnt,mask

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


def train(model,l1_criterion,optimizer,epoch,training_data_loader,tb,train_blk_size):

    model.train()

    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    
    pbar = tqdm(total = len(training_data_loader))
    
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):
        pbar.update(1)
        if args.cuda:
            lr_tensor = lr_tensor.to(device).float()  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device).float()  # ranges from [0, 1]
        
        
        t3_vol = lr_tensor.shape
        t7_vol = hr_tensor.shape
        sca = [t7_vol[i]/t3_vol[i] for i in range(1,4)]
        model.set_scale(sca)
        optimizer.zero_grad()
        lr_tensor = torch.permute(lr_tensor, (0,4,1,2,3))
        lr_tensor = torch.nn.functional.interpolate(lr_tensor,size = torch.Size(train_blk_size))
        pred_tensor = model(lr_tensor)
        pred_tensor = torch.permute(pred_tensor, (0,2,3,4,1)).float()

        loss_l1 = l1_criterion(pred_tensor, hr_tensor)

        loss_l1.backward()
        optimizer.step()
        
        # curr_lr = scheduler._last_lr[0]
        
        if(epoch == 0):
            curr_lr = optimizer.defaults['lr']
        else:
            curr_lr = scheduler._last_lr[0]
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
    
    scheduler.step(loss_l1)

    
def valid(model,epoch,tb,testing_data_loader,sca):
    # model.eval()
    avg_psnr, avg_ssim = 0, 0

    print('val =', epoch)
    pbar = tqdm(total = len(testing_data_loader))
    for iteration, (lr_tensor, hr_tensor,pnts,mask) in enumerate(testing_data_loader, 1):
        pbar.update(1)
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)
        # print(lr_tensor.element_size() * lr_tensor.nelement())
        model.set_scale(sca)
        lr_tensor = torch.permute(lr_tensor, (0,4,1,2,3))
        pred = []
        with torch.no_grad():
            pred_tensor = model(lr_tensor)
        # pred_tensor = np.stack(pred, axis=0)

        pred_tensor = torch.permute(pred_tensor, (0,2,3,4,1))

        print(pred_tensor.shape,hr_tensor.shape)
        c_psnr,c_ssim =  utils_met.compute_psnr_ssim(hr_tensor, pred_tensor,pnts,mask)
        avg_psnr += c_psnr
        avg_ssim += c_ssim
    tb.add_scalar("avg_psnr", avg_psnr / len(testing_data_loader), epoch)
    tb.add_scalar("avg_ssim", avg_ssim / len(testing_data_loader), epoch)
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader)))
    return avg_psnr / len(testing_data_loader),avg_ssim / len(testing_data_loader)


parameters = dict(
    models = ['arb'],
    lr = [0.02,0.05,0.08],
    batch_size = [16,32],
    block_size = [(16,16,16),(32,32,16)],
    test_blk_size = [(16,16,16),(64,64,16),(32,32,32)]
)


ids = utils.get_ids()
ids.sort()

total_vols = 10

ids = ids[:total_vols]
dataset_hcp = HCP_dataset
ids = dataset_hcp.load_data(args.dir,ids)
print("total vols:",len(ids))

####
train_vols = int(len(ids) * 0.70)
####
print(f'train vols:{len(ids[:train_vols])} test_vols:{len(ids[train_vols:])}')

testing_dataset = dataset_hcp.hcp_data_test_recon(args,ids[train_vols:],test = True)
training_dataset = dataset_hcp.hcp_data(args,ids[:train_vols])
print(len(testing_dataset),len(training_dataset))

param_values = [v for v in parameters.values()]

best_psnr,best_ssim = 0,0
opti = "adam"
for run_id, (models,lr,batch_size,block_size,test_blk_size) in enumerate(product(*param_values)):
    
    sections,scales,scales_text = random_scale(0)

    best_psnr,best_ssim = 0,0
    args.block_size = block_size
    args.batch_size = batch_size
    args.test_block_size = test_blk_size
    # args.scale = scale
    print("run id:", run_id + 1)
    if models == 'arb':
        model = dmri_arb.DMRI_SR()
        model.set_scale((1,1,1))
        
    print(f' model name {models} , num_params = {print_network(model)}')
    
    model = model.to('cuda')
    
    l1_criterion = torch.nn.MSELoss()
    if(opti == "adam"):
        optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr = lr, max_lr=0.08, step_size_up  = len(training_data_loader) * 4 , mode  = 'exp_range',cycle_momentum=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor = 0.7,patience=10,min_lr = 0.0005)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.05, steps_per_epoch=200, epochs=args.epochs)
    comment = f'batch_size = {batch_size} lr = {lr} block_size = {block_size} test_blk_size = {args.test_block_size}'
    print(comment)
    # tb = SummaryWriter(comment=comment)
    torch.cuda.empty_cache()
    next_change = 0
    args.scale = scales[next_change]

    logdir = f"runs/{models}/train/{comment}"
    tb = SummaryWriter(logdir)
    
    #####
    testing_dataset.set_scale(scale = args.scale,blk_size = test_blk_size)
    testing_data_loader = DataLoader(dataset=testing_dataset, batch_size=1,collate_fn=resize_test)
    ### DATALOADER RESIZING

    #### Model init validation
    curr_psnr,curr_ssim = valid(model,-1,tb,testing_data_loader,test_blk_size)

    #### Training Begins
    for epoch in range(0, args.epochs):

        if(next_change == epoch):
            
            args.scale = scales[next_change]
            tb.add_text("scale_change", scales_text[next_change], epoch)
            next_change = sections.pop(0)
            
            testing_dataset.set_scale(scale = args.scale,blk_size = test_blk_size)
            testing_data_loader = DataLoader(dataset=testing_dataset, batch_size=1,pin_memory=True,collate_fn=resize_test)

            training_dataset.set_scale(scale = args.scale,blk_size = block_size)
            training_data_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True,collate_fn=resize)
            print(len(training_data_loader))
            
        train(model,l1_criterion,optimizer,epoch,training_data_loader,tb,train_blk_size = block_size)
        
        curr_psnr,curr_ssim = valid(model,epoch,tb,testing_data_loader,test_blk_size)

        if(best_psnr<curr_psnr or best_ssim<curr_ssim):
            if(best_psnr<curr_psnr):
                best_psnr = curr_psnr
            if(best_ssim<curr_ssim):
                best_ssim = curr_ssim
            save_checkpoint(models,best_psnr,best_ssim)
    
    parms = {"batch_size":batch_size,"lr":lr,"train_blk_size":','.join(map(str,block_size)),"test_blk_size":','.join(map(str,args.test_block_size)),"optimizer":opti}
    metrics = {"ssim":best_ssim,"psnr":best_psnr}
    tb.add_hparams(parms,metrics)
