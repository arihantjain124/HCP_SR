import os
import math
import time
import datetime
from functools import reduce
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
import cv2
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from math import log10
import random
import matplotlib.pyplot as plt


import skimage.metrics as metrics
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_laplace

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = './experiment/' + args.save
        else:
            self.dir = './experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            misc.imsave('{}{}.png'.format(filename, p), ndarr)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def save_fig(x, y, pred, fig_name, srresult):
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray)
        ax[0].set_title('LR', fontsize=30)
       
        ax[1].imshow(pred, cmap=plt.cm.gray)
        ax[1].set_title('SR', fontsize=30)
        ax[1].set_xlabel("PSNR:{:.4f}\nSSIM:{:.4f}\nMSE:{:.4f}".format(srresult[0],srresult[1],srresult[2]),fontsize=20)

        ax[2].imshow(y, cmap=plt.cm.gray)
        ax[2].set_title('HR', fontsize=30)
        f.savefig(fig_name)
        plt.close()

def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma,
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
    elif args.decay_type.find("cycle") >= 0:
        scheduler = lrs.OneCycleLR(
            my_optimizer,
            max_lr = args.max_lr,
            steps_per_epoch  = steps_per_epoch,
            epochs = args.epochs
        )
    # scheduler.step(args.start_epoch - 1)

    return scheduler


def recon(x,lr,vol_size):
    num_blk = x.shape[0]
    vol = torch.empty(size=vol_size)
    for i in range(num_blk):
        vol[lr[i][0]:lr[i][1]+1,lr[i][2]:lr[i][3]+1,lr[i][4]:lr[i][5]+1,...] = x[i,...]
    return vol
        
def compute_ssim(hr,pred):
    hr = cp.array(hr.squeeze())
    pred = cp.array(pred.squeeze())
    return abs(float(metrics.structural_similarity(hr,pred,channel_axis =3,data_range=1,win_size=3)))

def compute_psnr(hr,pred):
    hr = cp.array(hr.squeeze())
    pred = cp.array(pred.squeeze())
    return float(metrics.peak_signal_noise_ratio(hr,pred,data_range=1))



def align(lr,hr,pred):
    pos = lr.shape.index(min(lr.shape[:3]))
    if(pos == 1):
        lr = np.transpose(lr,(0,2,1,3))
        hr = np.transpose(hr,(0,2,1,3))
        pred = np.transpose(pred,(0,2,1,3))
    elif(pos == 0):
        lr = np.transpose(lr,(1,2,0,3))
        hr = np.transpose(hr,(1,2,0,3))
        pred = np.transpose(pred,(1,2,0,3))
    return lr,hr,pred


def plot_train_pred(lr,hr,pred,logger,iter,epoch):
    
    
    lr = np.clip(lr.cpu().detach().numpy(),0,1)[0,...]
    pred = np.clip(pred.cpu().detach().numpy(),0,1)[0,...]
    hr = np.clip(hr.cpu().detach().numpy(),0,1)[0,...]
    
    
    fig, ax = plt.subplots(2,7)
    for i in range(2):
        for j in range(7):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            
    lr,hr,pred = align(lr,hr,pred)
    
    for i in range(7):    
        if(len(lr.shape) == 4):
            ax[0][i].imshow(lr[:,:,0,i])
        else:
            ax[0][i].imshow(lr[:,:,i])
            
    ax[1][0].set_title("pred_adc")
    ax[1][1].set_title("pred_fa")
    ax[1][2].set_title("pred_rgb")
    ax[1][3].set_title("hr_adc")
    ax[1][4].set_title("hr_fa")
    ax[1][5].set_title("hr_rgb")
    
    if(len(lr.shape) == 4):
        ax[1][0].imshow(pred[:,:,0,0])
        ax[1][1].imshow(pred[:,:,0,1])
        ax[1][2].imshow(pred[:,:,0,2:])
        ax[1][3].imshow(hr[:,:,0,0])
        ax[1][4].imshow(hr[:,:,0,1])
        ax[1][5].imshow(hr[:,:,0,2:])
    else:
        ax[1][0].imshow(pred[:,:,0])
        ax[1][1].imshow(pred[:,:,1])
        ax[1][2].imshow(pred[:,:,2:])
        ax[1][3].imshow(hr[:,:,0])
        ax[1][4].imshow(hr[:,:,1])
        ax[1][5].imshow(hr[:,:,2:])
    
    
    logger.add_figure("Training",fig,global_step = (epoch*10000)+iter)

    
def logger_sampling(hr,pred,lr,scale,logger,iter,epoch,hfen):
    # print(lr.shape,hr.shape,pred.shape)
    # print(type(pred.get()),pred.shape,type(hr),hr.shape)
    lr = np.clip(lr,0,1)[0,...]
    pred = np.clip(pred,0,1)[0,...]
    hr = np.clip(hr,0,1)[0,...]
    
    fig, ax = plt.subplots(3,4)
    # print(lr.shape,hr.shape)
    
    
    fig.suptitle(f'scale: {scale},blk_size: {lr.shape[:3]},HFEN: {hfen}', fontsize=10)
    

    for i in range(3):
        for j in range(4):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    
    ax[0][0].set_title("LR")
    ax[0][1].set_title("HR")
    ax[0][2].set_title("PRED")
    ax[0][3].set_title("HR - PRED")
    
    lr,hr,pred = align(lr,hr,pred)
    
    if(len(lr.shape) == 4):
        ax[0][0].imshow(lr[:,:,0,0])
        ax[0][1].imshow(hr[:,:,0,0])
        ax[0][2].imshow(pred[:,:,0,0])
        ax[0][3].imshow((hr-pred)[:,:,0,0])


        ax[1][0].imshow(lr[:,:,0,1])
        ax[1][1].imshow(hr[:,:,0,1])
        ax[1][2].imshow(pred[:,:,0,1])
        ax[1][3].imshow((hr-pred)[:,:,0,1])
        
        ax[2][0].imshow(lr[:,:,0,2:])
        ax[2][1].imshow(hr[:,:,0,2:])
        ax[2][2].imshow(pred[:,:,0,2:])
        diff = np.clip((hr-pred),0,1)
        ax[2][3].imshow(diff[:,:,0,2:])
    else:
        
        ax[0][0].imshow(lr[:,:,0])
        ax[0][1].imshow(hr[:,:,0])
        ax[0][2].imshow(pred[:,:,0])
        ax[0][3].imshow((hr-pred)[:,:,0])


        ax[1][0].imshow(lr[:,:,1])
        ax[1][1].imshow(hr[:,:,1])
        ax[1][2].imshow(pred[:,:,1])
        ax[1][3].imshow((hr-pred)[:,:,1])
        
        ax[2][0].imshow(lr[:,:,2:])
        ax[2][1].imshow(hr[:,:,2:])
        ax[2][2].imshow(pred[:,:,2:])
        diff = np.clip((hr-pred),0,1)
        ax[2][3].imshow(diff[:,:,2:])
    # fig.colorbar()
    logger.add_figure("Testing",fig,global_step = (epoch*10000)+iter)



def hfen_metric(reference, input_volume):
    # Apply Laplacian of Gaussian (LoG) filter to reference and input volumes
    log_reference = gaussian_laplace(reference, sigma=3)
    log_input = gaussian_laplace(input_volume, sigma=3)

    # Calculate the L2 norm of the difference between filtered volumes
    diff = log_reference - log_input
    l2_norm = np.linalg.norm(diff)

    # Normalize by the norm of the LoG-filtered reference
    norm_log_reference = np.linalg.norm(log_reference)
    hfen_value = l2_norm / norm_log_reference

    return hfen_value


def compute_scores(hr,pred,out,scale,logger=None,iter=None,mask=False,epoch=None):
    hr = hr.cpu().detach().numpy().squeeze()
    pred = pred.cpu().detach().numpy().squeeze()
    out = out.cpu().detach().numpy().squeeze()
    if(mask):   
        mask = (hr>0)
        # print(mask.device,hr.device,pred.device)
        hr = hr.squeeze()*mask
        pred = pred.squeeze()*mask
        mask = (out>0)
        out = out.squeeze()*mask
    else:
        hr = hr.squeeze()
        pred = pred.squeeze()
        out = out.squeeze()
        
    psnr = float(metrics.peak_signal_noise_ratio(hr,pred,data_range=1))
    hfen = hfen_metric(hr,pred)
    
    if(logger != None):
        logger_sampling(hr,pred,out,scale,logger,iter,epoch,hfen)
    return psnr,hfen