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

import cupy as cp
import cucim.skimage.metrics as metrics
import torch
import torch.nn as nn
import numpy as np

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


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

import random
import matplotlib.pyplot as plt

def plot_train(pred,hr,logger,epoch):
    pred = pred.cpu().detach().numpy()
    hr = hr.cpu().detach().numpy()
    samples = random.sample(range(pred.shape[0]), 6)
    fig, ax = plt.subplots(len(samples),6)
    k=0
    for i in samples:
        ax[k][0].imshow((pred[i,:,:,8,0]* 255).astype('uint8'))
        ax[k][1].imshow((pred[i,:,:,8,1]* 255).astype('uint8'))
        ax[k][2].imshow((pred[i,:,:,8,2:]* 255).astype('uint8'))
        ax[k][3].imshow((hr[i,:,:,8,0]* 255).astype('uint8'))
        ax[k][4].imshow((hr[i,:,:,8,1]* 255).astype('uint8'))
        ax[k][5].imshow((hr[i,:,:,8,2:]* 255).astype('uint8'))
        k+=1
    logger.add_figure("sample_train",fig,global_step = epoch)

    
def logger_sampling(pred,logger,epoch,hr):

    pred = pred.cpu().detach().numpy()
    size = pred.shape[:3]
    x,y,z = random.sample(range(40, 90), 3)
    fig, ax = plt.subplots(3,6)
    fa = pred[x,:,:,0]
    adc = pred[x,:,:,1]
    rgb = pred[x,:,:,2:]
    # print(fa.min(),fa.max())
    # print(adc.min(),adc.max())
    # print(rgb.min(),rgb.max())
    # print(fa.shape,adc.shape,rgb.shape)
    # print(type(fa),type(adc),type(rgb))
    ax[0][0].imshow((fa * 255).astype('uint8'))
    ax[0][1].imshow((adc * 255).astype('uint8'))
    ax[0][2].imshow((rgb * 255).astype('uint8'))

    
    fa = pred[:,y,:,0]
    adc = pred[:,y,:,1]
    rgb = pred[:,y,:,2:]
    ax[1][0].imshow((fa * 255).astype('uint8'))
    ax[1][1].imshow((adc * 255).astype('uint8'))
    ax[1][2].imshow((rgb * 255).astype('uint8'))

    
    fa = pred[:,:,z,0]
    adc = pred[:,:,z,1]
    rgb = pred[:,:,z,2:]
    ax[2][0].imshow((fa * 255).astype('uint8'))
    ax[2][1].imshow((adc * 255).astype('uint8'))
    ax[2][2].imshow((rgb * 255).astype('uint8'))
    # fig.colorbar()
    logger.add_figure("samples",fig,global_step = epoch)



def compute_psnr_ssim(hr,pred,pnts,logger=None,epoch=None):
    pred = recon(pred,pnts,vol_size=hr.shape)
    if(logger != None):
        logger_sampling(pred,logger,epoch,hr)
    hr = cp.array(hr.squeeze())
    pred = cp.array(pred.squeeze())
    return float(metrics.peak_signal_noise_ratio(hr,pred,data_range=1)),abs(float(metrics.structural_similarity(hr,pred,channel_axis =3,data_range=1)))