import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'TV':
                loss_function = nn.MSELoss()
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        
        if args.load != '.': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, pred,hr,pred_tv = None,hr_tv = None):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                if l['type'] == 'L1':
                    loss = 1*l['function'](pred, hr)
                    effective_loss = l['weight'] * loss
                    losses.append(effective_loss)

                elif l['type'] == 'MSE':
                    loss = 1*l['function'](pred, hr)
                    effective_loss = l['weight'] * loss
                    losses.append(effective_loss)

                elif l['type'] == 'TV':
                    loss = 1*l['function'](pred_tv,hr_tv)
                    effective_loss = l['weight'] * loss
                    losses.append(effective_loss)
                    

        loss_sum = sum(losses)

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()


    def get_loss_module(self):
        return self.loss_module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()
