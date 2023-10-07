import os
from importlib import import_module
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.save_models = args.save_models

        if args.model == 'dmri_arb':
            module = import_module('model.' + args.model.lower())
            self.model = module.DMRI_SR().to(self.device)
            self.model.set_scale((1,1,1))

        if args.precision == 'half': self.model.half()
        
        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )

    def forward(self, x ,sca):
        self.model.set_scale(sca)
        return self.model(x)

    def get_model(self):
        return self.model

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )

        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == 0:
            if pre_train!='None':
                pretrained_dict = torch.load(pre_train)
                if 'model' in pretrained_dict.keys():
                    pretrained_dict =pretrained_dict['model']['sd'] 

                model_dict = self.get_model().state_dict()
                
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.get_model().load_state_dict(model_dict)
                print('load from pre-trained model')

        elif resume > 0:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )
            print('load from model_' + str(resume) + '.pt')
