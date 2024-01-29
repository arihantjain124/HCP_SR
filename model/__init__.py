import os
from importlib import import_module
import torch
import torch.nn as nn
import model.dmri_model as dmri_model

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        print('Making model... here')
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.save_models = args.save_models

        if args.model == 'dmri_rdn':
            if args.model_type == '2d':    
                self.model = dmri_model.DMRI_RDN_2d(growth=args.growth).to(self.device)
            else:
                self.model = dmri_model.DMRI_RDN_3d(growth=args.growth).to(self.device)
        if args.model == 'dmri_rcan':
            if args.model_type == '2d':    
                self.model = dmri_model.DMRI_RCAN_2d(int_chans=args.growth).to(self.device)
            else:
                self.model = dmri_model.DMRI_RCAN_3d(int_chans=args.growth).to(self.device)
            

        if args.precision == 'half': self.model.half()
        

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

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.get_model().load_state_dict(
            torch.load(
                apath,
                **kwargs
            ),
            strict=False
        )
        print('load from model_' + str(apath) + '.pt')
