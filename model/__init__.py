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

        if args.type == '2d':
            self.model = dmri_model.DMRI_arb_2d(args).to(self.device)
        else:
            self.model = dmri_model.DMRI_arb(args).to(self.device)
        if args.precision == 'half': self.model.half()
        

    def forward(self, x ,sca,rel_coor = None):
        # print(x.shape,sca)
        return self.model(x,sca,rel_coor)

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
