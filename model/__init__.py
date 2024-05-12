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
        self.tensor_val = args.tensor_val
        self.attention = args.attention

        if args.model == 'dmri_rdn':
            if args.model_type == '2d':    
                self.model = dmri_model.DMRI_RDN_2d(growth=args.growth,tv = self.tensor_val).to(self.device)
            else:
                self.model = dmri_model.DMRI_RDN_3d(growth=args.growth,tv = self.tensor_val,attn = self.attention).to(self.device)
        elif args.model == 'dmri_arb':
            self.model = dmri_model.DMRI_arb(int_chans=args.growth,encoder_type=args.encoder,drop_prob = args.drop_prob,tv = self.tensor_val,attn = self.attention).to(self.device)
        else:
            print("check model name")
        if args.precision == 'half': self.model.half()
        

    def forward(self, x ,sca):
        # print(x.shape,sca)
        return self.model(x,sca)

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
