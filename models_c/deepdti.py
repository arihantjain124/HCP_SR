import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

import math
from model.rdn import make_rdn
from model.rdn_2d import make_rdn as make_rdn_2d
from model.resblock import ResBlock
from model.arb_decoder import ImplicitDecoder_3d,ImplicitDecoder_2d
import numpy as np


class deepdti(nn.Module):
    def __init__(self,args):
        super().__init__()

    def forward(self, inp,size,rel_coor):

    return 

