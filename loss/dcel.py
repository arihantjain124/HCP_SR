import torch
import torch.nn as nn
import numpy as np
import math
from torch import linalg as LA

class DCELoss(nn.Module):
    def __init__(self):
        super(DCELoss,self).__init__()


    def forward(self,pred,hr):

        batch_size = pred.shape[0]
        temp = 1
        for i in pred.shape[1:]:
            temp *=i


        S_srhr = torch.zeros((batch_size,batch_size))
        S_srsr = torch.zeros((batch_size,batch_size))
        M = torch.zeros((batch_size,batch_size))
        pred_temp = pred.reshape(batch_size,temp)
        hr_temp = hr.reshape(batch_size,temp)
        for i in range(batch_size):
            for j in range(batch_size):
                S_srhr[i][j] = torch.dot(pred_temp[i],hr_temp[j])/(LA.norm(pred_temp[i])* LA.norm(hr_temp[j]))
                S_srsr[i][j] = torch.dot(pred_temp[i],pred_temp[j])/(LA.norm(pred_temp[i])* LA.norm(pred_temp[j]))
                M[i][j] = min((-20)*torch.log10(LA.norm(hr_temp[i] - hr_temp[j])),0)
                
        Q_pos = torch.zeros((batch_size))
        Q_neg = torch.zeros((batch_size))


        n_thres = 30
        t_pos,t_neg = 0.5,0.5
        
        for i in range(batch_size):
            for j in range(batch_size):
                if( abs(M[i][j]) > abs(n_thres)):
                    Q_pos[i] +=  (( torch.exp(S_srsr[i][j]) + (2*torch.exp(S_srhr[i][j])) ) /t_pos )
                else:
                    Q_neg[i] += (( torch.exp(S_srsr[i][j]) + (2*torch.exp(S_srhr[i][j])) ) /t_neg )
                
        los = (-1/batch_size) * torch.log(Q_pos/Q_neg).sum()
        
        return los