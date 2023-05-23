from torch.nn import Conv3d
import torch.nn as nn
class DeepDTI_torch(nn.Module):
    def __init__(self):
        super(DeepDTI_torch, self).__init__()

        self.model_in = nn.Sequential(
            nn.Conv3d(7, 128, 3,padding="same"),      
            nn.ReLU()
        )
        self.model_mid = nn.Sequential(
                    nn.Conv3d(128, 128, 3,padding="same"),     
                    nn.BatchNorm3d(128),    
                    nn.ReLU()
        )
        self.model_out = nn.Sequential(
                    nn.Conv3d(128, 7, 3,padding="same"),  
                    nn.ReLU()
        )
        

    def forward(self,x):
        # Set 1
        output = self.model_in(x)
        for i in range(8):
            output = self.model_mid(output)
        output = self.model_out(output)
        return output


        
    
