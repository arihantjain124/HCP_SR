import torch
import torch.nn as nn

class DeepDTI(nn.Module):
    def __init__(self):
        super(DeepDTI, self).__init__()
        
        # First layer: 4D Convolution with ReLU activation
        self.conv1 = nn.Conv3d(8, 190, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
        # 8 layers with 3D Convolution, Batch Normalization, and ReLU activation
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(190, 190, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(190),
                nn.ReLU()
            ) for _ in range(8)
        ])
        
        # Output layer: 4D Convolution
        self.conv_out = nn.Conv3d(190, 5, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        
        for conv_layer in self.conv_layers:
            out = conv_layer(out)
        
        out = self.conv_out(out)
        
        return out