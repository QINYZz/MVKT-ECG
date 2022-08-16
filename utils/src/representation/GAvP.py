import torch
import torch.nn as nn


class GAvP(nn.Module):
     """Global Average pooling
        Widely used in ResNet, Inception, DenseNet, etc.
     """
     def __init__(self, input_dim=2048):
         super(GAvP, self).__init__()
         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
         self.output_dim = input_dim

     def forward(self, x):
         #import pdb; pdb.set_trace()
         x = x.view(x.size(0), x.size(1), x.size(2), 1)
         x = self.avgpool(x)
         #x = x.view(x.size(0), -1)
         return x
