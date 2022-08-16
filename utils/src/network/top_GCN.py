import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import math
from torch.nn import Parameter
from util import *
import sys
sys.path.append(r"/home/workspace/huichen/codes/gcn/src/representation/")
from src.representation import *


def conv3_1D(in_planes, out_planes, stride=1):
    """1D 3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1_1D(in_planes, out_planes, stride=1):
    """1D 1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv31_1D(in_planes, out_planes, stride=1):
    """"1D 31 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=31, stride=stride,
                     padding=kernel // 2, bias=False)
                     
                     
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1_1D(inplanes, planes)
        #self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3_1D(planes, planes, stride)
        #self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1_1D(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=False)
        #self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
       
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.leakyrelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leakyrelu(out)

        return out


class Multi_label_model(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
        super(Multi_label_model, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=False)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)       
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            #elif isinstance(m, nn.Linear):
             #   nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1_1D(self.inplanes, planes * block.expansion, stride),
                # nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.maxpool1(x)

        x1 = self.layer1(x)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)  #[32, 2048, 313]
     
        return x1

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Resnet_GCN(nn.Module):
    """build Resnet + GCN for the first stage of training"""
    def __init__(self, num_classes, in_channel=512):
        super(Resnet_GCN, self).__init__()
        self.feature = Multi_label_model(Bottleneck, [3, 4, 6, 3])
       
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.pooling = nn.AdaptiveAvgPool1d((1))        
        self.linear = nn.Linear(2048, 512)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=False)
        self.classi = nn.Linear(2048, num_classes)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x, label):
        batch_size = x.size(0)
        feature = self.feature(x)  # (batch_size, 2048, 313)
        feature = self.pooling(feature)  # (batch, 2048, 1)
        feature = feature.view(feature.size(0), -1)  # (batch, 2048)

        feature = self.linear(feature) #(batch, 100)
        feature = self.leakyrelu(feature)
        A = gen_A(batch_size, label.cpu())
        A = Parameter(A.float())
        adj = gen_adj_(A).cuda().detach()    
        gcn_out = self.gc1(feature, adj)
        gcn_out = self.leakyrelu(gcn_out)
        gcn_out = self.gc2(gcn_out, adj)
        gcn_out = self.leakyrelu(gcn_out)
        out = self.classi(gcn_out)
        out = self.sigmoid(out)
        return out       

class Resnet_BCNN_Bi(nn.Module):
    """build Resnet + BCNN + BiRNN Architecture for the second stage of training"""
    def __init__(self, representation, num_classes):
        super(Resnet_BCNN_Bi, self).__init__()
        self.feature = Multi_label_model(Bottleneck, [3, 4, 6, 3])
        self.BiRNN = nn.LSTM(512 * 4, 50, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2048*100, num_classes)
        self.sigmoid_2 = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=False)
        if representation is not None:
            representation_method = representation['function']
            representation.pop('function')
            representation_args = representation
            self.representation = representation_method(**representation_args)
            
    def forward(self, x):
        x = self.feature(x)
        x = self.representation(x) # [32, 2048, 2048]
        import pdb

        x, _ = self.BiRNN(x)  # [32, 2048, 100]
        x = self.leakyrelu(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.sigmoid_2(x)
        return x


def first_train(num_classes):
    """
    First stage, consists of Resnet and GCN;
    """
    model = Resnet_GCN(num_classes)
    return model
    
def second_train(representation, num_classes):
    """
    Second stage, consists of Resnet and BCNN and BiRNN;
    returns the model pre-trained in the first stage
    """
    model = Resnet_BCNN_Bi(representation, num_classes)

    return model
    
if __name__ == '__main__':
    representation = {'function':BCNN,
                      'is_vec':True,
                      'input_dim':2048}
    x = torch.rand([64, 1, 10000])
    inp = torch.randn([64, 10, 300])
    model = GCN_Resnet_BCNN(representation, num_classes=10, freezed_layer=0, pretrained=False)
    y = model(x, inp)
    import pdb
    pdb.set_trace()
    print(y)

