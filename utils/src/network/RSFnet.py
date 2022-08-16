import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import math
from torch.nn import Parameter
from util import *
from .ggcn2 import *
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
    def __init__(self, block, layers, num_classes=9, zero_init_residual=False):
        super(Multi_label_model, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3,
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
                nn.init.normal_(m.weight, 0, 1)
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
                #elif isinstance(m, BasicBlock):
                 #   nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1_1D(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
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
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)  #[32, 2048, 313]
     
        return x4, x2

    
##########################  Element_Wise_Layer  #######################

class Element_Wise_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Element_Wise_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        for i in range(self.in_features):
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            for i in range(self.in_features):
                self.bias[i].data.uniform_(-stdv, stdv)


    def forward(self, input):
        #print('input_size: {}'.format(input.size()))
        #(class_num, feature_dim)
        #print('weight size: {}'.format(self.weight.size()))
        x = input * self.weight
        #(class_num, 1)
        x = torch.sum(x,2)
        #print('after reducing(sum): {}'.format(x.size()))
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


class BC_DFL(nn.Module):
    """build Resnet + BCNN + BiRNN Architecture for the second stage of training"""
    def __init__(self, Bottleneck,
                       feature_dim,
                       representation,
                       middle_dim,
                       time_step,
                       adjacency_matrix, 
                       K, 
                       num_classes):
                       
        super(BC_DFL, self).__init__()
        self.feature_dim = feature_dim
        self.middle_dim = middle_dim
        self.K = K
        self.num_classes=num_classes
        self.feature = Multi_label_model(Bottleneck, [3, 4, 6, 3])
        self.BiRNN = nn.LSTM(self.feature_dim, self.middle_dim, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(self.feature_dim, self.K*self.num_classes, kernel_size=1, stride=1, bias=False)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avepool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=False)
        if representation is not None:
            representation_method = representation['function']
            representation.pop('function')
            representation_args = representation
            self.representation = representation_method(**representation_args)
        self.adjacency_matrix = adjacency_matrix
        self._in_matrix, self._out_matrix = self.load_matrix(t=0.2)
        self.time_step = time_step
        
        self.graph_net = GGNN(input_dim=self.K,
                              time_step=self.time_step,
                              in_matrix=self._in_matrix, 
                              out_matrix=self._out_matrix)
        self.classifier_G = nn.Linear(self.feature_dim*2*self.middle_dim, num_classes)
        self.classifier_S = Element_Wise_Layer(self.num_classes, self.K)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #self.classifier_S = nn.Linear(self.num_classes*71*self.K, self.num_classes)
        
                
    def load_matrix(self, t):
        result = pickle.load(open(self.adjacency_matrix, 'rb'))
        _adj = result['adj']  # (80,80)  #  main diagonal = 0
        _nums = result['nums']  # the numbers of each category
        _nums = _nums[:, np.newaxis]  # [[2243], [1171],[..]...]  # (80, 1)

        _adj = _adj / _nums  # P
        _adj[_adj < t] = 0  # binary
        _adj[_adj >= t] = 1
        _in_matrix, _out_matrix = _adj.astype(np.float32), _adj.T.astype(np.float32)
        _in_matrix = Variable(torch.from_numpy(_in_matrix), requires_grad=False).cuda()
        _out_matrix = Variable(torch.from_numpy(_out_matrix), requires_grad=False).cuda()
        return _in_matrix, _out_matrix
        
    def forward(self, x):
        feature_g, feature_l = self.feature(x)
        feature_ = self.representation(feature_g) # [32, 2048, 2048]
        
        
        x_G, _ = self.BiRNN(feature_) # [32, 2048, 100]
        x_G = self.leakyrelu(x_G) 
        x_G = x_G.view(x_G.size(0), -1) # [32, 2048*100]
        x_G = self.classifier_G(x_G) # [32, 10]
        
        x_S = self.conv(feature_g) # [32, K*num_classes, 313]
        x_S = self.leakyrelu(x_S)
        x_S = self.maxpool(x_S) # [32, K*num_classes, 1]
        
        x_S = self.leakyrelu(x_S)
        x_S = x_S.view(x_S.size(0), self.num_classes, -1) # [32,num_classes, K]
        crs = self.avepool(x_S) # [32, num_classes, 1]
        crs = self.leakyrelu(crs)
        x_S = self.graph_net(x_S)  # [32,num_classes, K]
        
        #x_S = x_S.view(x_S.size(0), -1)  # [32, num_classes, K]
        x_S = torch.tanh(x_S)
        x_S = self.classifier_S(x_S) # [32, num_classes]
        
        
        return x_G, x_S, crs.view(crs.size(0), -1)


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

