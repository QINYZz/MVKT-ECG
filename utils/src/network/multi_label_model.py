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
        self.maxpool2 = nn.AdaptiveMaxPool1d((1))        
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
        import pdb
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.maxpool1(x)

        x1 = self.layer1(x)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)  #[32, 2048, 313]
        x1 = self.conv2(x1)
        
        x1 = x1.permute(0, 2, 1)
        x1, _ = self.BiRNN(x1)
        x1 = x1.permute(0, 2, 1) # [32, 300, 313]
        x1 = self.maxpool2(x1) # [32, 2048, 1]
        x1 = x1.view(x.size(0), -1)
        x1 = self.fc_LSTM(x1)
        return x1


class Resnet_BCNN(nn.Module):
    """build Resnet + BCNN + BiRNN Architecture"""
    def __init__(self, representation, num_classes, freezed_layer, pretrained=False):
        super(Resnet_BCNN, self).__init__()
        self.model = Multi_label_model(Bottleneck, [3, 4, 6, 3])
        self.feature = nn.Sequential(*list(self.model.children())[:-2])
        self.BiRNN = nn.LSTM(512 * 4, 50, batch_first=True, bidirectional=True)
        self.maxpool2 = nn.AdaptiveMaxPool1d((1))    
        self.linear = nn.Linear(300, num_classes)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=False)
        if representation is not None:
            representation_method = representation['function']
            representation.pop('function')
            representation_args = representation
            self.representation = representation_method(**representation_args)
            
    def forward(self, x):
        x = self.feature(x)
        x = self.representation(x) # [32, 2048, 2048]
        #pdb.set_trace()
        x, _ = self.BiRNN(x)  # [32, 2048, 300]
       # x = x[:, -1, :] # [32, 300]
        # x = self.linear(x) # [32, 10]
        x = self.leakyrelu(x)
        return x
        

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


class GCNResnet(nn.Module):
    def __init__(self, representation, num_classes, freezed_layer, in_channel=300, t=0, adj_file=None, pretrained=False):
        super(GCNResnet, self).__init__()
        self.resnet_bcnn = Resnet_BCNN(representation, num_classes, freezed_layer, pretrained=False)
        self.num_classes = num_classes
        
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.classi = nn.Linear(2048*110, num_classes)
        self.sigmoid = nn.Sigmoid()
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())  #

    def forward(self, x, inp):
        import pdb
        # pdb.set_trace()
        feature = self.resnet_bcnn(x) # [32, 300]
        batch = feature.size(0)
        # feature = feature.view(feature.size(0), -1)
        inp = inp[0]  # what is inp??  label embedding because data_loader would return "batch_size" label embedding
        adj = gen_adj(self.A).detach()
        
        x = self.gc1(inp, adj) 
        x = self.leakyrelu(x)
        x = self.gc2(x, adj)  # [10,2048]
        x = x.transpose(0, 1) # [2048,10]
       # pdb.set_trace()
        
        feature_new = np.zeros((batch, 2048, 110))
        feature_new = Parameter(torch.from_numpy(feature_new).float()).cuda()
        for i in range(batch):
            feature_new[i] = torch.cat([feature[i], x], 1) # [32, 2048, 310]
        feature_new = feature_new.view(feature_new.size(0), -1)
        # x = torch.matmul(feature, x)
        x = self.classi(feature_new) 
        x = self.sigmoid(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]


def GCN_Resnet_BCNN(representation, num_classes, freezed_layer, pretrained=False):
    """Constructs a two-channel model. First is the "feature channel", consists of Resnet and BCNN;
                                       Second is the "GCN channel" including two GNN layers.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GCNResnet(representation, num_classes, freezed_layer, in_channel=300, t=0.20, adj_file="/home/workspace/huichen/codes/GCN/ecg_adj.pkl", pretrained=False)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
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
