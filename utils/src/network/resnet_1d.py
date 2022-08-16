import torch.nn as nn
import torch.utils.model_zoo as model_zoo


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
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3_1D(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1_1D(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=False)
        #self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
       
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
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
     
        return x4
        
class Resnet_101(nn.Module):
    def __init__(self, num_classes):
        super(Resnet_101, self).__init__()
        self.feature = Multi_label_model(Bottleneck, [3, 4, 23, 3])
        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(2048, num_classes)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=False)
    def forward(self, x):
        x_G = self.feature(x)  # [32, 2048, 313]
        x_G = self.avgpooling(x_G)  # [32, 2048, 1]
        x_G = self.leakyrelu(x_G)
        x_G = x_G.view(x_G.size(0), -1)
        output = self.linear(x_G)
        return output 