import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from .lib.non_local_simple_version import NONLocalBlock1D

__all__ = ['MultiBranchNet', 'ECG_7', 'ECG_4', 'ECG_6', 'ECG_1', 'ECG_2', 'ECG_3', 'ECG_5',\
           'ECG_8', 'ECG_0']


def conv1_x(in_channel, out_channel, kernel_size, stride, batch_norm):
    conv_layer = []
    conv = nn.Conv1d(in_channel, out_channel, \
              kernel_size=kernel_size, \
              stride=stride, \
              padding=int((kernel_size-1)/2), bias=False)
    conv_layer.append(conv)

    if batch_norm == True:
        conv_layer.append(nn.BatchNorm1d(out_channel))
    conv_layer.append(nn.ReLU(inplace=False))
    for m in conv_layer:
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, 0, 0.1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return nn.Sequential(*conv_layer)


class Branch(nn.Module):
    def __init__(self, kernel, channel, stride, batch_norm = True):
        super(Branch, self).__init__()
        self.kernel = kernel
        self.channel = channel
        self.stride = stride
        ## check the arguements
        assert len(kernel) == len(channel)-1 == len(stride)
        self.layer_num = len(kernel)
        layer_group = []
        in_channel = 0
        out_channel = 0
        for i in range(self.layer_num):
            in_channel += channel[i]
            out_channel = channel[i+1]
            layer_group.append(
                conv1_x(in_channel, out_channel, kernel[i], \
                        stride[i], batch_norm).cuda()
            )
        #import pdb; pdb.set_trace()
        self.layer_group = layer_group

    def forward(self, x):
        for i in range(self.layer_num):
            func = self.layer_group[i]
            y = func(x)
            length = y.size(-1)
            x = F.adaptive_avg_pool1d(x, length)
            out = torch.cat([x, y], 1)
            x = out
        return out


class MultiBranchNet(nn.Module):
    def __init__(self, kernel, channel, stride, batch_norm = True):
        super(MultiBranchNet, self).__init__()
        assert len(kernel) == len(channel) == len(stride)
        self.branch_num = len(kernel)
        branch = []
        for i in range(self.branch_num):
            branch.append(
                Branch(kernel[i], channel[i], stride[i], batch_norm)
            )
        self.branch = branch

    def forward(self, x):
        out = []
        for i in range(self.branch_num):
            branch = self.branch[i]
            out.append(branch(x))
        return torch.cat(out,1)


class Baseline(nn.Module):
    def __init__(self, channel, kernel, stride, img_channel=12):
        super(Baseline, self).__init__()
        self.conv1_1 = conv1_x(img_channel, channel[0], kernel[0], stride[0], False)
        self.conv1_2 = conv1_x(channel[0], channel[1], kernel[1], stride[1], False)
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv1_3 = conv1_x(channel[1], channel[2], kernel[2], stride[2], False)
        self.conv1_4 = conv1_x(channel[2], channel[3], kernel[3], stride[3], False)
        self.maxpool2 = nn.MaxPool1d(2)

        self.conv2_1 = conv1_x(img_channel, channel[0], kernel[0], stride[0], False)
        self.conv2_2 = conv1_x(channel[0], channel[1], kernel[1], stride[1], False)
        self.avgpool1 = nn.AvgPool1d(2)
        self.conv2_3 = conv1_x(channel[1], channel[2], kernel[2], stride[2], False)
        self.conv2_4 = conv1_x(channel[2], channel[3], kernel[3], stride[3], False)
        self.avgpool2 = nn.AvgPool1d(2)
        self.feature_dim = channel[3]*2
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# class Baseline(nn.Module):
#     # by chelsea
#     def __init__(self, channel, kernel, stride, img_channel=12):
#         super(Baseline, self).__init__()
#         self.conv1_1 = conv1_x(img_channel, channel[0], kernel[0][0], stride[0], False)
#         self.conv1_2 = conv1_x(channel[0], channel[1], kernel[0][1], stride[1], False)
#         self.maxpool1 = nn.MaxPool1d(2)
#         self.conv1_3 = conv1_x(channel[1], channel[2], kernel[0][2], stride[2], False)
#         self.conv1_4 = conv1_x(channel[2], channel[3], kernel[0][3], stride[3], False)
#         self.maxpool2 = nn.MaxPool1d(2)
#
#         self.conv2_1 = conv1_x(img_channel, channel[0], kernel[1][0], stride[0], False)
#         self.conv2_2 = conv1_x(channel[0], channel[1], kernel[1][1], stride[1], False)
#         self.avgpool1 = nn.AvgPool1d(2)
#         self.conv2_3 = conv1_x(channel[1], channel[2], kernel[1][2], stride[2], False)
#         self.conv2_4 = conv1_x(channel[2], channel[3], kernel[1][3], stride[3], False)
#         self.avgpool2 = nn.AvgPool1d(2)
#         self.feature_dim = channel[3] * 2
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.normal_(m.weight, 0, 0.1)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.maxpool1(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.maxpool2(x1)

        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)
        x2 = self.avgpool1(x2)
        x2 = self.conv2_3(x2)
        x2 = self.conv2_4(x2)
        x2 = self.avgpool2(x2)

        out = torch.cat([x1, x2], 1)
        return out

# def ECG_7(pretrained=False, **kwargs):
#     #kernel = [[5, 3, 3, 3, 1, 1],[7, 5, 5, 5, 3, 3],[9, 7, 7, 5, 5, 3]]
#     #channel = [[12, 32, 32, 32, 32, 32, 32],[12, 32, 32, 32, 32, 32, 32],[12, 32, 32, 32, 32, 32, 32]]
#     #stride = [[1, 4, 2, 2, 2, 2],[1, 4, 2, 2, 2, 2],[1, 4, 2, 2, 2, 2]]
#     kernel = [[9, 7, 7, 5]]
#     channel = [[12, 32, 32, 64, 64]]
#     stride = [[1, 4, 2, 2]]
#     model = MultiBranchNet(kernel, channel, stride, batch_norm=False)
#     #model = Baseline()
#     return model


class PAC(nn.Module):
    def __init__(self, channel, kernel, stride, img_channel=12):
        super(PAC, self).__init__()
        self.input_bn = nn.BatchNorm1d(img_channel)
        self.conv1_1 = conv1_x(img_channel, channel[0], kernel[0][0], stride[0], False)
        self.bn1_1 = nn.BatchNorm1d(channel[0])
        self.conv1_2 = conv1_x(channel[0], channel[1], kernel[0][1], stride[1], False)
        self.bn1_2 = nn.BatchNorm1d(channel[1])
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv1_3 = conv1_x(channel[1], channel[2], kernel[0][2], stride[2], False)
        self.bn1_3 = nn.BatchNorm1d(channel[2])
        self.conv1_4 = conv1_x(channel[2], channel[3], kernel[0][3], stride[3], False)
        self.bn1_4 = nn.BatchNorm1d(channel[3])
        self.maxpool2 = nn.MaxPool1d(2)

        self.conv2_1 = conv1_x(img_channel, channel[0], kernel[1][0], stride[0], False)
        self.bn2_1 = nn.BatchNorm1d(channel[0])
        self.conv2_2 = conv1_x(channel[0], channel[1], kernel[1][1], stride[1], False)
        self.bn2_2 = nn.BatchNorm1d(channel[1])
        self.avgpool1 = nn.AvgPool1d(2)
        self.conv2_3 = conv1_x(channel[1], channel[2], kernel[1][2], stride[2], False)
        self.bn2_3 = nn.BatchNorm1d(channel[2])
        self.conv2_4 = conv1_x(channel[2], channel[3], kernel[1][3], stride[3], False)
        self.bn2_4 = nn.BatchNorm1d(channel[3])
        self.avgpool2 = nn.AvgPool1d(2)
        self.feature_dim = channel[3] * 2
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.05)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        x1 = self.conv1_1(x)
        x1 = self.bn1_1(x1)
        x1 = self.conv1_2(x1)
        x1 = self.bn1_2(x1)
        x1 = self.maxpool1(x1)
        x1 = self.conv1_3(x1)
        x1 = self.bn1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.bn1_4(x1)
        x1 = self.maxpool2(x1)

        x2 = self.conv2_1(x)
        x2 = self.bn2_1(x2)
        x2 = self.conv2_2(x2)
        x2 = self.bn2_2(x2)
        x2 = self.avgpool1(x2)
        x2 = self.conv2_3(x2)
        x2 = self.bn2_3(x2)
        x2 = self.conv2_4(x2)
        x2 = self.bn2_4(x2)
        x2 = self.avgpool2(x2)

        out = torch.cat([x1, x2], 1)
        return out


def ECG_4(pretrained=False, **kwargs):
    channel = [16, 16, 64, 64]
    kernel = [8, 4, 8, 4]
    stride = [2, 2, 2, 1]
    model = Baseline(channel, kernel, stride)
    return model


def ECG_6(pretrained=False, **kwargs):
    channel = [16, 16, 64, 64]
    # kernel = [64, 64, 32, 32]
    kernel = [[64, 64, 32, 32], [64, 64, 32, 32]]
    stride = [2, 2, 2, 1]
    model = PAC(channel, kernel, stride, img_channel=12)
    return model


def ECG_1(pretrained=False, **kwargs):
    channel = [16, 16, 64, 64]
    kernel = [[8, 8, 4, 4], [8, 8, 4, 4]]
    stride = [2, 2, 2, 1]
    model = PAC(channel, kernel, stride, img_channel=12)
    return model


def ECG_2(pretrained=False, **kwargs):
    channel = [16, 16, 64, 64]
    kernel = [[64, 64, 32, 32], [64, 64, 32, 32]]
    stride = [2, 2, 2, 1]
    model = PAC(channel, kernel, stride, img_channel=12)
    return model


def ECG_3(pretrained=False, **kwargs):
    channel = [16, 16, 64, 64]
    kernel = [[8, 8, 4, 4], [8, 8, 4, 4]]
    stride = [2, 2, 2, 1]
    model = PAC(channel, kernel, stride, img_channel=12)
    return model

def ECG_5(pretrained=False, **kwargs):
    channel = [16, 16, 64, 64]
    # kernel = [64, 64, 32, 32]
    kernel = [[64, 64, 32, 32], [64, 64, 32, 32]]
    stride = [2, 2, 2, 1]
    model = PAC(channel, kernel, stride, img_channel=12)
    return model

def ECG_0(pretrained=False, **kwargs):
    channel = [16, 16, 64, 64]
    kernel = [8, 8, 8, 8]
    stride = [2, 2, 2, 1]
    model = Baseline(channel, kernel, stride)
    return model

def ECG_8(pretrained=False, **kwargs):
    channel = [16, 16, 64, 64]
    kernel = [[8, 8, 8, 8], [8, 8, 8, 8]]
    stride = [2, 2, 2, 1]
    model = PAC(channel, kernel, stride, img_channel=12)
    return model


def ECG_7(pretrained=False, **kwargs):
    # by Chelsea
    channel = [16, 16, 64, 64]
    # kernel = [64, 64, 32, 32]
    kernel = [[8, 8, 4, 4], [8, 8, 4, 4]]
    stride = [2, 2, 2, 1]
    model = PAC(channel, kernel, stride, img_channel=2)
    return model