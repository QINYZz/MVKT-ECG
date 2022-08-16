import torch
import torch.nn as nn
from torch.nn import Module


def Conv1d_leakyrelu(in_channel, out_channel, kernel_size, stride=1, padding=0):
    padding = kernel_size // 2
    return nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride, padding=padding),
                         nn.LeakyReLU(0.3, inplace=False))


def conv_block(in_channel, channel, final_kernel):
    block = nn.Sequential(
        Conv1d_leakyrelu(in_channel, channel, kernel_size=3),
        Conv1d_leakyrelu(channel, channel, kernel_size=3),
        Conv1d_leakyrelu(channel, channel, kernel_size=final_kernel, stride=2)
    )
    return block


class Classical(Module):
    def __init__(self, channel, signal_dim=1, num_classes=10):
        super(Classical, self).__init__()
        self.input_dim = signal_dim
        self.channel = channel
        self.num_classes = num_classes
        self.conv_block_1 = conv_block(self.input_dim, self.channel, 24)
        self.drop = nn.Dropout(0.2)
        self.conv_block_2 = conv_block(self.channel, self.channel, 24)
        self.conv_block_3 = conv_block(self.channel, self.channel, 24)
        self.conv_block_4 = conv_block(self.channel, self.channel, 24)
        self.conv_block_5 = conv_block(self.channel, self.channel, 48)
        self.gru = nn.GRU(input_size=12, hidden_size=12, num_layers=1, batch_first=True, bidirectional=True)
        self.batchnorm = nn.BatchNorm1d(24)
        self.leakyrelu = nn.LeakyReLU(0.3, inplace=False)
        self.fc = nn.Linear(24, self.num_classes)
        self.fc_ = nn.Linear(3768, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        out = self.conv_block_1(x)
        out = self.drop(out)
        out = self.conv_block_2(out)
        out = self.drop(out)
        out = self.conv_block_3(out)
        out = self.drop(out)
        out = self.conv_block_4(out)
        out = self.drop(out)
        out = self.conv_block_5(out)
        cnnout = self.drop(out)
        import pdb
        # pdb.set_trace()
        # out = cnnout.permute(0, 2, 1)
        #out, _ = self.gru(out)
        #out = out[:, -1, :]
        
        # out = self.leakyrelu(out)
        # out = self.drop(out)
        # out = self.batchnorm(cnnout)
        # out = self.leakyrelu(cnnout)
        out = cnnout.view(cnnout.size(0), -1)
        out = self.fc_(out)
        out = self.sigmoid(out)
        return out


def classical(channel=12):
    model = Classical(channel)
    return model