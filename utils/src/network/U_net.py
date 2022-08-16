import torch
import torch.nn as nn




def conv_relu(in_channel, out_channel, kernel, stride=1):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size=kernel, padding=kernel // 2),
        nn.LeakyReLU(inplace=False),
    )


class U_net(nn.Module):
    def __init__(self, channel, kernel,num_classes=10):
        super(U_net, self).__init__()
        self.conv1_1 = conv_relu(channel[0], channel[1], kernel=kernel)
        self.conv1_2 = conv_relu(channel[1], channel[1], kernel=kernel)
        self.maxpool1 = nn.MaxPool1d(2)

        self.conv2_1 = conv_relu(channel[1], channel[2], kernel=kernel)
        self.drop1 = nn.Dropout(0.2)
        self.conv2_2 = conv_relu(channel[2], channel[2], kernel=kernel)
        self.maxpool2 = nn.MaxPool1d(2)

        self.conv3_1 = conv_relu(channel[2], channel[3], kernel=kernel)
        self.conv3_2 = conv_relu(channel[3], channel[3], kernel=kernel)
        self.maxpool3 = nn.MaxPool1d(2)

        self.conv4_1 = conv_relu(channel[3], channel[4], kernel=kernel)
        self.drop2 = nn.Dropout(0.2)
        self.conv4_2 = conv_relu(channel[4], channel[4], kernel=kernel)
        self.maxpool4 = nn.MaxPool1d(2)

        self.conv5_1 = conv_relu(channel[4], channel[5], kernel=kernel)
        self.conv5_2 = conv_relu(channel[5], channel[5], kernel=kernel)

        self.up1 = nn.ConvTranspose1d(channel[5], channel[4], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv_4_1 = conv_relu(channel[5], channel[4], kernel=kernel)
        self.conv_4_2 = conv_relu(channel[4], channel[4], kernel=kernel)

        self.up2 = nn.ConvTranspose1d(channel[4], channel[3], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv_3_1 = conv_relu(channel[4], channel[3], kernel=kernel)
        self.conv_3_2 = conv_relu(channel[3], channel[3], kernel=kernel)

        self.up3 = nn.ConvTranspose1d(channel[3], channel[2], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv_2_1 = conv_relu(channel[3], channel[2], kernel=kernel)
        self.conv_2_2 = conv_relu(channel[2], channel[2], kernel=kernel)

        self.up4 = nn.ConvTranspose1d(channel[2], channel[1], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv_1_1 = conv_relu(channel[2], channel[1], kernel=kernel)
        self.conv_1_2 = conv_relu(channel[1], channel[1], kernel=kernel)

 
        self.conv6= nn.Conv1d(channel[1], channel[1], kernel_size=31, stride=2, padding=0)
        self.relu = nn.LeakyReLU(inplace=False)
        self.conv7= nn.Conv1d(channel[1], channel[1], kernel_size=31, stride=2, padding=0)
        self.conv8= nn.Conv1d(channel[1], 1, kernel_size=31, stride=2, padding=0)
        self.maxpool5 = nn.MaxPool1d(2)  # (batch, 10, 1248)
        self.conv = nn.Conv1d(channel[1], 1, kernel_size=1, stride = 1, padding=0)
        self.BiRNN = nn.LSTM(input_size=channel[1], hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        # self.attention = AttentionWithContext()
        self.maxpool5 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(10000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.1)
        # self.fc = nn.Linear(1248, num_classes)
        self.maxpool = nn.AdaptiveMaxPool1d((num_classes))
        
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # nn.init.normal_(m.weight, 0, 0.05)
                nn.init.kaiming_normal(m.weight, a=0, mode='fan_in')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        import pdb
        #pdb.set_trace()
        out = self.conv1_1(x)
        cat1_1 = self.conv1_2(out)
        out = self.maxpool1(cat1_1)

        out = self.conv2_1(out)
        cat2_1 = self.conv2_2(out)
        out = self.maxpool2(cat2_1)

        out = self.conv3_1(out)
        cat3_1 = self.conv3_2(out)
        out = self.maxpool3(cat3_1)

        out = self.conv4_1(out)
        cat4_1 = self.conv4_2(out)
        out = self.maxpool4(cat4_1)

        out = self.conv5_1(out)
        out = self.conv5_2(out)

        cat4_2 = self.up1(out)
        cat4 = torch.cat([cat4_1, cat4_2], 1)

        out = self.conv_4_1(cat4)
        out = self.conv_4_2(out)

        cat3_2 = self.up2(out)
        cat3 = torch.cat([cat3_1, cat3_2], 1)

        out = self.conv_3_1(cat3)
        out = self.conv_3_2(out)

        cat2_2 = self.up3(out)
        cat2 = torch.cat([cat2_1, cat2_2], 1)

        out = self.conv_2_1(cat2)
        out = self.conv_2_2(out)

        cat1_2 = self.up4(out)
        cat1 = torch.cat([cat1_1, cat1_2], 1) 

        out = self.conv_1_1(cat1)
        out = self.conv_1_2(out)
        # input:(batch,length,channel)
        # so need to change the axis
        # out = self.maxpool5(out)
        # out = out.permute(0,2,1)
        # out,_ = self.BiRNN(out) # out:[32, 5000, 256]
        # out = out[:,-1, :] # [32, 256]
        # out = self.attention(out)
        # out = self.out = self.dropout(out)
        out = self.conv(out)# 
        out = self.relu(out)
        out = out.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        
        # out = self.maxpool(out) 
        # out = self.relu(out)
        
        # pdb.set_trace()
        out = self.sigmoid(out)
        
        return out


def u_net(channel=[1, 16, 32, 64, 128, 256], kernel=31):
    model = U_net(channel, kernel)
    return model


if __name__ == '__main__':
    channel=[1, 16, 32, 64, 128, 256]
    a = torch.rand(16,1,10000)
    model = u_net(channel)
    b = model(a)
    
