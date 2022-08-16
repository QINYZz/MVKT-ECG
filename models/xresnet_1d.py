import torch
import torch.nn as nn

# Cell
def conv(n_inputs, n_filters, kernel_size=3, stride=1, bias=False) -> torch.nn.Conv1d:
    """Creates a convolution layer for `XResNet`."""
    return nn.Conv1d(n_inputs, n_filters,
                     kernel_size=kernel_size, stride=stride,
                     padding=kernel_size//2, bias=bias)

def conv_layer(n_inputs: int, n_filters: int,
               kernel_size: int = 3, stride=1,
               zero_batch_norm: bool = False, use_activation: bool = True,
               activation: torch.nn.Module = nn.ReLU(inplace=True)) -> torch.nn.Sequential:
    """Creates a convolution block for `XResNet`."""
    batch_norm = nn.BatchNorm1d(n_filters)
    # initializer batch normalization to 0 if its the final conv layer
    nn.init.constant_(batch_norm.weight, 0. if zero_batch_norm else 1.)
    layers = [conv(n_inputs, n_filters, kernel_size, stride=stride), batch_norm]
    if use_activation: layers.append(activation)
    return nn.Sequential(*layers)

class XResNetBlock(nn.Module):
    """Creates the standard `XResNet` block."""
    def __init__(self, expansion: int, n_inputs: int, n_hidden: int, stride: int = 1,
                 activation: torch.nn.Module = nn.ReLU(inplace=True)):
        super().__init__()

        n_inputs = n_inputs * expansion
        n_filters = n_hidden * expansion

        # convolution path
        if expansion == 1:
            layers = [conv_layer(n_inputs, n_hidden, 3, stride=stride),
                      conv_layer(n_hidden, n_filters, 3, zero_batch_norm=True, use_activation=False)]
        else:
            layers = [conv_layer(n_inputs, n_hidden, 1),
                      conv_layer(n_hidden, n_hidden, 3, stride=stride),
                      conv_layer(n_hidden, n_filters, 1, zero_batch_norm=True, use_activation=False)]

        self.convs = nn.Sequential(*layers)

        # identity path
        if n_inputs == n_filters:
            self.id_conv = nn.Identity()
        else:
            self.id_conv = conv_layer(n_inputs, n_filters, kernel_size=1, use_activation=False)
        if stride == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool1d(2, ceil_mode=True)

        self.activation = activation

    def forward(self, x):
        return self.activation(self.convs(x) + self.id_conv(self.pool(x)))

class XResNet(nn.Module):
    def __init__(self, expansion, layers, num_leads=12, num_classes=9):
        super(XResNet, self).__init__()

        n_filters_stem = [num_leads, 32, 64, 64]
        stem = [conv_layer(n_filters_stem[i], n_filters_stem[i + 1], stride=2 if i == 0 else 1)
                for i in range(3)]

        self.stem = nn.Sequential(*stem)
        n_filters_xres = [64 // expansion, 64, 128, 256, 512]

        res_layers = [self._make_layer(expansion, n_filters_xres[i], n_filters_xres[i + 1],
                                  n_blocks=l, stride=1 if i == 0 else 2)
                        for i, l in enumerate(layers)]

        self.res_layers = nn.Sequential(*res_layers)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool_final = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(2*n_filters_xres[-1]*expansion, num_classes)  
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, expansion, n_inputs, n_filters, n_blocks, stride):
        return nn.Sequential(
            *[XResNetBlock(expansion, n_inputs if i==0 else n_filters, n_filters, stride if i==0 else 1)
              for i in range(n_blocks)])
    
    '''def _forward_impl(self, x, is_feat=False):
        # See note [TorchScript super()]
        x = self.stem(x)
        from IPython import embed
        #embed()
        x = self.maxpool(x)
        x = self.res_layers(x)
        x_avg = self.avgpool(x)
        x_avg = torch.flatten(x_avg , 1)
        x_max = self.maxpool_final(x)
        x_max = torch.flatten(x_max,1)
        feature = torch.cat((x_avg,x_max), 1)
        x = self.fc(feature)
        if is_feat:
            return [feature], x
        else:
            return x'''
    def _forward_impl(self, x, is_feat_crd=False, is_feat_dist=False):
        # See note [TorchScript super()]
        features = []
        x = self.stem(x)
        from IPython import embed
        #embed()
        x = self.maxpool(x)
        x = self.res_layers[0](x)
        features.append(x)
        x = self.res_layers[1](x)
        features.append(x)
        x = self.res_layers[2](x)
        features.append(x)
        x = self.res_layers[3](x)
        features.append(x)

        x_avg = self.avgpool(x)
        x_avg = torch.flatten(x_avg , 1)
        x_max = self.maxpool_final(x)
        x_max = torch.flatten(x_max,1)
        feature = torch.cat((x_avg,x_max), 1)
        #feature = x_avg
        x = self.fc(feature)
        if is_feat_crd:
            return features, [feature], x
        elif is_feat_dist:
            return features , x
        else:
            return x
    def forward(self, x, is_feat_crd=False, is_feat_dist=False):
        return self._forward_impl(x, is_feat_crd, is_feat_dist)



def xresnet18 (**kwargs): return XResNet(1, [2, 2,  2, 2], **kwargs)
def xresnet34 (**kwargs): return XResNet(1, [3, 4,  6, 3], **kwargs)
def xresnet50 (**kwargs): return XResNet(4, [3, 4,  6, 3], **kwargs)
def xresnet101(**kwargs): return XResNet(4, [3, 4, 23, 3], **kwargs)
def xresnet152(**kwargs): return XResNet(4, [3, 8, 36, 3], **kwargs)

if __name__ =='__main__':
    net = xresnet101()
    print(net)
    _input = torch.rand(2, 12, 10000)
    out = net(_input)