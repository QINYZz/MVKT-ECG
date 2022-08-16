import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch

class semantic(nn.Module):
    def __init__(self, num_classes, image_feature_dim, word_feature_dim, intermediary_dim=1024):
        super(semantic, self).__init__()
        self.num_classes = num_classes
        self.image_feature_dim = image_feature_dim
        self.word_feature_dim = word_feature_dim
        self.intermediary_dim = intermediary_dim
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=False)
        #self.fc_0 = nn.Linear(self.num_classes, self.word_feature_dim)
        self.fc_1 = nn.Linear(self.image_feature_dim, self.intermediary_dim, bias=False)
        self.fc_2 = nn.Linear(self.word_feature_dim, self.intermediary_dim, bias=False)
        self.fc_3 = nn.Linear(self.intermediary_dim, self.intermediary_dim)
        self.fc_a = nn.Linear(self.intermediary_dim, 1)

    def forward(self,batch_size, img_feature_map, word_features):

        length = img_feature_map.size()[2]
        import pdb
        #pdb.set_trace()
        img_feature_map = torch.transpose(img_feature_map, 1, 2)
        f_wh_feature = img_feature_map.contiguous().view(batch_size*length, -1)
        f_wh_feature = self.fc_1(f_wh_feature).view(batch_size*length, 1, -1).repeat(1, self.num_classes, 1)
        #word_features = self.fc_0(word_features)
        f_wd_feature = self.fc_2(word_features).view(1, self.num_classes, 1024).repeat(batch_size*length, 1, 1)
        lb_feature = torch.tanh(f_wh_feature*f_wd_feature).view(-1,1024)
        coefficient = self.fc_a(lb_feature)
        #coefficient = self.leakyrelu(coefficient)
        coefficient = torch.transpose(coefficient.view(batch_size, length, self.num_classes),1,2)

        coefficient = F.softmax(coefficient, dim=2)
        print(coefficient)
        coefficient = torch.transpose(coefficient,1,2)
        coefficient = coefficient.view(batch_size, length, self.num_classes, 1).repeat(1,1,1,self.image_feature_dim)
        img_feature_map = img_feature_map.view(batch_size, length, 1, self.image_feature_dim).repeat(1, 1, self.num_classes, 1) * (coefficient)
        # graph_net_input = torch.sum(img_feature_map,1)
        return img_feature_map
