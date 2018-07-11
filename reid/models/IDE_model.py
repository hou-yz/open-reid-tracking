from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from .resnet import *
import torchvision


class IDE_model(nn.Module):
    def __init__(self, num_features=256, num_classes=0, norm=False, dropout=0):
        super(IDE_model, self).__init__()
        # Create IDE_only model
        self.num_features = num_features
        self.num_classes = num_classes

        # ResNet50: from 3*384*128 -> 2048*12*4 (Tensor T; of column vector f's)
        self.base = nn.Sequential(
            *list(resnet50(pretrained=True, cut_at_pooling=True, norm=norm, dropout=dropout).base.children())[:-2])

        ################################################################################################################
        '''Global Average Pooling: 2048*12*4 -> 2048*1*1'''
        # Tensor T [N, 2048, 1, 1]
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # dropout after pool5 (or what left of it) at p=0.5
        self.dropout = dropout
        self.drop_layer = nn.Dropout2d(self.dropout)

        ################################################################################################################

        # 1*1 Conv(fc): 1*1*2048 -> 1*1*256 (g -> h)
        # 6 separate convs
        self.one_one_conv = nn.Sequential(nn.Conv2d(2048, self.num_features, 1, bias=False),
                                          nn.BatchNorm2d(self.num_features))
        init.kaiming_normal(self.one_one_conv[0].weight, mode='fan_out')

        # fc + softmax:
        if self.num_classes > 0:
            self.fc = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.fc.weight, std=0.001)
            init.constant(self.fc.bias, 0)

        pass

    def forward(self, x):
        """
        Returns:
          h_s: each member with shape [N, c]
          prediction_s: each member with shape [N, num_classes]
        """
        # Tensor T [N, 2048, 12, 4]
        x = self.base(x)
        x = self.global_avg_pool(x)
        if self.dropout:
            x = self.drop_layer(x)

        x = self.one_one_conv(x).view(x.size()[0], self.num_features)

        prediction = self.fc(x)

        x_s = x.view(x.shape[0], -1)
        prediction_s = list(prediction)

        return x_s, prediction_s
