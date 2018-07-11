from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from .resnet import *
import torchvision


class PCB_model(nn.Module):
    def __init__(self, num_parts=6, num_features=256, num_classes=0, norm=False, dropout=0):
        super(PCB_model, self).__init__()
        # Create PCB_only model
        self.num_parts = num_parts
        self.num_features = num_features
        self.num_classes = num_classes
        self.rpp = False

        # ResNet50: from 3*384*128 -> 2048*24*8 (Tensor T; of column vector f's)
        self.base = nn.Sequential(
            *list(resnet50(pretrained=True, cut_at_pooling=True, norm=norm, dropout=dropout).base.children())[:-2])
        # decrease the downsampling rate
        # change the stride2 conv layer in self.layer4 to stride=1
        self.base[7][0].conv2.stride = (1, 1)
        # change the downsampling layer in self.layer4 to stride=1
        self.base[7][0].downsample[0].stride = (1, 1)

        self.f_dimension = self.base[7][2].conv3.out_channels  # 2048
        ################################################################################################################
        '''Average Pooling: 2048*24*8 -> 2048*6*1 (f -> g)'''
        # Tensor T [N, 2048, 24, 8]
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 1))

        # dropout after pool5 (or what left of it) at p=0.5
        self.dropout = dropout
        self.drop_layer = nn.Dropout2d(self.dropout)

        '''RPP: Refined part pooling'''
        # get sampling weights from f [2048*1*1]
        # avg pooling along the channel dimension for a [256*1*1]
        self.classifier_pool = nn.AdaptiveAvgPool1d(self.f_dimension)
        # 6 classifier for f:[256*1*1] -> weight_s:[6*1*1]
        self.sampling_weight_layer = nn.Sequential(
            nn.Conv1d(self.f_dimension, self.num_parts, kernel_size=1),
            nn.Softmax(dim=1))

        # return a [N,6,24,8] tensor
        ################################################################################################################

        # 1*1 Conv: 6*1*2048 -> 6*1*256 (g -> h)
        self.one_one_conv = nn.Sequential(nn.Conv2d(self.f_dimension, self.num_features, 1),
                                          nn.BatchNorm2d(self.num_features))

        init.kaiming_normal(self.one_one_conv[0].weight, mode='fan_out'),
        init.constant(self.one_one_conv[1].weight, 1)
        init.constant(self.one_one_conv[1].bias, 0)

        # 6 branches of fc's:
        if self.num_classes > 0:
            self.fc_s = nn.ModuleList()
            for _ in range(self.num_parts):
                fc = nn.Linear(self.num_features, self.num_classes)
                init.normal(fc.weight, std=0.001)
                init.constant(fc.bias, 0)
                self.fc_s.append(fc)

        pass

    def enable_RPP(self):
        self.rpp = True

    def forward(self, x):
        """
        Returns:
          h_s: each member with shape [N, c]
          prediction_s: each member with shape [N, num_classes]
        """
        # Tensor T [N, 2048, 24, 8]
        x = self.base(x)
        f_shape = x.size()

        # g_s [N, 2048, 6, 1]
        if not self.rpp:
            g_s = self.avg_pool(x)
            if self.dropout > 0:
                g_s = self.drop_layer(g_s)
        else:
            f_s = x.view(f_shape[0], f_shape[1], f_shape[2] * f_shape[3])
            f_s = (self.classifier_pool(f_s.permute(0, 2, 1)).permute(0, 2, 1))
            weight_s = self.sampling_weight_layer(f_s).permute(0, 2, 1)
            g_s = torch.matmul(f_s, weight_s).view(f_shape[0], self.f_dimension, self.num_parts, 1)
            pass

        assert g_s.size(2) % self.num_parts == 0

        # h_s [N, 256, 6, 1]
        h_s = self.one_one_conv(g_s)
        prediction_s = []
        for i in range(self.num_parts):
            # 4d vector h -> 2d vector h
            h = h_s[:, :, i, :].squeeze(2)
            prediction_s.append(self.fc_s[i](h))

        x_s = h_s.view(f_shape[0], -1)
        return x_s, prediction_s
