from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from .resnet import *
import torchvision


class PCB_model(nn.Module):
    def __init__(self, num_parts=6, num_features=256, num_classes=0, norm=False, dropout=0, last_stride=2,
                 reduced_dim=256, output_feature=None):
        super(PCB_model, self).__init__()
        # Create PCB_only model
        self.num_parts = num_parts
        self.num_features = num_features
        self.num_classes = num_classes
        self.rpp = False
        self.reduced_dim = reduced_dim
        self.output_feature = output_feature

        # ResNet50: from 3*384*128 -> 2048*24*8 (Tensor T; of column vector f's)
        self.base = nn.Sequential(
            *list(resnet50(pretrained=True, cut_at_pooling=True, norm=norm, dropout=dropout).base.children())[:-2])
        # decrease the downsampling rate
        if last_stride != 2:
            # decrease the downsampling rate
            # change the stride2 conv layer in self.layer4 to stride=1
            self.base[7][0].conv2.stride = last_stride
            # change the downsampling layer in self.layer4 to stride=1
            self.base[7][0].downsample[0].stride = last_stride

        '''Average Pooling: 256*24*8 -> 256*6*1 (f -> g)'''
        # Tensor T [N, 256, 24, 8]
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 1))

        # dropout after pool5 (or what left of it) at p=0.5
        self.dropout = dropout
        if self.dropout > 0:
            self.drop_layer = nn.Dropout2d(self.dropout)

        '''channel reduce: 2048*24*8 -> 256*24*8'''
        # avg pooling along the channel dimension for a [256*1*1]
        self.RPP_pool_channel_reduce = nn.AdaptiveAvgPool1d(self.reduced_dim)

        #############################################  PCB  ############################################################
        '''feat & feat_bn'''
        if self.num_features > 0:
            # 1*1 Conv: 6*1*2048 -> 6*1*256 (g -> h)
            self.PCB_conv_channel_reduce = nn.Sequential(nn.Conv2d(2048, self.num_features, kernel_size=1),
                                                         nn.BatchNorm2d(self.num_features),
                                                         nn.ReLU())
            init.kaiming_normal_(self.PCB_conv_channel_reduce[0].weight, mode='fan_out')
            init.constant_(self.PCB_conv_channel_reduce[0].bias, 0)
            init.constant_(self.PCB_conv_channel_reduce[1].weight, 1)
            init.constant_(self.PCB_conv_channel_reduce[1].bias, 0)
        #############################################  RPP  ############################################################

        '''RPP: Refined part pooling'''
        # first, extract each f's; then,
        # get sampling weights from each f's [2048*1*1]
        # avg pooling along the channel dimension for a [256*1*1]
        self.classifier_pool = nn.AdaptiveAvgPool1d(self.num_features)
        # 6 classifier for f:[256*1*1] -> weight_s:[6*1*1]
        self.sampling_weight_layer = nn.Sequential(nn.Conv1d(self.num_features, self.num_parts, kernel_size=1),
                                                   nn.Softmax(dim=1))
        init.kaiming_normal_(self.sampling_weight_layer[0].weight, mode='fan_out')
        init.constant_(self.sampling_weight_layer[0].bias, 0)  # return a [N,6,24,8] tensor

        ################################################################################################################

        # 6 branches of fc's:
        if self.num_classes > 0:
            self.fc_s = nn.ModuleList()
            for _ in range(self.num_parts):
                fc = nn.Linear(self.num_features, self.num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
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
            x = self.avg_pool(x)
            g_s = x
            if self.dropout:
                x = self.drop_layer(x)
        else:
            f_s = x.view(f_shape[0], f_shape[1], f_shape[2] * f_shape[3])
            f_s = (self.classifier_pool(f_s.permute(0, 2, 1)).permute(0, 2, 1))
            weight_s = self.sampling_weight_layer(f_s).permute(0, 2, 1)
            g_s = torch.matmul(f_s, weight_s).view(f_shape[0], self.reduced_dim, self.num_parts, 1)
            pass

        assert g_s.size(2) == self.num_parts

        # h_s [N, 256, 6, 1]
        h_s = self.PCB_conv_channel_reduce[0](x)
        x_s = self.PCB_conv_channel_reduce(x)

        prediction_s = []
        for i in range(self.num_parts):
            # 4d vector h -> 2d vector h
            x = x_s[:, :, i, :].squeeze(2)
            prediction_s.append(self.fc_s[i](x))

        if self.output_feature == 'pool5':
            x_s = g_s.view(f_shape[0], -1) / g_s.norm()
        else:
            x_s = h_s.view(f_shape[0], -1) / h_s.norm()
        return x_s, prediction_s
