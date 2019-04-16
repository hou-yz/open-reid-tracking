from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn import functional as F
# from .resnet import *
from torchvision.models import resnet50, densenet121


class IDE_model(nn.Module):
    def __init__(self, num_features=256, num_classes=0, norm=False, dropout=0, last_stride=2, output_feature='fc',
                 arch='resnet50'):
        super(IDE_model, self).__init__()
        # Create IDE_only model
        self.num_features = num_features
        self.num_classes = num_classes
        self.output_feature = output_feature
        self.norm = norm

        if arch == 'resnet50':
            # ResNet50: from 3*384*128 -> 2048*12*4 (Tensor T; of column vector f's)
            self.base = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
            if last_stride != 2:
                # decrease the downsampling rate
                # change the stride2 conv layer in self.layer4 to stride=1
                self.base[7][0].conv2.stride = last_stride
                # change the downsampling layer in self.layer4 to stride=1
                self.base[7][0].downsample[0].stride = last_stride
            base_channel = 2048
        elif arch == 'densenet121':
            self.base = nn.Sequential(*list(densenet121(pretrained=True).children())[:-1])[0]
            if last_stride != 2:
                # remove the pooling layer in last transition block
                self.base[-3][-1].stride = 1
                self.base[-3][-1].kernel_size = 1
                pass
            base_channel = 1024

        ################################################################################################################
        '''Global Average Pooling: 2048*12*4 -> 2048*1*1'''
        # Tensor T [N, 2048, 1, 1]
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # dropout after pool5 (or what left of it) at p=0.5
        self.dropout = dropout
        if self.dropout > 0:
            self.drop_layer = nn.Dropout2d(self.dropout)

        ################################################################################################################
        '''feat & feat_bn'''
        if self.num_features > 0:
            # 1*1 Conv(fc): 1*1*2048 -> 1*1*256 (g -> h)
            self.one_one_conv = nn.Sequential(nn.Conv2d(base_channel, self.num_features, 1),
                                              nn.BatchNorm2d(self.num_features),
                                              nn.ReLU())
            init.kaiming_normal_(self.one_one_conv[0].weight, mode='fan_out')
            init.constant_(self.one_one_conv[0].bias, 0)
            init.constant_(self.one_one_conv[1].weight, 1)
            init.constant_(self.one_one_conv[1].bias, 0)
        # else:
        #     # 128 dim pooling for triplet
        #     self.classifier_pool = nn.AdaptiveAvgPool1d(128)

        # fc for softmax:
        if self.num_classes > 0:
            self.fc = nn.Linear(self.num_features, self.num_classes)
            init.normal_(self.fc.weight, std=0.001)
            init.constant_(self.fc.bias, 0)

        pass

    def forward(self, x, eval_only=False):
        """
        Returns:
          h_s: each member with shape [N, c]
          prediction_s: each member with shape [N, num_classes]
        """
        # Tensor T [N, 2048, 12, 4]
        x = self.base(x)
        x = self.global_avg_pool(x)

        out0 = x.view(x.shape[0], -1)

        if self.dropout > 0:
            x = self.drop_layer(x)

        if self.num_features > 0:
            x = self.one_one_conv(x).view(x.shape[0], -1)
        out1 = x.view(x.shape[0], -1)

        prediction_s = []
        if self.num_classes > 0 and not eval_only:
            prediction = self.fc(x)
            prediction_s.append(prediction)

        if self.norm:
            out0 = F.normalize(out0)
            out1 = F.normalize(out1)

        if self.output_feature == 'pool5':
            return out0, tuple(prediction_s)
        else:
            return out1, tuple(prediction_s)
