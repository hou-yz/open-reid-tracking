from __future__ import absolute_import

from torch import nn
from torch.nn import init
from .resnet import *
import torchvision


class PCBModel(nn.Module):
    def __init__(self, num_stripes=6, num_features=256, num_classes=0, norm=False, dropout=0, ):
        super(PCBModel, self).__init__()
        self.num_stripes = num_stripes

        # ResNet50: from 384*128*3 -> 24*8*2048 (Tensor T; of column vector f's)
        self.base = nn.Sequential(
            *list(resnet50(pretrained=True, cut_at_pooling=True, norm=norm, dropout=dropout).base.children())[:-2])
        # decrease the downsampling rate
        # change the stride2 conv layer in self.layer4 to stride=1
        old_ds_conv_layer = self.base._modules['7']._modules['0'].conv2
        new_ds_conv_layer = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        new_ds_conv_layer.weight.data = old_ds_conv_layer.weight.data
        if old_ds_conv_layer.bias is not None:
            new_ds_conv_layer.bias.data = old_ds_conv_layer.bias.data
        self.base._modules['7']._modules['0'].conv2 = new_ds_conv_layer
        # change the downsampling layer in self.layer4 to stride=1
        old_ds_conv_layer = list(self.base._modules['7']._modules['0'].downsample)[0]
        new_ds_conv_layer = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        new_ds_conv_layer.weight.data = old_ds_conv_layer.weight.data
        if old_ds_conv_layer.bias is not None:
            new_ds_conv_layer.bias.data = old_ds_conv_layer.bias.data
        self.base._modules['7']._modules['0'].downsample = nn.Sequential(
            new_ds_conv_layer,
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        # Average Pooling: 24*8*2048 -> 6*1*2048 (f -> g)
        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 8), stride=(4, 8))

        # 1*1 Conv: 6*1*2048 -> 6*1*num_features (g -> h)
        self.one_one_conv = nn.Sequential(nn.Conv2d(2048, num_features, 1), nn.BatchNorm2d(num_features),
                                          nn.ReLU(inplace=True))

        # 6 branches of fc's:
        if num_classes > 0:
            self.fc_list = nn.ModuleList()
            for _ in range(num_stripes):
                fc = nn.Linear(num_features, num_classes)
                init.normal(fc.weight, std=0.001)
                init.constant(fc.bias, 0)
                self.fc_list.append(fc)

    def forward(self, x):
        """
        Returns:
          h_s: each member with shape [N, c]
          prediction_s: each member with shape [N, num_classes]
        """
        # Tensor T [N, 2048, 24, 8]
        x = self.base(x)
        # g [N, 2048, 6, 1]
        x = self.avg_pool(x)
        # h_s [N, 256, 6, 1]
        h_s_four_d = self.one_one_conv(x)

        assert h_s_four_d.size(2) % self.num_stripes == 0

        h_s = []
        prediction_s = []
        for i in range(self.num_stripes):
            # h [N, 256, 1, 1]
            h = h_s_four_d[:, :, i, :]
            # 4d vector h -> 2d vector h
            h = h.view(h.size(0), -1)
            h_s.append(h)
            if hasattr(self, 'fc_list'):
                prediction_s.append(self.fc_list[i](h))

        if hasattr(self, 'fc_list'):
            return h_s, prediction_s

        return h_s
