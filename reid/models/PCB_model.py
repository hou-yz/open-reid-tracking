from __future__ import absolute_import

from torch import nn
from torch.nn import init
from .resnet import *
import torchvision


class PCBModel(nn.Module):
    def __init__(self, num_parts=6, num_features=256, num_classes=0, norm=False, dropout=0, ):
        super(PCBModel, self).__init__()
        self.num_parts = num_parts
        self.num_features = num_features

        # ResNet50: from 384*128*3 -> 24*8*2048 (Tensor T; of column vector f's)
        self.base = nn.Sequential(
            *list(resnet50(pretrained=True, cut_at_pooling=True, norm=norm, dropout=dropout).base.children())[:-2])
        # decrease the downsampling rate
        # change the stride2 conv layer in self.layer4 to stride=1
        self.base[7][0].conv2.stride = (1, 1)
        # change the downsampling layer in self.layer4 to stride=1
        self.base[7][0].downsample[0].stride = (1, 1)

        # Average Pooling: 24*8*2048 -> 6*1*2048 (f -> g)
        # Tensor T [N, 2048, 24, 8]
        self.avg_pool = nn.AdaptiveMaxPool2d((6, 1))

        # 1*1 Conv: 6*1*2048 -> 6*1*256 (g -> h)
        # 6 separate convs
        self.one_one_conv_s = nn.ModuleList()
        for _ in range(self.num_parts):
            self.one_one_conv_s.append(nn.Sequential(
                nn.Conv2d(2048, self.num_features, 1),
                nn.BatchNorm2d(self.num_features),
                nn.ReLU(inplace=True)
            ))

        # 6 branches of fc's:
        if num_classes > 0:
            self.fc_s = nn.ModuleList()
            for _ in range(self.num_parts):
                fc = nn.Linear(self.num_features, num_classes)
                init.normal(fc.weight, std=0.001)
                init.constant(fc.bias, 0)
                self.fc_s.append(fc)

        pass

    def forward(self, x):
        """
        Returns:
          h_s: each member with shape [N, c]
          prediction_s: each member with shape [N, num_classes]
        """
        # Tensor T [N, 2048, 24, 8]
        x = self.base(x)
        # g [N, 2048, 6, 1]
        g_s = self.avg_pool(x)

        assert g_s.size(2) % self.num_parts == 0

        h_s = []
        prediction_s = []
        for i in range(self.num_parts):
            g = g_s[:, :, i, :].unsqueeze(2)
            # h [N, 256, 1, 1]
            h = self.one_one_conv_s[i](g)
            # 4d vector h -> 2d vector h
            h = h.view(h.size(0), -1)
            h_s.append(h)
            if hasattr(self, 'fc_s'):
                prediction_s.append(self.fc_s[i](h))

        if hasattr(self, 'fc_s'):
            return h_s, prediction_s

        return h_s
