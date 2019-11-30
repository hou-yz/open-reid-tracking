from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torchvision.models import resnet50


class PCB_model(nn.Module):
    def __init__(self, num_stripes=6, feature_dim=256, num_classes=0, norm=False, dropout=0, last_stride=1, ):
        super(PCB_model, self).__init__()
        # Create PCB_only model
        self.num_stripes = num_stripes
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.norm = norm

        self.base = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        base_dim = 2048

        if last_stride != 2:
            # decrease the downsampling rate
            # change the stride2 conv layer in self.layer4 to stride=1
            self.base[7][0].conv2.stride = last_stride
            # change the downsampling layer in self.layer4 to stride=1
            self.base[7][0].downsample[0].stride = last_stride

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 1))

        # feat & feat_bn
        if self.feature_dim > 0:
            self.feat_fc = nn.Conv2d(base_dim, feature_dim, kernel_size=1, padding=0)
            init.kaiming_normal_(self.feat_fc.weight, mode='fan_out')
            init.constant_(self.feat_fc.bias, 0.0)
        else:
            feature_dim = base_dim

        self.feat_bn = nn.BatchNorm2d(feature_dim)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        # dropout before classifier
        self.dropout = dropout
        if self.dropout > 0:
            self.drop_layer = nn.Dropout2d(self.dropout)

        # 6 branches of classifiers:
        if self.num_classes > 0:
            self.classifier_s = nn.ModuleList()
            for _ in range(self.num_stripes):
                classifier = nn.Linear(feature_dim, self.num_classes, bias=False)
                init.normal_(classifier.weight, std=0.001)
                self.classifier_s.append(classifier)

    def forward(self, x):
        """
        Returns:
          h_s: each member with shape [N, c]
          prediction_s: each member with shape [N, num_classes]
        """
        x = self.base(x)
        x = self.avg_pool(x)

        if self.dropout > 0:
            x = self.drop_layer(x)

        feature_base = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
        feature_base = feature_base.view(feature_base.shape[0], -1)

        if self.feature_dim > 0:
            x = self.feat_fc(x)
            feature_pcb = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
            feature_pcb = feature_pcb.view(feature_pcb.shape[0], -1)
        else:
            feature_pcb = feature_base

        x = self.feat_bn(x)
        # no relu after feature_fc


        x_s = x.chunk(self.num_stripes, 2)
        prediction_s = []
        if self.num_classes > 0 and self.training:
            for i in range(self.num_stripes):
                prediction_s.append(self.classifier_s[i](x_s[i].view(x.shape[0], -1)))

        if self.norm:
            feature_pcb = F.normalize(feature_pcb)

        return feature_pcb, tuple(prediction_s)
