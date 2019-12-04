import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


class MLP_metric(nn.Module):
    def __init__(self, feature_dim=256, num_class=0):
        super(MLP_metric, self).__init__()
        self.num_class = num_class
        layer_dim = 128

        self.fc1 = nn.Sequential(nn.Linear(feature_dim, layer_dim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(layer_dim, layer_dim), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(layer_dim, layer_dim), nn.ReLU())
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(layer_dim, self.num_class, bias=False)
        init.normal_(self.classifier.weight, std=0.001)

    def forward(self, feat1, feat2):
        out = self.fc1((feat2 - feat1).abs())
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.classifier(out)
        return out
