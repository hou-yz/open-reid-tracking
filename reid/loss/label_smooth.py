from __future__ import print_function, absolute_import

import torch
from torch import nn
from torch.autograd import Variable


class LSR_loss(nn.Module):
    def __init__(self, smooth=0.1):
        super(LSR_loss, self).__init__()
        self.smooth = smooth

    def _class_to_one_hot(seldf, targets, num_class, smooth):
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.FloatTensor(targets.size()[0], num_class)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets, (1 - smooth))
        targets_onehot.add_(smooth / num_class)
        return targets_onehot

    def forward(self, outputs, targets):
        num_class = outputs.size()[1]
        targets = self._class_to_one_hot(targets.data.cpu(), num_class, self.smooth)
        targets = Variable(targets.cuda())
        outputs = torch.nn.LogSoftmax(dim=1)(outputs)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss
