from __future__ import print_function, absolute_import
import time

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable

from .models import PCB_model, IDE_model
from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=10):
        self.model.train()

        # set the bn layers to eval() and don't change weight & bias
        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if m.affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if isinstance(self.model.module, PCB_model) or isinstance(self.model.module, IDE_model):
            h_s = outputs[0]
            prediction_s = outputs[1]
            loss = 0
            for pred in prediction_s:
                loss += self.criterion(pred, targets)
            if isinstance(self.model.module, PCB_model):
                prediction = prediction_s[2]
            else:
                prediction = prediction_s[0]
            # use the sum of 6 id-predictions as the input for accuracy(_, _)
            # prediction_sum = Variable(
            #     torch.from_numpy(np.sum(prediction_s[i].cpu().data.numpy() for i in range(len(prediction_s)))).cuda())
            prec, = accuracy(prediction.data, targets.data)
            prec = prec[0]
            pass

        else:
            if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                loss = self.criterion(outputs, targets)
                prec, = accuracy(outputs.data, targets.data)
                prec = prec[0]
                pass
            elif isinstance(self.criterion, OIMLoss):
                loss, outputs = self.criterion(outputs, targets)
                prec, = accuracy(outputs.data, targets.data)
                prec = prec[0]
            elif isinstance(self.criterion, TripletLoss):
                loss, prec = self.criterion(outputs, targets)
            else:
                raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec
