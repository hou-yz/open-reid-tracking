from __future__ import absolute_import
from collections import OrderedDict
import torch
import numpy as np

from ..models import PCB_model, IDE_model
from torch.autograd import Variable

from ..utils import to_torch


def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs, volatile=True)
    if modules is None:
        outputs = model(inputs)
        if isinstance(model.module, PCB_model):
            # set the feature as 6 h's, which has a total dimension of 6*256=1536
            outputs = outputs[0]
        elif isinstance(model.module, IDE_model):
            # set the feature as 1 h, which has a total dimension of 1*256=256
            outputs = outputs[1]
        outputs = outputs.data.cpu()
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None

        def func(m, i, o): outputs[id(m)] = o.data.cpu()

        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())
