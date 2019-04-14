from __future__ import absolute_import

from .triplet import TripletLoss
from .label_smooth import LSR_loss

__all__ = [
    'TripletLoss',
    'LSR_loss'
]
