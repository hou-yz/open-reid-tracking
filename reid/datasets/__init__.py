from __future__ import absolute_import
import warnings

from .dukemtmc import DukeMTMC
from .ai_city import AI_City
from .market1501 import Market1501
from .vehicleid import vehicleID
from .veri import VeRi
from .veri_vehicleid import veri_vehicleID

__factory = {
    'market1501': Market1501,
    'duke_tracking': DukeMTMC,
    'dukemtmc': DukeMTMC,
    'aic_tracking': AI_City,
    'vehicleid': vehicleID,
    'veri': VeRi,
    'veri_vehicleid': veri_vehicleID,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](*args, **kwargs)

