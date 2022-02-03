'''
Author: Liweileev
Date: 2022-01-04 23:31:56
LastEditors: Liweileev
LastEditTime: 2022-01-31 01:48:36
'''

import click
import pickle
import re
import copy
import numpy as np
import torch
import dnnlib
from torch_utils import misc

#----------------------------------------------------------------------------

def load_network_pkl(f):
    data = _LegacyUnpickler(f).load()

    # Add missing fields.
    if 'training_set_kwargs' not in data:
        data['training_set_kwargs'] = None
        
    # Validate contents.
    assert isinstance(data['G'], torch.nn.Module)
    assert isinstance(data['D0'], torch.nn.Module)
    assert isinstance(data['G_ema'], torch.nn.Module)
    assert isinstance(data['training_set_kwargs'], (dict, type(None)))

    return data

#----------------------------------------------------------------------------

class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module, name)

#----------------------------------------------------------------------------
