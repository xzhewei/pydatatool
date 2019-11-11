from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append('utils')
import json
import numpy as np
from .utils.box import *
from .utils.draw import *
from .utils.infrastructure import *
from .utils.detbox import *
def save_results(records,fpath):

    with open(fpath,'w') as fid:
        for record in records:
            line = json.dumps(record)+'\n'
            fid.write(line)
    return fpath

def load_func(fpath):

    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records

def clip_boundary(dtboxes,height,width):

    num = dtboxes.shape[0]
    dtboxes[:,0] = np.maximum(dtboxes[:,0], 0)
    dtboxes[:,1] = np.maximum(dtboxes[:,1], 0)
    dtboxes[:,2] = np.minimum(dtboxes[:,2], width)
    dtboxes[:,3] = np.minimum(dtboxes[:,3], height)
    return dtboxes


def recover_func(dtboxes):

    assert dtboxes.shape[1]>=4
    dtboxes[:,2] += dtboxes[:,0]
    dtboxes[:,3] += dtboxes[:,1]
    return dtboxes
