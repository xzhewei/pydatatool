# Copyright (c) 2018, Zhewei Xu
# [xzhewei-at-gmail.com]
# Licensed under The MIT License [see LICENSE for details]

from scipy.io import loadmat
import glob
import os

def load_vbb(filename):
    """
    A is a dict load from the caltech vbb file has the same data structure.
        INPUT
            filename: vbb path
        OUTPUT
            vbb: vbb annotation
        EXAMPLE
            import pydatatool as pdt
            vbb = pdt.caltech.load_vbb('/home/all/datasets/caltech/annotations/set00/V000.vbb')
    """
    vbb = loadmat(filename)
    nFrame = int(vbb['A'][0][0][0][0][0])
    objLists = vbb['A'][0][0][1][0]
    maxObj = int(vbb['A'][0][0][2][0][0])
    objInit = vbb['A'][0][0][3][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
    objStr = vbb['A'][0][0][5][0]
    objEnd = vbb['A'][0][0][6][0]
    objHide = vbb['A'][0][0][7][0]
    altered = int(vbb['A'][0][0][8][0][0])
    log = vbb['A'][0][0][9][0]
    logLen = int(vbb['A'][0][0][10][0][0])

    obj_list = dict()
    for frame_id, obj in enumerate(objLists):
        objs = []
        if obj.shape[1] > 0:
            for id, pos, occl, lock, posv in zip(obj['id'][0], obj['pos'][0], obj['occl'][0], obj['lock'][0],
                                                 obj['posv'][0]):
                id = int(id[0][0]) - 1  # matlab is 1-start
                pos = pos[0].tolist()
                occl = int(occl[0][0])
                lock = int(lock[0][0])
                posv = posv[0].tolist()
                keys_obj = ('id', 'pos', 'occl', 'lock', 'posv', 'ignore')
                datum = dict(zip(keys_obj, [id, pos, occl, lock, posv, False]))
                datum['lbl'] = objLbl[id]
                objs.append(datum)
        obj_list[frame_id] = objs

    keys_vbb = (
    'nFrame', 'objLists', 'maxObj', 'objInit', 'objLbl', 'objStr', 'objEnd', 'objHide', 'altered', 'log', 'logLen')
    A = dict(zip(keys_vbb, [nFrame, obj_list, maxObj, objInit, objLbl, objStr, objEnd, objHide, altered, log, logLen]))
    return A

def load_vbbs(ann_dir):
    """
    Read all annotations from dir vbb files, data[set_name][video_name]=A
        INPUT
            ann_dir: caltech annotations dir
        OUTPUT
            vbbs:    all caltech vbb anno, vbbs[set_name][video_name]
        EXAMPLE
            import pydatatool as pdt
            vbbs = pdt.caltech.load_vbbs('/home/all/datasets/caltech/annotations')
    """
    vbbs = dict()
    for dname in sorted(glob.glob(ann_dir+'/set*')):
        set_name = os.path.basename(dname)
        vbbs[set_name] = dict()
        for anno_fn in sorted(glob.glob('{}/*.vbb'.format(dname))):
            vid_anno = load_vbb(anno_fn)
            video_name = os.path.splitext(os.path.basename(anno_fn))[0]
            vbbs[set_name][video_name] = vid_anno
    return vbbs

# TODO
# move same function to a common file.