# Copyright (c) 2018, Zhewei Xu
# [xzhewei-at-gmail.com]
# Licensed under The MIT License [see LICENSE for details]

import math

def bbox_filter(bboxs, filter, param):
    """
    Filter bbox by param.

        Param is a dictionary containing some filter conditions. The conditions please see get_default_filter().

        bbox which not in param['lbls'] or param['ilbls'] would excluded.
        bbox wihch dose not meet other param, bbox['ignore'] would set True.

        USEAGE
            bbox_filter(bboxs,param)

        INPUT
            param       - a dict created by get_default_filter()
            bboxs       - a list contains bbox annotations

        OUPUT
            bbox_filted - a list contains filted bbox

        EXAMPLE
            import pydatatool as pdt
            vbbs = pdt.caltech.load_vbbs('/home/all/datasets/caltech/annotations')
            bboxs = vbbs['set00']['V000']['objLists']
            param=get_default_filter()
            param['lbls']=['person']
            param['ilbls']=['people']
            param['squarify']=[3,0.41]
            param['hRng']=[50,float('inf')]
            param['vRng']=[1,1]
            bboxs_filted = bbox_filter(bboxs,param)

        See also get_default_filter(), bbox_squarify(), bbox_resize()
    """
    # Tracer()()
    n = len(bboxs)
    bbox_filted = []

    if len(param['lbls']) != 0:
        lbl = set(param['lbls']) | (set(param['ilbls']))
        for i in range(n):
            if bboxs[i]['lbl'] in lbl:
                bbox_filted.append(bboxs[i])

    for i in range(len(bbox_filted)):
        bbox_filted[i]['ignore'] = filter(bbox_filted[i], param)

        if (bbox_filted[i]['ignore'] == 0) and (len(param['squarify']) != 0):
            bbox_filted[i]['pos'] = bbox_squarify(bbox_filted[i]['pos'], param['squarify'][0], param['squarify'][1])
    return bbox_filted

def bbox_squarify(bb, flag, ar=1):
    """
    Fix bb aspect ratios (without moving the bb centers).
    Reimplimentation of bbGt('squarify) from Piotr's CV matlab toolbox.

        The w or h of each bb is adjusted so that w/h=ar.
        The parameter flag controls whether w or h should change:
        flag==0: expand bb to given ar
        flag==1: shrink bb to given ar
        flag==2: use original w, alter h
        flag==3: use original h, alter w
        flag==4: preserve area, alter w and h
        If ar==1 (the default), always converts bb to a square, hence the name.

        USAGE
        bbr = squarify(bb, flag, [ar])

        INPUTS
        bb     - [nx4] original bbs
        flag   - controls whether w or h should change
        ar     - [1] desired aspect ratio

        OUTPUT
        bbr    - the output 'squarified' bbs

        EXAMPLE
        bbr = bbox_squarify([0 0 1 2],0)
    """

    bbr = list(bb)
    if flag == 4:
        bbr = bbox_resize(bb, 0, 0, ar)
        return bbr
    usew = (flag == 0 and (bb[2] > bb[3] * ar)) or (flag == 1 and (bb[2] < bb[3] * ar)) or flag == 2
    if usew:
        bbr = bbox_resize(bb, 0, 1, ar)
    else:
        bbr = bbox_resize(bb, 1, 0, ar)

    return bbr

def bbox_resize(bb, hr, wr, ar=0):
    """
    Resize the bbs (without moving their centers).

        If wr>0 or hr>0, the w/h of each bb is adjusted in the following order:
            if hr!=0: h=h*hr
            if wr1=0: w=w*wr
            if hr==0: h=w/ar
            if wr==0: w=h*ar
        Only one of hr/wr may be set to 0, and then only if ar>0. If, however,
        hr=wr=0 and ar>0 then resizes bbs such that areas and centers are
        preserved but aspect ratio becomes ar.

        USAGE
            bb = ( resize, bb, hr, wr, [ar] )

        INPUT
            bb     - [nx4] original bbs
            hr     - ratio by which to multiply height (or 0)
            wr     - ratio by which to multiply width (or 0)
            ar     - [0] target aspect ratio (used only if hr=0 or wr=0)

        OUTPUT
            bb    - [nx4] the output resized bbs

        EXAMPLE
            bb = bbox_resize([0 0 1 1],1.2,0,.5) % h'=1.2*h; w'=h'/2;
    """
    # Tracer()()
    assert len(bb) == 4
    assert (hr > 0 and wr > 0) or ar > 0
    if hr == 0 and wr == 0:
        a = math.sqrt(bb[2] * bb[3])
        ar = math.sqrt(ar)
        d = a * ar - bb[2];
        bb[0] = bb[0] - d / 2;
        bb[2] = bb[2] + d
        d = a * ar - bb[3];
        bb[1] = bb[1] - d / 2;
        bb[3] = bb[3] + d
        return bb
    if hr != 0:
        d = (hr - 1) * bb[3];
        bb[1] = bb[1] - d / 2;
        bb[3] = bb[3] + d
    if wr != 0:
        d = (hr - 1) * bb[2];
        bb[0] = bb[0] - d / 2;
        bb[2] = bb[2] + d
    if hr == 0:
        d = bb[2] / ar - bb[3];
        bb[1] = bb[1] - d / 2;
        bb[3] = bb[3] + d
    if wr == 0:
        d = bb[3] * ar - bb[2];
        bb[0] = bb[0] - d / 2;
        bb[2] = bb[2] + d

    return bb

# TODO
# move same function to a common file.