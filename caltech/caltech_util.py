#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# caltech_utils
# Copyright (c) 2017 Zhewei Xu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from scipy.io import loadmat
from collections import defaultdict
import json
import glob
import os
import math
import time

IPDB = False
if IPDB:
    from ipdb import set_trace

def load_vbb(filename):
    '''
    A is a dict load from the caltech vbb file has the same data structure.
        INPUT
            filename: vbb path
        OUTPUT
            vbb: vbb annotation
        EXAMPLE
            import caltech_utils
            vbb = caltech_utils.load_vbb('/home/all/datasets/caltech/annotations/set00/V000.vbb')
    '''
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
        if len(obj) > 0:
            for id, pos, occl, lock, posv in zip(obj['id'][0], obj['pos'][0], obj['occl'][0], obj['lock'][0], obj['posv'][0]):
                id = int(id[0][0])-1 # matlab is 1-start
                pos = pos[0].tolist()
                occl = int(occl[0][0])
                lock = int(lock[0][0])
                posv = posv[0].tolist()
                keys_obj = ('id','pos','occl','lock','posv','ignore')
                datum = dict(zip(keys_obj, [id, pos, occl, lock, posv, False]))
                datum['lbl'] = objLbl[id]
                objs.append(datum)
        obj_list[frame_id] = objs
    
    keys_vbb = ('nFrame','objLists','maxObj','objInit','objLbl','objStr','objEnd','objHide','altered','log','logLen')
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
            import caltech_utils
            vbbs = caltech_utils.load_vbbs('/home/all/datasets/caltech/annotations')
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

def get_image_identifiers(imageSets_file):
    """
    Get image identifiers from ImageSets dir's file for train or test. like ['set00_V000_I00000','set00_V000_I00001'...]
    """
    with open(imageSets_file, 'r') as f:
        lines = f.readlines()
    image_identifiers = [x.strip() for x in lines]
    return image_identifiers

def save_image_set(fname, image_ids):
    """
    Save image_ids to a txt file as image set.

        INPUT
            fname: The imageset filename
            image_ids: a list contaions image info dicts.
        
        EXAMPLE
            import caltech_utils
            vbbs = caltech_utils.load_vbbs('/home/all/datasets/caltech/annotations')
            image_ids = caltech_utils.get_image_ids('caltech_train',vbbs,30)
            caltech_utils.save_image_set('./ImageSets/caltech_train_1x.txt', image_ids)
    """
    path, filename = os.path.split(fname)
    if not os.path.exists(path):
        os.mkdir(path)
    f = open(fname,'w')
    for img in image_ids:
        f.write("{}\n".format(img['file_name']))

def bbox_filter(bboxs,param):
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
            import caltech_utils
            vbbs = caltech_utils.load_vbbs('/home/all/datasets/caltech/annotations')
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
        lbl = set(param['lbls'])|(set(param['ilbls'])) 
        for i in range(n):            
            if bboxs[i]['lbl'] in lbl:
                bbox_filted.append(bboxs[i])
    
    for i in range(len(bbox_filted)):
        # Tracer()()
        bbox_filted[i]['ignore'] = filter(bbox_filted[i],param)
        
        if (bbox_filted[i]['ignore'] == 0) and (len(param['squarify']) != 0):
            bbox_filted[i]['pos'] = bbox_squarify(bbox_filted[i]['pos'],param['squarify'][0],param['squarify'][1])
        # Tracer()()
    # Tracer()()
    return bbox_filted

def filter(obj,param={}):
    """
    Caltech Ground Truth filter, like toolbox-detector-bbGt-bbLoad.
        INPUT
            obj: an annotation in vbb objLists.
            param: filter parameter
        OUTPUT
            flag: should be ignore or not
        EXAMPLE
            import caltech_utils
            vbbs = caltech_utils.load_vbbs('/home/all/datasets/caltech/annotations')
            bbox = vbbs['set00']['V000']['objLists'][0]
            param=get_default_filter()
            param['lbls']=['person']
            param['ilbls']=['people']
            param['squarify']=[3,0.41]
            param['hRng']=[50,float('inf')]
            param['vRng']=[1,1]
            filter(bbox_filted[i],param)
    """
    flag = False

    if len(param)==0:
        param=get_default_filter()

    if len(param['ilbls']) != 0:
        flag = flag or (obj['lbl'] in param['ilbls'])
    if len(param['xRng']) != 0:
        v = obj['pos'][0]
        flag = flag or v < param['xRng'][0] or v > param['xRng'][1]
        v =obj['pos'][0] + obj['pos'][2]
        flag = flag or v < param['xRng'][0] or v > param['xRng'][1]
    if len(param['yRng']) != 0:
        v = obj['pos'][1]
        flag = flag or v < param['yRng'][0] or v > param['yRng'][1]
        v = obj['pos'][1] + obj['pos'][3]
        flag = flag or v < param['yRng'][0] or v > param['yRng'][1]
    if len(param['wRng']) != 0:
        v = obj['pos'][2]
        flag = flag or v < param['wRng'][0] or v > param['wRng'][1]
    if len(param['hRng']) != 0:
        v = obj['pos'][3]
        flag = flag or v < param['hRng'][0] or v > param['hRng'][1]
    if len(param['aRng']) != 0:
        v = obj['pos'][2] * obj['pos'][3]
        flag = flag or v < param['aRng'][0] or v > param['aRng'][1]
    if len(param['arRng']) != 0:
        v = obj['pos'][2] / obj['pos'][3]
        flag = flag or v < param['arRng'][0] or v > param['arRng'][1]
    if len(param['vRng']) != 0:
        pos  = obj['pos']
        posv = obj['posv']
        if obj['occl']==0 or set(posv)=={0}:
            v = 1
        elif posv==pos:
            v = 0
        else:
            if IPDB: set_trace()
            v = (posv[2]*posv[3])/(pos[2]*pos[3])
        flag = flag or v < param['vRng'][0] or v > param['vRng'][1]
    if len(param['occl']) != 0:
        '''occl= 0|1|2 represent for no occl | occl | no and occl'''
        if param['occl'] != 2:
            flag = flag or (obj['occl'] != param['occl'])
    
    return flag

def bbox_squarify(bb,flag,ar=1):
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

    # Tracer()()
    bbr=list(bb)
    if flag == 4:
        bbr = bbox_resize(bb,0,0,ar)
        return bbr
    usew = (flag == 0 and (bb[2]>bb[3]*ar)) or (flag == 1 and (bb[2]<bb[3]*ar)) or flag == 2
    if usew:
        bbr = bbox_resize(bb,0,1,ar)
    else:
        bbr = bbox_resize(bb,1,0,ar)
        
    return bbr

def bbox_resize(bb,hr,wr,ar=0):
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
    assert len(bb)==4
    assert (hr>0 and wr>0) or ar>0
    if hr==0 and wr==0:
        a = math.sqrt(bb[2]*bb[3])
        ar= math.sqrt(ar)
        d = a*ar - bb[2]; bb[0]=bb[0]-d/2; bb[2]=bb[2]+d
        d = a*ar - bb[3]; bb[1]=bb[1]-d/2; bb[3]=bb[3]+d
        return bb
    if hr!=0:
        d=(hr-1)*bb[3]; bb[1]=bb[1]-d/2; bb[3]=bb[3]+d
    if wr!=0:
        d=(hr-1)*bb[2]; bb[0]=bb[0]-d/2; bb[2]=bb[2]+d
    if hr==0:
        d=bb[2]/ar-bb[3]; bb[1]=bb[1]-d/2; bb[3]=bb[3]+d
    if wr==0:
        d=bb[3]*ar-bb[2]; bb[0]=bb[0]-d/2; bb[2]=bb[2]+d
    
    return bb

def get_default_filter():
    """
    Defalut filter parameter.
    There type annotation always be set as ignore:
        1. 
    """
    df = dict()
    df['format']  = 0
    df['ellipse'] = 1
    df['squarify']= []
    df['lbls']    = ['person']
    df['ilbls']   = ['people','person-fa','person?']
    df['hRng']    = [20,float('inf')]
    df['wRng']    = []
    df['aRng']    = []
    df['arRng']   = []
    df['oRng']    = []
    df['xRng']    = []
    df['yRng']    = []
    df['vRng']    = [0.2,1]
    df['occl']    = []
    df['squarify'] = []
    
    return df

def get_categories():
    cat = [{'id':1, 'name':'person'},
           {'id':2, 'name':'people'},
           {'id':3, 'name':'person-fa'},
           {'id':4, 'name':'person?'}]
    return cat

def get_category_id(name):
    id = -1
    cats = get_categories()
    for cat in cats:
        if cat['name'] == name:
            id = cat['id']
    return id

def get_category_name(id):
    name=''
    cats = get_categories()
    for cat in cats:
        if cat['id'] == id:
            name = cat['id']
    return name

def get_dbInfo(dbName):
    db={}
    db['caltech']={
        'setIds':range(0,11), #set00-set10
        'vidIds':[range(0,15), #V000-V014
                  range(0,6),  #V000-V005
                  range(0,12), #V000-V011
                  range(0,13), #V000-V012
                  range(0,12), #V000-V011
                  range(0,13), #V000-V012
                  range(0,19), #V000-V018
                  range(0,12), #V000-V011
                  range(0,11), #V000-V010
                  range(0,12), #V000-V011
                  range(0,12)], #V000-V011
        'skip':30,
        'ext':'jpg'
    }

    db['caltech_train']={
        'setIds':range(0,6), #set00-set10
        'vidIds':[range(0,15), #V000-V014
                  range(0,6),  #V000-V005
                  range(0,12), #V000-V011
                  range(0,13), #V000-V012
                  range(0,12), #V000-V011
                  range(0,13), #V000-V012
                  range(0,19), #V000-V018
                  range(0,12), #V000-V011
                  range(0,11), #V000-V010
                  range(0,12), #V000-V011
                  range(0,12)], #V000-V011
        'skip':30,
        'ext':'jpg'
    }

    db['caltech_test']={
        'setIds':range(6,11), #set00-set10
        'vidIds':[range(0,15), #V000-V014
                  range(0,6),  #V000-V005
                  range(0,12), #V000-V011
                  range(0,13), #V000-V012
                  range(0,12), #V000-V011
                  range(0,13), #V000-V012
                  range(0,19), #V000-V018
                  range(0,12), #V000-V011
                  range(0,11), #V000-V010
                  range(0,12), #V000-V011
                  range(0,12)], #V000-V011
        'skip':30,
        'ext':'jpg'
    }

    return db[dbName]

def get_image_ids(dbName,vbbs,skip=1):
    """
    Get a list of image_ids, as [{'id':0,'file_name':'set00_V000_I0000','height':640, 'width':480},...]
    """
    dbInfo = get_dbInfo(dbName)
    image_ids = []
    frames=0
    for s in dbInfo['setIds']:
        for v in dbInfo['vidIds'][s]:
            frames = vbbs['set{:0>2}'.format(s)]['V{:0>3}'.format(v)]['nFrame']
            for i in range(skip-1,frames,skip):
                id = get_image_id(s,v,i)
                file_name = "set{:0>2}_V{:0>3}_I{:0>5}".format(s,v,i)
                image_ids.append({'id':id,'file_name':file_name,'height':640, 'width':480})
    return image_ids

def load_json(filename):
    A = json.open(filename)
    return A

def get_image_id(s,v,i):
    """
    image_id = XX(set)XXX(vid)XXXXX(image)
    """
    return s*(10**8)+v*(10**5)+i

def vbbs2cocos(vbbs,dbName,annId_str=0,objId_str=0):
    """
    Convert caltech a subset, like train_1x or test_1x, to coco style.

        INPUT
            vbbs:        caltech vbb annoations from load_vbbs, vbbs[set_name][vid_name] is a vbb anno.
            dbName:      a caltech subset name, caltech/caltech_train/caltech_test, see get_dbInfo()
            annId_str:   in coco annotations every ann need a unique id
            objId_str:   in caltech the unqiue obj id in vidoes, we set the unique id in whole dataset
        OUPUT
            annotations: coco style annotation
            annId_str:  next convert operate start ann id
            objId_str:  next convert operate start obj id
        EXAMPLE
            import caltech_utils
            vbbs = caltech_utils.load_vbbs('/home/all/datasets/caltech/annotations')
            # convet train set
            annotations_train, annId_str, objId_str = caltech_utils.vbbs2cocos(vbbs,'caltech_train')
            image_ids_train = caltech_utils.get_image_ids('caltech_train',vbbs)
            caltech_utils.save_json(annotations_train,image_ids_train,'caltech_train.json')
            # convet test set
            annotations_test, annId_str, objId_str = caltech_utils.vbbs2cocos(vbbs,'caltech_test',annId_str, objId_str)
            image_ids_test = caltech_utils.get_image_ids('caltech_test',vbbs)
            caltech_utils.save_json(annotations_test,image_ids_test,'caltech_test.json')
    """
    annotations = []

    dbInfo = get_dbInfo(dbName)
    for s in dbInfo['setIds']:
        set_name = 'set{:0>2}'.format(s)
        for v in dbInfo['vidIds'][s]:
            vid_name = 'V{:0>3}'.format(v)
            anns, annId_str, objId_str = vbb2coco(s,v,vbbs[set_name][vid_name],annId_str,objId_str)
            annotations.extend(anns)
    
    return annotations, annId_str, objId_str

def vbb2coco(setId,vidId,vbb,annId_str=0,objId_str=0):
    """
    Convert vbb struct to coco style.

        INPUT
            setId:      the vbb file set id
            vidId:      the vbb file video id
            vbb:        the vbb data from vbb_load
            annId_str:  in coco annotations every ann need a unique id
            objId_str:  in caltech the unqiue obj id in vidoes, we set the unique id in whole dataset
        
        OUTPUT
            anns:       a list contains ann dicts
            annId_str:  next convert operate start ann id
            objId_str:  next convert operate start obj id
        
        EXAMPLE
            import caltech_utils
            vbbs = caltech_utils.load_vbbs('/home/all/datasets/caltech/annotations')
            anns, annId_str, objId_str = caltech_utils.vbb2coco(1,1,vbbs['set01']['V001'])
    """
    anns = []
    for i in range(0,vbb['nFrame']):

        objs = vbb['objLists'][i]
        if len(objs) > 0:
            for obj in objs:
                ann={}
                ann['id']=annId_str
                ann['obj_id']=objId_str+obj['id']
                ann['image_id']=get_image_id(setId,vidId,i)

                ann['category_name']=vbb['objLbl'][obj['id']]
                ann['category_id']=get_category_id(vbb['objLbl'][obj['id']])

                ann['bbox']=obj['pos']
                ann['bbox_v']=obj['posv']
                ann['ignore']=filter(obj)
                ann['iscrowd']=ann['ignore']
                ann['occl']=obj['occl']
                annId_str=annId_str+1
                anns.append(ann)

    return anns, annId_str, objId_str + vbb['maxObj']

def save_json(annotations,image_ids,json_file):
    """
    Save annotations to a json file, as coco style.

    INPUT
        annotations: a list contaions annotation dicts
        image_ids: a list contaions image dicts
        json_file: the file to save

    EXAMPLE
        import caltech_utils
        vbbs = caltech_utils.load_vbbs('/home/all/datasets/caltech/annotations')
        annotations, annId_str, objId_str = caltech_utils.vbbs2cocos(vbbs,'caltech')
        image_ids = caltech_utils.get_image_ids('caltech',vbbs)
        caltech_utils.save_json(annotations,image_ids,'caltech.json')
    """
    json_data = {'info':{}, 'images':[], 'annotations':[], 'categories':[]}
    json_file = open(json_file,'w')

    json_data['images'] = image_ids

    json_data['categories'] = get_categories()

    json_data['annotations'] = annotations

    json_data['info']={'description': 'This is a json version label of the Caltech Pedestrian dataset, converted from vbb version.',
            'version': '1.1',
            'vbb_version': '1.4',
            'create_time': '2018-04-03'}

    json.dump(json_data, json_file)
    json_file.close()

def save_pkl(data,filename):
    """
    Save a data to a .pkl file
    """
    import cPickle
    with open(filename,'w') as f:
        cPickle.dump(data,f)

def load_pkl(filename):
    """
    Load a data from a .pkl file.
    """
    import cPickle
    with open(filename,'r') as f:
        data = cPickle.load(f)
    return data

def seqs2imgs(dbName,sdir,tdir,skip=1):
    """
    Convert caltech seq set to images, like the dbExtract.m

    INPUT
        dbName: a string defind in get_dbInfo
        sdir: a directory contains the videos, e.g. the videos dir in caltech dataset.
        tdir: the ouput images dir
        skip: interval of frames

    EXAMPLE
        seqs2imgs('caltech','/home/all/datasets/caltech/videos','./extract',30)
    """
    dbInfo = get_dbInfo(dbName)
    if not os.path.exists(sdir):
        print("The videos directory {} is not exits.".format(sdir))
    for s in dbInfo['setIds']:
        set_name = 'set{:0>2}'.format(s)
        for v in dbInfo['vidIds'][s]:
            vid_name = 'V{:0>3}.seq'.format(v)
            vname = os.path.join(sdir,set_name,vid_name)
            print("Extrace {} to imgs.".format(vname))
            tic = time.time()
            seq2imgs(vname,tdir,skip)
            print('Done (t={:0.2f}s)'.format(time.time()-tic))

def seq2imgs(vname,tdir,skip=1):
    """
    Convert a seq file to images.
        INPUT
            vname: the seq file path
            tdir: the ouput images dir
            skip: interval of frames
        EXAMPLE
            caltech_utils.seq2imgs('/home/all/datasets/caltech/videos/set00/V000.seq','./extract',30)
    """
    f = open(vname, 'rb')
    s,v = os.path.split(vname)
    v,_ = os.path.splitext(v)
    _,s = os.path.split(s)
    string = str(f.read())
    splitstring = '\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46'
    mkdir_if_missing(tdir)
    strlist = string.split(splitstring)
    # deal with file segment, every segment is an image except the first one 
    # Skip the first one, which is filled with .seq header
    n_frames = 0
    for idx, img in enumerate(strlist[1:]):
        if ((idx+1) % skip) == 0:
            fname = "{}_{}_{:0>5}.jpg".format(s,v,idx) 
            fname=os.path.join(tdir, fname)
            
            with open(fname,'wb+') as i:
                #print('\rWriting image: {:s}'.format(fname))
                i.write(splitstring + img)
                i.close()
                n_frames += 1
    print('#Frames: {:d}'.format(n_frames))
    f.close()

def mkdir_if_missing(path):
    """
    If path not exist, then mkdir.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False

def get_classes():
    """
    Compatible with PASCAL dataset operate.
    """
    classes = ('__background__','person','people','person?','person-fa')
    return classes

def write_voc_results_file(all_boxes,image_ids,path,classes):
    """
    Convert the voc style result to caltech offical eval code.

        INPUT
            all_boxes: the detection result first dim is class, second dim is im_ind
            image_ids: all_boxes im_ind related to image_ids[im_ind] is the image name, like 'set00_V000_I00000'
            path:      ouput dir
            classes:   caltech classes, see get_classes, only convert 'person'
        OUPUT
            PATH---set06---V000.txt
                 |       |-V001.txt
                 |       |...
                 |-set07---V000.txt
                 |...    |...
    """
    # set_trace()
    for cls_ind, cls in enumerate(classes):
        if cls != 'person':
            continue
        print 'Writing {} VOC results file'.format(cls)
        tmp=''
        f=False
        for im_ind, im_id in enumerate(image_ids):
            s,v,i = im_id.split('_')
            vname = os.path.join(path,s,v+'.txt')

            if vname!=tmp:
                if f:
                    f.close()
                mkdir_if_missing(os.path.split(vname)[0])
                f = open(vname,'w')
                tmp = vname
                
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # the VOCdevkit expects 1-based indices
            for k in xrange(dets.shape[0]):
                line = "{:d},{:.3f},{:.3f},{:.3f},{:.3f},{:.7f}\n".format(
                    int(i[1:])+1, dets[k,0]+1, dets[k,1]+1, dets[k,2]-dets[k,0]+1, dets[k,3]-dets[k,1]+1, dets[k,-1])
                f.write(line)

def convert_voc_annoations(image_identifiers, ann_dir, param={}):
    '''
    Get all image_identifiers annotation

        INPUTS:
            image_identifiers   - a list contaions image identifier like 'set00_V000_00000'
            ann_dir             - 
            param               - filter param
        OUPUT:
            anno : {'set00_V000_I00121': [{'id': 3,
                                        'lbl': 'person',
                                        'lock': 0,
                                        'occl': 1,
                                        'pos': [230.68536627473338,131.379092381353,7.045344709775236,13.1208549490851],
                                        'posv': [230.68536627473338,131.379092381353,7.045344709775236,13.1208549490851]}],
                                        ...
                    ...}
    '''
    anno = {}
    data = defaultdict(dict)
    for dname in sorted(glob.glob(ann_dir+'/set*')):
        set_name = os.path.basename(dname)
        data[set_name] = defaultdict(dict)
        for anno_fn in sorted(glob.glob('{}/*.vbb'.format(dname))):
            vid_anno = load_vbb(anno_fn)
            video_name = os.path.splitext(os.path.basename(anno_fn))[0]
            data[set_name][video_name] = vid_anno['objLists']
            
    for image_identifier in image_identifiers:
        image_set_name = image_identifier[0:5]
        image_seq_name = image_identifier[6:10]
        image_id       = int(image_identifier[11:])
        #Tracer()()
        if image_id in data[image_set_name][image_seq_name]:
            if len(param)!=0:
                anno[image_identifier] = bbox_filter(data[image_set_name][image_seq_name][image_id],param)
            else:
                anno[image_identifier] = data[image_set_name][image_seq_name][image_id]
        else:
            print "Warning: No %s.jpg found in annotations" %(image_identifier)
           
        #vis_annotations(image_identifier, anno[image_identifier])
    return anno