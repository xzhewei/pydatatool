import os
import json
import glob
import re
import numpy as np


def load_ecp(path,time,mode):
    gt_path = os.path.join(path,'{}/labels/{}'.format(time,mode))
    gt_ext = '.json'
    gt_files = glob.glob(gt_path + '/*/*' + gt_ext)
    gt_files.sort()
    gts_ecp = []
    images = []
    city_id = get_city_id()
    for gt_file in gt_files:
        gt_fn = os.path.basename(gt_file)
        gt_frame_id = re.search('(.*?)' + gt_ext, gt_fn).group(1)
        with open(gt_file, 'rb') as f:
            gt = json.load(f)
        gt_frame = get_gt_frame(gt)
        for gt in gt_frame['children']:
            _prepare_ecp_gt(gt)
        c_i = gt_fn.split('_')[0]
        f_i = gt_fn.split('_')[1].split('.')[0]
        im_name = "{}/{}_{}.png".format(c_i,c_i,f_i)
        id = "{:0>2}{}".format(city_id[c_i],f_i)
        image = {
            "id": int(id),
            "file_name": im_name,
            "height": gt_frame['imageheight'],
            "width": gt_frame['imagewidth'],
            "tags": gt_frame['tags']
        }
        images.append(image)
        gts_ecp.append(gt_frame['children'])
    return gts_ecp,images

def ecp_to_coco(gt_ecp,objId,imgId):
    y0 = max(0, gt_ecp['y0'])
    y1 = min(1024, gt_ecp['y1'])
    x0 = max(0, gt_ecp['x0'])
    x1 = min(1920, gt_ecp['x1'])
    h = max(0,y1-y0)
    w = max(0,x1-x0)
    anno = {}
    anno['id'] = objId
    anno['instance_id'] = objId
    anno['image_id'] = imgId
    anno['category_id'] = get_category_id(gt_ecp['identity'])
    anno['identity'] = gt_ecp['identity']
    anno['bbox'] = [x0,y0,w,h]
    anno['bbox'] = [x0,y0,w,h]
    anno['ignore'] = 0
    anno['iscrowd'] = anno['category_id']==3
    anno['occl'] = ecp_occlusion_to_coco(gt_ecp['tags'])
    anno['segmentation'] = []
    anno['area'] = w*h
    anno['trunc'] = ecp_truncation_to_coco(gt_ecp['tags'])
    anno['orient'] = gt_ecp.get('orient', None)
    anno['tags'] = gt_ecp['tags']
    return anno

def convert_to_coco(path,time,mode):
    gts_ecp,images = load_ecp(path,time,mode)
    annotations = []
    objId = 1
    for gts,img in zip(gts_ecp,images):
        for gt in gts:
            ann = ecp_to_coco(gt,objId,img['id'])
            objId += 1
            annotations.append(ann)
    gt_coco = {"categories":get_categories(),
                "images":images,
                "annotations":annotations}
    return gt_coco

def get_categories():
    cat = [{'id':1, 'name':'pedestrian'},
           {'id':2, 'name':'rider'},
           {'id':3, 'name':'bicycle'},
           {'id':4, 'name':'motorbike'},
           {'id':5, 'name':'person-group-far-away'},
           {'id':6, 'name':'rider+vehicle-group-far-away'},
           {'id':7, 'name':'bicycle-group'},
           {'id':8, 'name':'buggy-group'},
           {'id':9, 'name':'motorbike-group'},
           {'id':10, 'name':'scooter-group'},
           {'id':11, 'name':'tricycle-group'},
           {'id':12, 'name':'wheelchair-group'}
           ]
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
            name = cat['name']
    return name
def ecp_occlusion_to_coco(tags):
    val = 0
    level_tags = {
        'occluded>10':10,
        'occluded>40':40,
        'occluded>80':80
    }
    for t in tags:
        if t in level_tags:
            val = level_tags[t]/100.0
    return val     
def ecp_truncation_to_coco(tags):
    val = 0
    for t in tags:
        if 'truncated' in t:
            for t_val in [10, 40, 80]:
                if str(t_val) in t:
                    val = t_val / 100.0
                    break
    return val

def _prepare_ecp_gt(gt):
    def translate_ecp_pose_to_image_coordinates(angle):
        angle = angle + 90.0

        # map to interval [0, 360)
        angle = angle % 360

        if angle > 180:
            # map to interval (-180, 180]
            angle -= 360.0

        return np.deg2rad(angle)

    orient = None
    if gt['identity'] == 'rider':
        if len(gt['children']) > 0:  # vehicle is annotated
            for cgt in gt['children']:
                if cgt['identity'] in ['bicycle', 'buggy', 'motorbike', 'scooter', 'tricycle',
                                       'wheelchair']:
                    orient = cgt.get('Orient', None) or cgt.get('orient', None)
    else:
        orient = gt.get('Orient', None) or gt.get('orient', None)

    if orient:
        gt['orient'] = translate_ecp_pose_to_image_coordinates(orient)
        gt.pop('Orient', None)

def get_gt_frame(gt_dict):
    if gt_dict['identity'] == 'frame':
        pass
    elif '@converter' in gt_dict:
        gt_dict = gt_dict['children'][0]['children'][0]
    elif gt_dict['identity'] == 'seqlist':
        gt_dict = gt_dict['children']['children']

    # check if json layout is corrupt
    assert gt_dict['identity'] == 'frame'
    return gt_dict

def get_city_id():
    return {'amsterdam': 0,
            'barcelona': 1,
            'basel': 2,
            'berlin': 3,
            'bologna': 4,
            'bratislava': 5,
            'brno': 6,
            'budapest': 7,
            'dresden': 8,
            'firenze': 9,
            'hamburg': 10,
            'koeln': 11,
            'leipzig': 12,
            'ljubljana': 13,
            'lyon': 14,
            'marseille': 15,
            'milano': 16,
            'montpellier': 17,
            'nuernberg': 18,
            'pisa': 19,
            'potsdam': 20,
            'prague': 21,
            'roma': 22,
            'stuttgart': 23,
            'szczecin': 24,
            'torino': 25,
            'toulouse': 26,
            'ulm': 27,
            'wuerzburg': 28,
            'zagreb': 29,
            'zuerich': 30}