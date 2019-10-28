import argparse
import os
from scipy import io as scio
from pydatatool.utils import save_json
"""
{
    "categories":[{"id":1,"name":"pedestrian"},
                  {"id":2,"name":"rider"},
                  {"id":3,"name":"sitting person"},
                  {"id":4,"name":"other person"},
                  {"id":5,"name":"people group"},
                  {"id":0,"name":"ignore region"}]
    "images":[{"id":1,"im_name":"frankfurt_000000_000294_leftImg8bit.png","height":1024,"width":2048},
             ...]
    "annotations":[{"id":1,"image_id":1,"category_id":1,"iscrowd":0,"ignore":0,
                    "bbox":[947,406,17,40],"vis_bbox":[950,407,14,39],
                    "height":40,"vis_ratio":0.802941176471},
                  ...]
}
"""

"""
images_root: ./leftImg8bit_trainvaltest/leftImg8bit/
"""


def parse_mat(anno, objId, imgId):
    rows, cols = 1024, 2048
    cityname = anno[0][0][0][0].encode().decode('utf-8')
    imgname = anno[0][0][1][0].encode().decode('utf-8')
    gts = anno[0][0][2]

    annotation = []
    for i in range(len(gts)):
        # label, x1, y1, w, h = gts[i, :5]
        # x1, y1 = max(int(x1), 0), max(int(y1), 0)
        # w, h = min(int(w), cols - x1 - 1), min(int(h), rows - y1 - 1)
        # xv1, yv1, wv, hv = gts[i, 6:]
        # xv1, yv1 = max(int(xv1), 0), max(int(yv1), 0)
        # wv, hv = min(int(wv), cols - xv1 - 1), min(int(hv), rows - yv1 - 1)
        label, x1, y1, w, h = map(int, gts[i, :5])
        xv1, yv1, wv, hv = map(int, gts[i, 6:])
        anno={}
        anno['id'] = objId+i
        anno['instance_id'] = int(gts[i,5])
        anno['image_id'] = imgId
        anno['category_id'] = int(label)
        anno['iscrowd'] = 0 if any([xv1,yv1,wv,hv]) else 1
        ratio = (wv*hv)/(w*h)
        anno['ignore'] = 0 if (label == 1 and h >= 50 and ratio>=0.65) else 1
        anno['bbox'] = [x1, y1, w, h]
        anno['vis_bbox'] = [xv1,yv1,wv,hv]
        anno['height'] = h
        anno['vis_ratio'] = ratio
        # anno['ignore'] = (anno['vis_ratio']<0.65) or anno['ignore']
        annotation.append(anno)
    image = {
        "id":imgId,
        "im_name": os.path.join(cityname,imgname),
        "height": rows,
        "width": cols
    }
    return annotation, image

def parse_mat_to_csp_train(anno, objId, imgId):
    rows, cols = 1024, 2048
    cityname = anno[0][0][0][0].encode().decode('utf-8')
    imgname = anno[0][0][1][0].encode().decode('utf-8')
    gts = anno[0][0][2]

    annotation = []
    for i in range(len(gts)):
        label, x1, y1, w, h = gts[i, :5]
        x1, y1 = max(int(x1), 0), max(int(y1), 0)
        w, h = min(int(w), cols - x1 - 1), min(int(h), rows - y1 - 1)
        xv1, yv1, wv, hv = gts[i, 6:]
        xv1, yv1 = max(int(xv1), 0), max(int(yv1), 0)
        wv, hv = min(int(wv), cols - xv1 - 1), min(int(hv), rows - yv1 - 1)
        # label, x1, y1, w, h = map(int, gts[i, :5])
        # xv1, yv1, wv, hv = map(int, gts[i, 6:])
        anno={}
        anno['id'] = objId+i
        anno['instance_id'] = int(gts[i,5])
        anno['image_id'] = imgId
        anno['category_id'] = 1
        anno['iscrowd'] = 0 if (label == 1 and h >= 50) else 1
        ratio = (wv*hv)/(w*h)
        anno['ignore'] = 0
        anno['bbox'] = [int(x1), int(y1), int(w), int(h)]
        anno['vis_bbox'] = [int(xv1),int(yv1),int(wv),int(hv)]
        anno['height'] = h
        anno['vis_ratio'] = ratio
        # anno['ignore'] = (anno['vis_ratio']<0.65) or anno['ignore']
        annotation.append(anno)
    image = {
        "id":imgId,
        "im_name": os.path.join(cityname,imgname),
        "height": rows,
        "width": cols
    }
    return annotation, image

def parse_mat_as_val_gt(anno, objId, imgId):
    """
    Patse the annotation like offical file val_gt.json
    :param anno:
    :param objId:
    :param imgId:
    :return:
    """
    rows, cols = 1024, 2048
    cityname = anno[0][0][0][0].encode().decode('utf-8')
    imgname = anno[0][0][1][0].encode().decode('utf-8')
    gts = anno[0][0][2]

    annotation = []
    for i in range(len(gts)):
        label, x1, y1, w, h = map(int,gts[i, :5])
        # x1, y1 = max(int(x1), 0), max(int(y1), 0)
        # w, h = min(int(w), cols - x1 - 1), min(int(h), rows - y1 - 1)
        xv1, yv1, wv, hv = map(int,gts[i, 6:])
        # xv1, yv1 = max(int(xv1), 0), max(int(yv1), 0)
        # wv, hv = min(int(wv), cols - xv1 - 1), min(int(hv), rows - yv1 - 1)
        anno={}
        anno['id'] = objId+i
        # anno['instance_id'] = int(gts[i,5])
        anno['image_id'] = imgId
        anno['category_id'] = 1
        anno['iscrowd'] = 0 if any([xv1,yv1,wv,hv]) else 1
        # anno['ignore'] = 0 if (label == 1) else 1 #for test
        anno['ignore'] = 0 if (label == 1 and h >= 50) else 1 #for train
        anno['bbox'] = [x1, y1, w, h]
        anno['vis_bbox'] = [xv1,yv1,wv,hv]
        anno['height'] = h
        vis_ratio = (wv*hv)/(w*h)
        vis_ratio = round(vis_ratio, 12)
        if vis_ratio == 0 or vis_ratio == 1:
            anno['vis_ratio'] = int(vis_ratio)
        else:
            anno['vis_ratio'] = vis_ratio
        annotation.append(anno)
    image = {
        "id":imgId,
        "im_name": os.path.join(cityname,imgname),
        "height": rows,
        "width": cols
    }
    return annotation, image

def cvt_annotations(type, data_path):
    all_anno_path = os.path.join(data_path, 'annotations')
    anno_path = os.path.join(all_anno_path, 'anno_' + type + '.mat')
    annos = scio.loadmat(anno_path)
    index = 'anno_' + type + '_aligned'
    objId = 1
    imgId = 1
    annotations = []
    images = []

    for l in range(len(annos[index][0])):
        anno = annos[index][0][l]
        annotation, image = parse_mat(anno, objId, imgId)
        annotations.extend(annotation)
        images.append(image)
        imgId += 1
        objId += len(annotation)

    coco = {"categories":get_categories(),
            "images":images,
            "annotations":annotations}

    return coco

def cvt_CSP_train_annotations(type, data_path):
    all_anno_path = os.path.join(data_path, 'annotations')
    anno_path = os.path.join(all_anno_path, 'anno_' + type + '.mat')
    annos = scio.loadmat(anno_path)
    index = 'anno_' + type + '_aligned'
    objId = 1
    imgId = 1
    annotations = []
    images = []

    for l in range(len(annos[index][0])):
        anno = annos[index][0][l]
        annotation, image = parse_mat_to_csp_train(anno, objId, imgId)
        annotations.extend(annotation)
        images.append(image)
        imgId += 1
        objId += len(annotation)

    coco = {"categories":get_categories(),
            "images":images,
            "annotations":annotations}

    return coco

def cvt_annotations_as_val_gt(type, data_path):
    all_anno_path = os.path.join(data_path, 'annotations')
    anno_path = os.path.join(all_anno_path, 'anno_' + type + '.mat')
    annos = scio.loadmat(anno_path)
    index = 'anno_' + type + '_aligned'
    objId = 1
    imgId = 1
    annotations = []
    images = []

    for l in range(len(annos[index][0])):
        anno = annos[index][0][l]
        annotation, image = parse_mat_as_val_gt(anno, objId, imgId)
        annotations.extend(annotation)
        images.append(image)
        imgId += 1
        objId += len(annotation)

    coco = {"categories":get_categories(),
            "images":images,
            "annotations":annotations}

    return coco

def get_categories(id=None):
    cat = [{"id": 1, "name": "pedestrian"},
           {"id": 2, "name": "rider"},
           {"id": 3, "name": "sitting person"},
           {"id": 4, "name": "other person"},
           {"id": 5, "name": "people group"},
           {"id": 0, "name": "ignore region"}]
    return cat[id] if id else cat

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CityPersons annotations to mmdetection format')
    parser.add_argument('data_path', help='CityPerson dataset path')
    parser.add_argument('-o', '--out-dir', help='output path')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    data_path = args.data_path
    out_dir = args.out_dir if args.out_dir else data_path
    train_json = cvt_annotations('train', data_path)
    val_json = cvt_annotations('val', data_path)
    save_json(train_json,os.path.join(out_dir,"anno_train.json"))
    save_json(val_json,os.path.join(out_dir,"anno_val.json"))

if __name__ == '__main__':
    main()