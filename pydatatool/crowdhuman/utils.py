import os
import json
from PIL import Image
from tqdm import tqdm


# load .odgt annotations from crowdhuman
def load_file(fpath):
    assert os.path.exists(fpath)
    with open(fpath, 'r') as f:
        lines = f.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records


def crowdhuman2coco(odgt_path, json_path):
    records = load_file(odgt_path)
    num_records = len(records)
    print("number of odgt-records to transform: ", num_records)

    # storage for json annotations
    json_dict = {"images": [], "annotations": [], "categories": []}
    image = {}
    annotation = {}
    categories = {'person':1}
    image_id = 1
    bbox_id = 1
    # start
    print("start the transformation!")

    for i in tqdm(range(num_records)):
        file_name = records[i]['ID'] + '.jpg'
        im = Image.open("/home/xzw/code/pydatatool/data/crowdhuman/Images/" + file_name)
        image = {'file_name': file_name, 'height': im.size[1], 'width': im.size[0], 'id': image_id}
        json_dict['images'].append(image)

        gt_box = records[i]['gtboxes']
        gt_box_len = len(gt_box)
        for j in range(gt_box_len):
            category = gt_box[j]['tag']
            # if category == 'person':
            # category_id = 1
            # else:
            # category_id = 2
            # if category not in {'person','mask'}:
            #     new_id = len(categories) + 1
            #     categories[category] = new_id
            # category_id = categories[category]
            category_id = 1
            fbox = gt_box[j]['fbox'] # x,y,w,h

            # handle ignore
            ignore = 0
            if 'ignore' in gt_box[j]['extra']:
                ignore = gt_box[j]['extra']['ignore']

            # occ
            occ = 0
            if 'occ' in gt_box[j]['extra']:
                occ = gt_box[j]['extra']['occ']

            # fill annotation
            annotation = {
                'area': fbox[2] * fbox[3],
                'iscrowd': ignore,
                'image_id': image_id,
                'bbox': fbox,
                'hbox': gt_box[j]['hbox'],
                'vbox': gt_box[j]['vbox'],
                'category_id': category_id,
                'id': bbox_id,
                'head_attr': gt_box[j]['head_attr'],
                'ignore': ignore,
                'occ': occ
            }
            json_dict['annotations'].append(annotation)

            bbox_id += 1
        image_id += 1

    # for all data
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    # json_fp = open(json_path, 'w')
    # json_str = json.dumps(json_dict)
    # json_fp.write(json_str)
    # json_fp.close()
    from pydatatool.utils import save_json
    save_json(json_dict,json_path)
    print("Done.")

if __name__ == '__main__':
    odgt_path = '/home/xzw/code/pydatatool/data/crowdhuman/annotation_val.odgt'
    json_path = '/home/xzw/code/pydatatool/output/crowdhuman/crowdhuman_val_cat1.json'

    crowdhuman2coco(odgt_path, json_path)
    # print("hello world")

