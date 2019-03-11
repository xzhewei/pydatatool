# Auto generate roadline reference to json annotation
import pydatatool as pdt
from pycocotools.coco import COCO
import numpy as np
from collections import defaultdict
import os.path
import logging

logging.basicConfig(level=logging.DEBUG)

# json_file = '../output/json/caltech_train_10x.json'
# json_file = '../output/json/caltech_test_1x.json'
# json_file = '../output/json/scut_train_10x.json'
# json_file = '../output/json/scut_test_1x.json'
# json_file = '../output/json/kaist_train_all_10x.json'
json_file = '../output/json/kaist_test_all_1x.json'

scut_train_json = pdt.load_json(json_file)

scut_train = COCO(json_file)

for index, img in enumerate(scut_train_json['images']):
    anns = scut_train.imgToAnns[img['id']]
    anns_f = [ann for ann in anns if not ann['iscrowd']]
    if len(anns_f):
        y_centers = [ann['bbox'][1]+ann['bbox'][3]/2 for ann in anns_f]
    else:
        y_centers = -1
    # simple average center
    # TODO(xzw): more smooth raodline
    roadline = np.mean(y_centers)

    scut_train_json['images'][index]['roadline'] = roadline

    logging.debug("Image ID:{} RoadLine:{}".format(img['id'],roadline))

y_centers = defaultdict(list)
for img in scut_train_json['images']:
    y_centers[img['id']] = img['roadline']
logging.debug("Done.")

output_name,ext = os.path.splitext(json_file)
output_name = output_name + '_roadline_pure' + ext
logging.debug("Save json to {}.".format(output_name))
pdt.save_json(scut_train_json,output_name)

output_name,ext = os.path.splitext(json_file)
output_name = output_name + '_roadline_only_pure' + ext
logging.debug("Save roadline json to {}.".format(output_name))
pdt.save_json(y_centers,output_name)

logging.debug("Done.")
