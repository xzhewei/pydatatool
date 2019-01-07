# Auto generate roadline reference to json annotation
import pydatatool as pdt
from pycocotools.coco import COCO
import numpy as np
from collections import defaultdict
import os.path
import logging

logging.basicConfig(level=logging.DEBUG)

json_file = '../output/json/scut_train_10x.json'

scut_train_json = pdt.load_json(json_file)

scut_train = COCO(json_file)

for index, img in enumerate(scut_train_json['images']):
    anns = scut_train.imgToAnns[img['id']]
    if len(anns):
        y_centers = [ann['bbox'][1]+ann['bbox'][3]/2 for ann in anns]
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
output_name = output_name + '_roadline' + ext
logging.debug("Save json to {}.".format(output_name))
pdt.save_json(scut_train_json,output_name)

output_name,ext = os.path.splitext(json_file)
output_name = output_name + '_roadline_only' + ext
logging.debug("Save roadline json to {}.".format(output_name))
pdt.save_json(y_centers,output_name)

logging.debug("Done.")
