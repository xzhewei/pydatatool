import os
from pydatatool.citypersons import cvt_annotations,cvt_CSP_train_annotations
from pydatatool.citypersons import cvt_annotations_as_val_gt
import pydatatool as pdt

data_path = "../data/citypersons"
out_dir = "../output/citypersons/annotations"
# train_json = cvt_annotations_as_val_gt('train', data_path)
# val_json = cvt_annotations_as_val_gt('val', data_path)
# like CSP
# pdt.save_json(train_json, os.path.join(out_dir, "anno_train_annotaions.json"))

# like val_gt.json
# pdt.save_json(val_json, os.path.join(out_dir, "anno_val_gt_12.json"))

# same as val_gt.json
# val_gt = pdt.load_json("../data/citypersons/annotations/val_gt.json")
# val_json['annotations'] = val_gt['annotations']
# pdt.save_json(val_json, os.path.join(out_dir, "anno_val_gt_annotations.json"))

# like CSP
train_json = cvt_CSP_train_annotations('train', data_path)
pdt.save_json(train_json, os.path.join(out_dir, "train_csp.json"))