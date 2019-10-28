from pydatatool.eurocity.utils import convert_to_coco
path = '/home/xzw/code/pydatatool/pydatatool/eurocity/ECPB/data'
time = 'night'
mode = 'train'
gt_coco = convert_to_coco(path,time,mode)

from pydatatool.utils import save_json
import os
out_dir = '/home/xzw/code/pydatatool/output/ecpb'
save_path = os.path.join(out_dir,'ecpb_{}_{}.json'.format(mode,time))
save_json(gt_coco,save_path)