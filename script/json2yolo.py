import pydatatool as pdt
import os
from os import getcwd
from os.path import join
from collections import defaultdict

def convert(size,box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = box[0] + box[2]/2.0 - 1
    y = box[1] + box[3]/2.0 - 1
    w = box[2]
    h = box[3]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(sname,anns,img):
    w = img['width']
    h = img['height']
    out_file = open('output/yolo/labels/%s/%s.txt'%(sname,img['file_name'].split('.')[0]),'w')

    for ann in anns:
        cls_id = ann['category_id']
        ignore = ann['ignore']
        if ignore | False:
            continue

        b = ann['bbox']
        bb = convert((w,h),b)

        out_file.write(str(cls_id-1) + " " + " ".join([str(a) for a in bb]) + '\n')

print "Load vbbs ..."
vbbs = pdt.scut.load_vbbs('/home/all/datasets/SCUT_FIR_101/annotations_vbb')
print "Done."
print "Convert train..."
annotations_train, annId_str, objId_str = pdt.scut.vbbs2cocos(vbbs,'scut_train')
print "Convert test..."
annotations_test, annId_str, objId_str = pdt.scut.vbbs2cocos(vbbs,'scut_test',annId_str, objId_str)
print "Done."
print "get scut image_ids_train.."
image_ids_train = pdt.scut.get_image_ids('scut_train',vbbs,2)
print "Done."
print "get scut image_ids_test.."
image_ids_test = pdt.scut.get_image_ids('scut_test',vbbs,25)
print "Done."

imgs = {'train':image_ids_train,'test':image_ids_test}
imgid2anns_train = defaultdict(list)
imgid2anns_test = defaultdict(list)

for ann in annotations_train:
    imgid2anns_train[ann['image_id']].append(ann)
for ann in annotations_test:
    imgid2anns_test[ann['image_id']].append(ann)

imgid2anns = {}
imgid2anns['train'] = imgid2anns_train
imgid2anns['test'] = imgid2anns_test

sets = ['train','test']

# wd = getcwd()
wd = "/home/xuzhewei/lib/darknet/data/scut"
for s in sets:
    if not os.path.exists('output/yolo/labels/%s'%(s)):
        os.makedirs('output/yolo/labels/%s'%(s))
    list_file = open('output/yolo/%s.txt'%(s),'w')

    for img in imgs[s]:
        anns = imgid2anns[s][img['id']]
        fname= img['file_name'].split('.')[0]

        if len(anns)==0 and s=='train':
            # filt empty img
            continue

        convert_annotation(s,anns,img)
        list_file.write('%s/images/%s/%s.jpg\n' % (wd, s, fname))
        print('%s/images/%s/%s.jpg\n' % (wd, s, fname))
    list_file.close()







