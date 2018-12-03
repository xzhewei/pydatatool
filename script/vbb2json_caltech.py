import pydatatool as pdt
print "Load vbbs ..."
vbbs = pdt.caltech.load_vbbs('/home/all/datasets/caltech/annotations')
print "Done."
print "Convert train..."
annotations_train, annId_str, objId_str = pdt.caltech.vbbs2cocos(vbbs,'caltech_train')
print "Convert test..."
annotations_test, annId_str, objId_str = pdt.caltech.vbbs2cocos(vbbs,'caltech_test',annId_str, objId_str)
print "Done."
print "Save caltech train.."
skip = 3
image_ids_train = pdt.caltech.get_image_ids('caltech_train',vbbs,skip)
pdt.caltech.save_coco(annotations_train,image_ids_train,'../output/json/caltech_train_10x.json')
print "Done."
print "Save caltech test.."
skip = 30
image_ids_test = pdt.caltech.get_image_ids('caltech_test',vbbs,skip)
pdt.caltech.save_coco(annotations_test,image_ids_test,'../output/json/caltech_test_1x.json')
print "Done."
# print vbbs