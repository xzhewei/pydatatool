import pydatatool as pdt
print("Load vbbs ...")
vbbs = pdt.scut.load_vbbs('/home/all/datasets/SCUT_FIR_101/annotations_vbb')
print("Done.")
print("Convert train...")
param = pdt.scut.get_default_filter()
# param['hRng'] = [50,float('inf')]
# param['vRng'] = [1,1]
annotations_train, annId_str, objId_str = pdt.scut.vbbs2cocos(vbbs,'scut_train',0,0,param)
print("Convert test...")
annotations_test, annId_str, objId_str = pdt.scut.vbbs2cocos(vbbs,'scut_test',annId_str, objId_str,param)
print("Done.")
print("Save scut train..")
skip = 2
image_ids_train = pdt.scut.get_image_ids('scut_train',vbbs,skip)
pdt.scut.save_coco(annotations_train,image_ids_train,'../output/json/scut_train_10x.json')
print("Done.")
print("Save scut test..")
skip = 25
image_ids_test = pdt.scut.get_image_ids('scut_test',vbbs,skip)
pdt.scut.save_coco(annotations_test,image_ids_test,'../output/json/scut_test_1x.json')
print("Done.")