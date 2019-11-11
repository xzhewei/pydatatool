#coding:utf-8
import os
import sys
from .box import *
import pickle
from .matching import maxWeightMatching
def compute_JC(detection:np.ndarray,gt:np.ndarray,iou_thresh:np.ndarray):

    if detection.shape[0] ==0 or gt.shape[0] == 0:
        return list()

    nr_det = detection.shape[0]
    nr_gt = gt.shape[0]
    bipartite = list()
    for i in range(nr_det):
        for j in range(nr_gt):
            det_box = Box(detection[i,:4])
            gt_box = Box(gt[j,:4])
            elt_iou = det_box.IoU(gt_box)
            if elt_iou > iou_thresh:
                bipartite.append((i + 1,j + nr_det + 1,elt_iou))
    mates = maxWeightMatching(bipartite)
    cordinates = []
    if len(mates):
        row_id,col_id = list(),list()
        for i in range(nr_det):
            col = mates[i + 1] - nr_det - 1
            if col > -1:
                row_id.append(i)
                col_id.append(col)
        cordinates = [(row,col) for row,col in zip(row_id,col_id)]
    return cordinates

def compute_maximal_iou(proposals:np.ndarray,gt:np.ndarray):
	nr_proposals = proposals.shape[0]
	nr_gt = gt.shape[0]
	ious = np.zeros([nr_proposals])
	for i in range(nr_proposals):
		max_iou = 0.
		for j in range(nr_gt):
			p = Box(proposals[i,:4])
			b = Box(gt[j,:4])
			iou = p.IoU(b)
			if iou > max_iou:
				max_iou = iou
		ious[i] = max_iou
	return ious
	
def compute_iou_matrix(proposals:np.ndarray,gt:np.ndarray):
	nr_proposals = proposals.shape[0]
	nr_gt = proposals.shape[0]
	nr_gt = gt.shape[0]
	ious = np.zeros([nr_proposals,nr_gt])
	for i in range(nr_proposals):
		for j in range(nr_gt):
			p = Box(proposals[i,:4])
			b = Box(gt[j,:4])		
			ious[i,j] = p.IoU(b)
	return ious

def load(fpath):
	assert os.path.exists(fpath)
	with open(fpath,'rb') as fid:
		data = pickle.load(fid)
	return data