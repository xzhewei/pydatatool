from .common import *
from tqdm import tqdm
from multiprocessing import Process,Queue
import numpy as np
import math
from .utils.infrastructure import compute_JC

def commom_process(func, data, nr_procs, *args):

    total = len(data)
    stride = math.ceil(total/nr_procs)
    result_queue = Queue(1000)
    results, procs = [],[]

    tqdm.monitor_interval = 0
    pbar = tqdm(total = total)

    for i in range(nr_procs):
        start = i*stride
        end = np.min([start+stride,total])
        sample_data = data[start:end]
        p = Process(target= func,args=(result_queue, sample_data, *args))
        p.start()
        procs.append(p)

    for i in range(total):

        t = result_queue.get()
        if t is None:
            pbar.update(1)
            continue
        results.append(t)
        pbar.update()
    for p in procs:
        p.join()
    return results

def _is_ignore(rb):
    flag = False
    if 'extra' in rb:
        if 'ignore' in rb['extra']:
            if rb['extra']['ignore']:
                flag = True
    return flag

def worker(result_queue, records, gt, bm_thr):

    total, eps = len(records), 1e-6
    for i in range(total):
        record = records[i]
        ID = record['ID']
        height, width = record['height'], record['width']

        if len(record['dtboxes']) < 1:
            result_queue.put_nowait(None)
            continue

        _gt = list(filter(lambda rb:rb['ID'] == ID, gt))
        if len(_gt) < 1:
            result_queue.put_nowait(None)
            continue
        
        _gt = _gt[0]
        flags = np.array([_is_ignore(rb) for rb in _gt['gtboxes']])
        rows = np.where(~flags)[0]

        gtboxes = np.vstack([_gt['gtboxes'][j]['fbox'] for j in rows])
        gtboxes = recover_func(gtboxes)
        gtboxes = clip_boundary(gtboxes, height, width).astype(np.float32)

        dtboxes = np.vstack([np.hstack([rb['box'], rb['score']]) for rb in record['dtboxes']])
        dtboxes = recover_func(dtboxes)
        dtboxes = clip_boundary(dtboxes, height, width).astype(np.float32)

        matches = compute_JC(dtboxes, gtboxes, bm_thr)

        k = len(matches)
        m = gtboxes.shape[0]
        n = dtboxes.shape[0]

        ratio = k / (m + n -k + eps)
        recall = k / (m + eps)
        cover = k / (n + eps)
        noise = 1 - cover

        result_dict = dict(ID = ID, ratio = ratio, recall = recall , noise = noise ,
            cover = cover, valid= k ,total = n, gtn = m)
        result_queue.put_nowait(result_dict)
        
if __name__ == '__main__':

    fpath = 'data/gt_human.odgt'
    gt = load_func(fpath)
    
    fpath = 'data/epoch-22.human'
    records = load_func(fpath)

    results = commom_process(worker, records, 4, gt, 0.5)
    mean_ratio = np.sum([rb['ratio'] for rb in results]) / len(results)
    print('mJI@0.5 is {:.4f}'.format(mean_ratio))
