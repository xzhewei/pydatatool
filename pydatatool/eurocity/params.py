class ParamsFactory:
    def __init__(self,
                 detections_type=['pedestrian'],
                 difficulty='',
                 ignore_other_vru=True,
                 tolerated_other_classes=['rider'],
                 dont_care_classes=['person-group-far-away'],
                 ignore_type_for_skipped_gts=1,
                 size_limits={'reasonable': 40, 'small': 30, 'occluded': 40, 'all': 20},
                 occ_limits={'reasonable': 40, 'small': 40, 'occluded': 80, 'all': 80},
                 size_upper_limits={'small': 60},
                 occ_lower_limits={'occluded': 40},
                 discard_depictions=True,
                 ):
        self.difficulty = difficulty
        self.ignore_other_vru = ignore_other_vru
        self.tolerated_other_classes = tolerated_other_classes
        self.dont_care_classes = dont_care_classes
        self.ignore_type_for_skipped_gts = ignore_type_for_skipped_gts
        self.detections_type = detections_type
        self.size_limits = size_limits
        self.occ_limits = occ_limits
        self.size_upper_limits = size_upper_limits
        self.occ_lower_limits = occ_lower_limits
        self.discard_depictions = discard_depictions
    
    def ignore_gt(self, gt):
        '''
        OUTPUT:
            None    filtered gt
            0       ground truth
            1       ignore ground truth
            2       ignore IoU thres = intersection/det['w']*det['h']
            3       ingore IoU thres = intersection/gt['w']*gt['h']
        '''
        h = gt['bbox'][3]

        if gt['identity'] in self.detections_type:
            pass
        elif self.ignore_other_vru and gt['identity'] in self.tolerated_other_classes:
            return 1
        elif gt['identity'] in self.dont_care_classes:
            if self.discard_depictions and gt['identity'] == 'person-group-far-away' and \
                    'depiction' in gt['tags']:
                return None
            else:
                return 2
        else:
            # None means don't use this annotation in skip_gt
            return None
        if gt['identity'] == 'pedestrian':
            for tag in gt['tags']:
                if tag in ['sitting-lying', 'behind-glass']:
                    return 1

        import re
        truncation = 0
        occlusion = 0
        for t in gt['tags']:
            if 'occluded' in t:
                matches = re.findall(r'\d+', t)
                if len(matches) == 1:
                    occlusion = int(matches[0])
            elif 'truncated' in t:
                matches = re.findall(r'\d+', t)
                if len(matches) == 1:
                    truncation = int(matches[0])

        if h < self.size_limits[self.difficulty] or \
                occlusion >= self.occ_limits[self.difficulty] or \
                truncation >= self.occ_limits[self.difficulty]:

            return self.ignore_type_for_skipped_gts

        if self.difficulty in self.size_upper_limits:
            if h > self.size_upper_limits[self.difficulty]:
                return self.ignore_type_for_skipped_gts

        if self.difficulty in self.occ_lower_limits:
            if occlusion < self.occ_lower_limits[self.difficulty]:
                return self.ignore_type_for_skipped_gts
        return 0
    
    def is_skip_gt(self,gt):
        if self.ignore_gt(gt) is None:
            return True
        return False
    
    def is_ignore_gt(self,gt):
        flag = self.ignore_gt(gt)
        if not flag and flag > 0:
            return True
        return False