import itertools
import json

import numpy as np
from mmdet.apis import inference_detector, init_detector


class ClothesDetector(object):


    def __init__(self, score_thr=0.3):
        self.model = init_detector(
            'configs/fashionformer/fashionpedia/fashionformer_r50_mlvl_feat_3x.py',
            'fashionformer_r50_3x.pth',
            device='cuda:0'
        )
        self.descriptions = json.load(open("category_attributes_descriptions.json"))
        self.score_thr = score_thr

    def detect(self, img):
        bbox_result, segm_result, attr_result  = inference_detector(self.model, np.array(img))
        bboxes = np.vstack(bbox_result)
        scores = bboxes[:, -1]
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        attrs = np.vstack([attr for attr in attr_result if len(attr) > 0])
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = list(itertools.chain(*segm_result))
            segms = np.stack(segms, axis=0)

        inds = scores > self.score_thr
        segms = segms[inds, ...]
        labels = labels[inds]
        scores = scores[inds]
        attrs = attrs[inds, :]

        attr_mask = [0]
        for i in range(1, len(attrs)):
            for j in range(0, i):
                if np.equal(attrs[i], attrs[j]).sum() == len(attrs[0]):
                    attr_mask += [j]
                    break
            else:
                attr_mask += [i]


        categories = self.descriptions['categories']

        tops = []
        bottoms = []
        shoes = []
        mask = np.zeros(img.size)
        for attr_idx in set(attr_mask):
            most_confident_mask = scores[np.array(attr_mask) == attr_idx].argmax()

            label = labels[np.array(attr_mask) == attr_idx][most_confident_mask]
            attr = attrs[np.array(attr_mask) == attr_idx][most_confident_mask]
            item_mask = segms[np.array(attr_mask) == attr_idx][most_confident_mask]

            if categories[label]['supercategory'] in ['upperbody', 'wholebody']:
                tops.append({
                    "name": categories[label]['name'],
        #             "attr_scores": attr,
                    "atttrs": self._get_attrs(attr),
                })
                mask[item_mask.T] = 1
            if categories[label]['supercategory'] in ['lowerbody']:
                bottoms.append({
                    "name": categories[label]['name'],
        #             "attr_scores": attr,
                    "atttrs": self._get_attrs(attr),
                })
                mask[item_mask.T] = 2
            if categories[label]['supercategory'] in ['legs and feet']:
                shoes.append({
                    "name": categories[label]['name'],
        #             "attr_scores": attr,
                    "atttrs": self._get_attrs(attr),
                })
                mask[item_mask.T] = 3
        return {
            'tops': tops,
            'bottoms': bottoms,
            'shoes': shoes,
            'mask': mask
        }


    def _get_attrs(self, attr_scores, threshold=0.2):
        attributes = self.descriptions['attributes']
        ignore_attributes = ['symmetrical', 'no non-textile material', 'no special manufacturing technique']
        supercategories = {}
        for at, score in zip(np.array(attributes)[attr_scores>threshold], attr_scores[attr_scores>threshold]):
            if at['name'] in ignore_attributes:
                continue
            if at['supercategory'] in supercategories:
                if score > supercategories[at['supercategory']]['score']:
                    at['score'] = score
                    supercategories[at['supercategory']] = at
            else:
                at['score'] = score
                supercategories[at['supercategory']] = at
        attrs = [at['name'] for at in supercategories.values()]

        return attrs
