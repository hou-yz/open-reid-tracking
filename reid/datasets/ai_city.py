from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re


class AI_City(object):

    def __init__(self, root, type='reid', fps=1, trainval=False):
        if type == 'aic_gt':
            if not trainval:
                train_dir = '~/Data/AIC19/ALL_gt_bbox/train'
            else:
                train_dir = '~/Data/AIC19/ALL_gt_bbox/trainval'
            val_dir = '~/Data/AIC19/ALL_gt_bbox/val'
            self.images_dir = osp.join(osp.expanduser(train_dir), ('gt_bbox_{}_fps'.format(fps)))
            self.train_path = self.images_dir
            self.gallery_path = osp.join(osp.expanduser(val_dir), ('gt_bbox_{}_fps'.format(fps)))
            self.query_path = osp.join(osp.expanduser(val_dir), ('gt_bbox_{}_fps'.format(fps)))
        elif type == 'aic_det':
            self.images_dir = osp.join(root)
            self.train_path = self.images_dir
            self.gallery_path = osp.join(self.images_dir, 'bounding_box_test')
            self.query_path = osp.join(self.images_dir, 'query')
        else:
            self.images_dir = osp.join(root)
            self.train_path = osp.join(self.images_dir, 'bounding_box_train')
            self.gallery_path = osp.join(self.images_dir, 'bounding_box_test')
            self.query_path = osp.join(self.images_dir, 'query')
        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        self.type = type
        self.load()

    def preprocess(self, path, relabel=True, type='reid'):
        pattern = re.compile(r'([-\d]+)_s(\d+)_c(\d+)_f(\d+)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, scene, cam, frame = map(int, pattern.search(fname).groups())
            if type == 'aic_det':
                pid = 1
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append((fname, pid, cam))
        return ret, int(len(all_pids))

    def load(self):
        self.train, self.num_train_ids = self.preprocess(self.train_path, True, self.type)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False, self.type)
        self.query, self.num_query_ids = self.preprocess(self.query_path, False, self.type)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
