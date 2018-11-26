from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re


class DukeMTMC(object):

    def __init__(self, root, has_subdir=False, duke_my_GT=False, iCams=list(range(1, 9)), fps=1, trainval=False):
        if duke_my_GT:
            if not trainval:
                train_dir = '~/Data/DukeMTMC/ALL_gt_bbox/train'
            else:
                train_dir = '~/Data/DukeMTMC/ALL_gt_bbox/trainval'
            val_dir = '~/Data/DukeMTMC/ALL_gt_bbox/train'
            self.train_path = osp.join(osp.expanduser(train_dir), ('gt_bbox_{}_fps'.format(fps)))
            self.gallery_path = osp.join(osp.expanduser(val_dir), ('gt_bbox_{}_fps'.format(fps)))
            self.query_path = osp.join(osp.expanduser(val_dir), ('gt_bbox_{}_fps'.format(fps)))
        else:
            self.images_dir = osp.join(root)
            self.train_path = osp.join(self.images_dir, 'bounding_box_train')
            self.gallery_path = osp.join(self.images_dir, 'bounding_box_test')
            self.query_path = osp.join(self.images_dir, 'query')
        self.camstyle_path = osp.join(self.images_dir, 'bounding_box_train_camstyle')
        self.train, self.query, self.gallery, self.camstyle = [], [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids, self.num_camstyle_ids = 0, 0, 0, 0

        self.has_subdir = has_subdir
        self.iCams = iCams
        self.load()

    def preprocess(self, path, relabel=True, has_subdir=False):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        ret = []
        if has_subdir:
            fpaths = []
            for iCam in self.iCams:
                fpaths += sorted(glob(osp.join(path, 'camera' + str(iCam), '*.jpg')))
        else:
            fpaths = sorted(glob(osp.join(path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
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
        self.train, self.num_train_ids = self.preprocess(self.train_path, True, self.has_subdir)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False)
        self.query, self.num_query_ids = self.preprocess(self.query_path, False)
        self.camstyle, self.num_camstyle_ids = self.preprocess(self.camstyle_path)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
        print("  camstyle  | {:5d} | {:8d}"
              .format(self.num_camstyle_ids, len(self.camstyle)))
