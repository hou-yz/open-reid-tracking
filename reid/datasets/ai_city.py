from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re
from collections import defaultdict
import xml.dom.minidom as XD


class AI_City(object):

    def __init__(self, root, type='reid', fps=5, trainval=False):
        if type == 'tracking_gt':
            if not trainval:
                train_dir = '~/Data/AIC19/ALL_gt_bbox/train'
            else:
                train_dir = '~/Data/AIC19/ALL_gt_bbox/trainval'
            val_dir = '~/Data/AIC19/ALL_gt_bbox/val'
            self.train_path = osp.join(osp.expanduser(train_dir), ('gt_bbox_{}_fps'.format(fps)))
            self.gallery_path = osp.join(osp.expanduser(val_dir), ('gt_bbox_{}_fps'.format(fps)))
            self.query_path = osp.join(osp.expanduser(val_dir), ('gt_bbox_{}_fps'.format(fps)))
        elif type == 'tracking_det':
            self.train_path = root
            self.gallery_path = None
            self.query_path = None
        else:  # reid
            root = osp.expanduser('~/Data/AIC19-reid')
            self.train_path = osp.join(root, 'image_train')
            val_dir = '~/Data/AIC19/ALL_gt_bbox/val'
            self.gallery_path = osp.join(osp.expanduser(val_dir), ('gt_bbox_{}_fps'.format(fps)))
            self.query_path = osp.join(osp.expanduser(val_dir), ('gt_bbox_{}_fps'.format(fps)))

            xml_dir = osp.join(root, 'train_label.xml')
            self.reid_info = XD.parse(xml_dir).documentElement.getElementsByTagName('Item')
            self.index_by_fname_dict = defaultdict()
            for index in range(len(self.reid_info)):
                fname = self.reid_info[index].getAttribute('imageName')
                self.index_by_fname_dict[fname] = index

        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        self.type = type
        self.load()

    def preprocess(self, path, relabel=True, type='reid'):
        if type == 'tracking_det':
            pattern = re.compile(r'c(\d+)_f(\d+)')
        elif type == 'tracking_gt':
            pattern = re.compile(r'([-\d]+)_c(\d)')
        else:  # reid
            pattern = None
        all_pids = {}
        ret = []
        if path is None:
            return ret, int(len(all_pids))
        fpaths = sorted(glob(osp.join(path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            if type == 'tracking_det':
                cam, frame = map(int, pattern.search(fname).groups())
                pid = 1
            elif type == 'tracking_gt':
                pid, cam = map(int, pattern.search(fname).groups())
            else:  # reid
                pid, cam = map(int, [self.reid_info[self.index_by_fname_dict[fname]].getAttribute('vehicleID'),
                                     self.reid_info[self.index_by_fname_dict[fname]].getAttribute('cameraID')[1:]])
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
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False, 'tracking_gt')
        self.query, self.num_query_ids = self.preprocess(self.query_path, False, 'tracking_gt')

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
