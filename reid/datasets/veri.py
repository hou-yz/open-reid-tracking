from __future__ import print_function, absolute_import

import os.path as osp
import re
from glob import glob


class VeRi(object):
    def __init__(self, root):
        self.root = osp.join(root, 'VeRi')
        self.train_path = osp.join(root, 'VeRi/image_train/')
        self.gallery_path = osp.join(root, 'VeRi/image_query/')
        self.query_path = osp.join(root, 'VeRi/image_test/')

        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0
        self.num_cams = 20

        self.load()

    def preprocess(self, path):
        pattern = re.compile(r'(\d+)_c(\d+)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]
            ret.append((fname, pid, cam))
        return ret, int(len(all_pids))

    def load(self):
        self.train, self.num_train_ids = self.preprocess(self.train_path)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path)
        self.query, self.num_query_ids = self.preprocess(self.query_path)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
