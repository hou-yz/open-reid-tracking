from __future__ import print_function, absolute_import
import os.path as osp
import re
from glob import glob


class veri_vehicleID(object):
    def __init__(self):
        vid_train_file = '~/Data/VehicleID_V1.0/train_test_split/train_list.txt'
        train_dir = '~/Data/veri_vehicleid/image/'
        query_dir = '~/Data/AIC19_ReID/image_test'
        gallery_dir = '~/Data/AIC19_ReID/image_test'
        test_file = '~/Data/AIC19_ReID/test_track.txt'
        self.vid_train_file = osp.expanduser(vid_train_file)
        self.train_path = osp.expanduser(train_dir)

        veri_train_dir = '~/Data/VeRi/image_train/'
        self.veri_train_path = osp.expanduser(veri_train_dir)

        self.gallery_path = osp.expanduser(gallery_dir)
        self.query_path = osp.expanduser(query_dir)
        self.test_file = osp.expanduser(test_file)

        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        self.type = type
        self.load()

    def preprocess_train(self, vid_file, path, relabel):
        all_pids = {}
        ret = []
        # vehicle_id
        with open(vid_file) as vidf:
            vid_image_list = vidf.readlines()
        vid_image_list = [item.strip('\n') for item in vid_image_list]
        for item in vid_image_list:
            item_list = item.split(' ')
            image_num = item_list[0]
            pid = int(item_list[1])
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            fname = image_num + '.jpg'
            cam = 0
            ret.append((fname, pid, cam))
        # veri
        veri_select_pattern = re.compile(r'(\S+)/(\d+)_c(\d+)')
        veri_pattern = re.compile(r'(\d+)_c(\d+)')
        veri_fpaths = sorted(glob(osp.join(path, '*.jpg')))
        veri_fpaths = [item for item in veri_fpaths if veri_select_pattern.match(item)]
        for fpath in veri_fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, veri_pattern.search(fname).groups())
            pid = pid + 30000
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            ret.append((fname, pid, cam))

        return ret, int(len(all_pids))

    def query_preprocess(self, data):
        all_pids = {}
        ret = []

        with open(data, 'r') as d:
            image_list = d.readlines()

        image_list = [item.strip(' \n') for item in image_list]

        for index, item in enumerate(image_list):
            fname = item.split(' ')[0]
            pid = index
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]
            cam = int(fname.strip('.jpg'))
            ret.append((fname, pid, cam))

        return ret, int(len(all_pids))

    def gallery_preprocess(self, data):
        all_pids = {}
        ret = []

        with open(data, 'r') as d:
            image_list = d.readlines()

        image_list = [item.strip(' \n') for item in image_list]

        for index, item in enumerate(image_list):
            pid = index
            fnames = item.split(' ')[1:]
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]
            for fname in fnames:
                cam = int(fname.strip('.jpg'))
                ret.append((fname, pid, cam))

        return ret, int(len(all_pids))

    def load(self):
        self.train, self.num_train_ids = self.preprocess_train(self.vid_train_file, self.train_path, True)
        self.gallery, self.num_gallery_ids = self.gallery_preprocess(self.test_file)
        self.query, self.num_query_ids = self.query_preprocess(self.test_file)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
