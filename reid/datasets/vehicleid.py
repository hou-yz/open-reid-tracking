from __future__ import print_function, absolute_import
import os.path as osp


class vehicleID(object):
    def __init__(self):
        train_file = '~/Data/VehicleID_V1.0/train_test_split/train_list.txt'
        test_file = '~/Data/VehicleID_V1.0/train_test_split/test_list_800.txt'

        self.train_file = osp.expanduser(train_file)
        self.gallery_file = osp.expanduser(test_file)
        self.query_file = osp.expanduser(test_file)

        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        self.load()

    def preprocess(self, file, relabel, query=0):
        with open(file) as f:
            image_list = f.readlines()
        image_list = [item.strip('\n') for item in image_list]
        all_pids = {}
        ret = []
        for item in image_list:
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
            if query:
                cam = 1
            else:
                cam = 0
            ret.append((fname, pid, cam))

        return ret, int(len(all_pids))

    def load(self):
        self.train, self.num_train_ids = self.preprocess(self.train_file, True, 0)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_file, False, 0)
        self.query, self.num_query_ids = self.preprocess(self.query_file, False, 1)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
