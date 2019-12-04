import numpy as np
import torch
import h5py
from glob import glob
import os.path as osp
from collections import defaultdict
from torch.utils.data import Dataset


class HyperFeat(Dataset):
    def __init__(self, root, ):
        self.root = root
        self.data = []
        fpaths = sorted(glob(osp.join(root, '*.h5')))
        for fpath in fpaths:
            h5file = h5py.File(fpath, 'r')
            self.data.append(np.array(h5file['emb']))
        self.data = np.concatenate(self.data, axis=0)
        self.data = self.data[self.data[:, 1] != -1, :]  # rm -1 terms
        self.features, self.labels = torch.from_numpy(self.data[:, 3:]).float(), self.data[:, :3]
        del self.data
        # iCam, pid, centerFrame, 256-dim feat
        self.feature_dim = self.features.shape[1]
        self.pid_dic = []
        self.index_by_icam_pid_dic = defaultdict(dict)
        self.index_by_pid_dic = defaultdict(list)
        self.index_by_pid_icam_dic = defaultdict(dict)
        for index in range(self.labels.shape[0]):
            [icam, pid, frame] = self.labels[index, :]
            if pid not in self.pid_dic:
                self.pid_dic.append(pid)

            if icam not in self.index_by_icam_pid_dic:
                self.index_by_icam_pid_dic[icam] = defaultdict(list)
            self.index_by_icam_pid_dic[icam][pid].append(index)

            self.index_by_pid_dic[pid].append(index)

            if pid not in self.index_by_pid_icam_dic:
                self.index_by_pid_icam_dic[pid] = defaultdict(list)
            self.index_by_pid_icam_dic[pid][icam].append(index)

        pass

    def __getitem__(self, index):
        feat = self.features[index, :]
        iCam, pid, frame = map(int, self.labels[index, :])
        return feat, iCam, pid, frame

    def __len__(self):
        return self.labels.shape[0]


class SiameseHyperFeat(Dataset):
    def __init__(self, h_dataset, ):
        self.h_dataset = h_dataset
        self.feature_dim = self.h_dataset.feature_dim

    def __len__(self):
        return len(self.h_dataset)

    def __getitem__(self, index):
        feat1, cam1, pid1, frame1 = self.h_dataset.__getitem__(index)
        target = np.random.randint(0, 2)
        if pid1 == -1:
            target = 0

        # 1 for same
        if target == 1:
            siamese_index = index
            index_pool = self.h_dataset.index_by_pid_dic[pid1]
            if len(index_pool) > 1:
                while siamese_index == index:
                    siamese_index = np.random.choice(index_pool)
        # 0 for different
        else:
            pid_pool = self.h_dataset.pid_dic
            pid2 = np.random.choice(pid_pool)
            if len(pid_pool) > 1:
                while pid2 == pid1:
                    pid2 = np.random.choice(pid_pool)
            index_pool = self.h_dataset.index_by_pid_dic[pid2]
            siamese_index = np.random.choice(index_pool)

        feat2, cam2, pid2, frame2 = self.h_dataset.__getitem__(siamese_index)
        if target != (pid1 == pid2):
            target = (pid1 == pid2)
            pass

        return (feat1, feat2), target
