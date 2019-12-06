import numpy as np
import os.path as osp
import glob
import re

path = osp.expanduser('~/Data/VeRi/image_train')
fpaths = sorted(glob.glob(osp.join(path, '*.jpg')))
pattern = re.compile(r'(\d+)_c(\d+)_(\d+)')
all_pids = {}
ret = []
for fpath in fpaths:
    fname = osp.basename(fpath)
    pid, line, frame = map(int, pattern.search(fname).groups())
    if pid == -1: continue
    if pid not in all_pids:
        all_pids[pid] = len(all_pids)
    pid = all_pids[pid]
    ret.append((pid, line - 1, frame))

affinity_matrix = np.zeros([20, 20])
pid_cam_frame = np.array(ret)
for pid in all_pids.values():
    indices = np.where(pid_cam_frame[:, 0] == pid)
    samepid_cam_frame = pid_cam_frame[indices]
    samepid_cam_frame = samepid_cam_frame[samepid_cam_frame[:, 2].argsort()]
    for i in range(samepid_cam_frame.shape[0]):
        if i == 0:
            last_line = samepid_cam_frame[i, :]
        else:
            last_line = line
        line = samepid_cam_frame[i, :]
        if last_line[1] != line[1] or line[2] - last_line[2] > 200:
            affinity_matrix[last_line[1], line[1]] += 1
affinity_matrix += affinity_matrix.T
np.savetxt('affinity_matrix.txt', affinity_matrix, '%d')
pass
