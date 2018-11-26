import os.path as osp
from reid.utils.serialization import read_json, write_json
import shutil
from glob import glob
import csv
import re

for iCam in range(1, 9):
    fpaths = sorted(
        glob(osp.join(osp.expanduser('~/Data/DukeMTMC/ALL_gt_bbox/val/gt_bbox_1_fps/camera' + str(iCam)), '*.jpg')))
    res = []
    pattern = re.compile(r'([-\d]+)_c(\d)')
    for fpath in fpaths:
        fname = osp.basename(fpath)
        pid, cam = map(int, pattern.search(fname).groups())
        res.append([pid, fpath])
        pass

# Assuming res is a list of lists
with open('file_list.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(res)

pass
