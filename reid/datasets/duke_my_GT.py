from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class DukeMyGT(Dataset):

    def __init__(self, root, split_id=0, num_val=10, download=True, iCams=list(range(1, 9)), fps=60):
        super(DukeMyGT, self).__init__(root, split_id=split_id)

        MTMC_dir = '/home/wangzd/Data/DukeMTMC/ALL_gt_bbox'
        if download:
            self.download(iCams, fps, MTMC_dir)

        # if not self._check_integrity():
        #     raise RuntimeError("Dataset not found or corrupted. " +
        #                        "You can use download=True to download it.")

        self.load(num_val)

    def download(self, iCams, fps, MTMC_dir):
        # if self._check_integrity():
        #     print("Files already downloaded and verified")
        #     return

        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        market_raw_dir = osp.join(self.root, 'market_raw')
        mkdir_if_missing(market_raw_dir)
        # Download the raw zip file
        fpath = osp.join(market_raw_dir, 'Market-1501-v15.09.15.zip')
        # Extract the file
        exdir = osp.join(market_raw_dir, 'Market-1501-v15.09.15')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=market_raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)
        duke_raw_dir = osp.join(MTMC_dir, ('gt_bbox_'+str(fps)+'_fps'))

        # 1501 identities (+1 for background) with 6 camera views each
        # and more than 7000 ids from dukemtmc
        identities = [[[] for _ in range(8 + 6)] for _ in range(10000)]

        def market_register(subdir, pattern=re.compile(r'([-\d]+)_c(\d)')):
            fpaths = sorted(glob(osp.join(exdir, subdir, '*.jpg')))
            pids = set()
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                if pid == -1: continue  # junk images are just ignored
                assert 0 <= pid <= 1501  # pid == 0 means background
                assert 1 <= cam <= 6
                cam = cam - 1 + 8
                pid += 8000
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'.format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
            return pids

        def duke_register(pattern=re.compile(r'([-\d]+)_c(\d)')):
            pids = set()
            copy_flag = 1
            for iCam in iCams:
                cam_dir = 'camera' + str(iCam)
                fpaths = sorted(glob(osp.join(duke_raw_dir, cam_dir, '*.jpg')))
                for fpath in fpaths:
                    fname = osp.basename(fpath)
                    pid, cam = map(int, pattern.search(fname).groups())
                    if pid == -1: continue  # junk images are just ignored
                    assert 0 <= pid <= 8000  # pid == 0 means background
                    assert 1 <= cam <= 8
                    cam -= 1  # from range[1,8]to range[0,7]
                    pids.add(pid)
                    # fname = ('{:08d}_{:02d}_{:04d}.jpg'.format(pid, cam, len(identities[pid][cam])))
                    identities[pid][cam].append(fname)
                    # only copy once
                    if osp.isfile(osp.join(images_dir, fname)) and copy_flag:
                        copy_flag = 0
                    if copy_flag:
                        shutil.copy(fpath, osp.join(images_dir, fname))
                pass
            return pids

        trainval_pids = duke_register()
        gallery_pids = market_register('bounding_box_test')
        query_pids = market_register('query')
        assert query_pids <= gallery_pids
        assert trainval_pids.isdisjoint(gallery_pids)

        # Save meta information into a json file
        meta = {'name': 'DukeTo1501', 'shot': 'multiple', 'num_cameras': 14,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))
