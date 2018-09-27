from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import read_json, write_json


def _pluck(identities, indices, relabel=False, relabel_offset=0):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                # x, y, _ = map(int, name.split('_'))
                name_parts = name.split('_')
                if len(name_parts) == 5:
                    x, _, _, _, y = name_parts
                else:
                    x, y, _ = name_parts
                x = int(x)
                if 'c' in y and len(name_parts) == 3:
                    y = int(y.split('c')[-1]) - 1
                else:
                    y = int(y)

                if len(name_parts) == 3:
                    assert pid == x and camid == y
                # keep intergrity of the trainval set
                if relabel:
                    ret.append((fname, index + relabel_offset, camid))
                else:
                    ret.append((fname, pid, camid))
    return ret


class DukeMyGT(Dataset):

    def __init__(self, root, split_id=0, download=True, iCams=list(range(1, 9)), fps=60, camstyle=False):
        super(DukeMyGT, self).__init__(root, split_id=split_id)

        camstyle_path = '/home/wangzd/Data/DukeMTMC/ALL_gt_bbox/gt_bbox_6_fps/allcam_camstyle_stargan4reid'
        self.camstyle = []
        mygt_dir = '/home/wangzd/Data/DukeMTMC/ALL_gt_bbox'
        if download:
            self.download(iCams, fps, mygt_dir, camstyle_path, camstyle)

        # if not self._check_integrity():
        #     raise RuntimeError("Dataset not found or corrupted. " +
        #                        "You can use download=True to download it.")

        self.load(camstyle=camstyle)

    def download(self, iCams, fps, mygt_dir, camstyle_path, camstyle):
        # if self._check_integrity():
        #     print("Files already downloaded and verified")
        #     return

        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        reid_raw_dir = osp.join(self.root, 'reid_raw')
        mkdir_if_missing(reid_raw_dir)
        # reid zip file dir
        fpath = osp.join(reid_raw_dir, 'DukeMTMC-reID.zip')
        # Extract reid zip file
        exdir = osp.join(reid_raw_dir, 'DukeMTMC-reID')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=reid_raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)
        mygt_raw_dir = osp.join(mygt_dir, ('gt_bbox_' + str(fps) + '_fps'))

        # 7k+ ids from mygt
        # 7k+ ids from reid
        # 7k+ ids from fake
        identities = [[[] for _ in range(8)] for _ in range(30000)]

        def reid_register(subdir, pattern=re.compile(r'([-\d]+)_c(\d)')):
            fpaths = sorted(glob(osp.join(exdir, subdir, '*.jpg')))
            pids = set()
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                if pid == -1: continue  # junk images are just ignored
                assert 0 <= pid <= 8000  # pid == 0 means background
                assert 1 <= cam <= 8
                cam = cam - 1
                pid += 10000
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'.format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                # only copy once
                copy_flag = 1
                if osp.isfile(osp.join(images_dir, fname)) and copy_flag:
                    copy_flag = 0
                if copy_flag:
                    shutil.copy(fpath, osp.join(images_dir, fname))
            return pids

        def mygt_register(subdir, pattern=re.compile(r'([-\d]+)_c(\d)')):
            pids = set()
            for iCam in iCams:
                cam_dir = 'camera' + str(iCam)
                fpaths = sorted(glob(osp.join(subdir, cam_dir, '*.jpg')))
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
                    copy_flag = 1
                    if osp.isfile(osp.join(images_dir, fname)) and copy_flag:
                        copy_flag = 0
                    if copy_flag:
                        shutil.copy(fpath, osp.join(images_dir, fname))
                pass
            return pids

        def fake_register(subdir, trainval_pids):
            pids = set()
            og_pattern = re.compile(r'([-\d]+)_c(\d)')
            fake_cam_pattern = re.compile(r'fake_(\d)')  # use fakes transferred to iCam style
            fpaths = sorted(glob(osp.join(subdir, '*.jpg')))
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid, source_cam = map(int, og_pattern.search(fname).groups())
                fake_cam = int(fake_cam_pattern.search(fname).groups()[0])

                if pid == -1: continue  # junk images are just ignored
                if fake_cam == source_cam: continue  # skip self-transformed imgs
                # if pid not in trainval_pids: continue  # skip imgs not in trainval
                if fake_cam not in iCams: continue  # skip imgs not belong to iCams list

                assert 0 <= pid <= 8000  # pid == 0 means background
                assert 1 <= fake_cam <= 8
                fake_cam -= 1  # from range[1,8]to range[0,7]
                pid += 20000
                pids.add(pid)
                # fname = ('{:08d}_{:02d}_{:04d}.jpg'.format(pid, cam, len(identities[pid][cam])))
                identities[pid][fake_cam].append(fname)
                # only copy once
                copy_flag = 1
                if osp.isfile(osp.join(images_dir, fname)) and copy_flag:
                    copy_flag = 0
                if copy_flag:
                    shutil.copy(fpath, osp.join(images_dir, fname))
            pass

            return pids

        trainval_pids = mygt_register(mygt_raw_dir)
        gallery_pids = reid_register('bounding_box_test')
        query_pids = reid_register('query')
        if camstyle:
            camstyle_pids = fake_register(camstyle_path, trainval_pids)
        else:
            camstyle_pids = set()
        assert query_pids <= gallery_pids
        assert trainval_pids.isdisjoint(gallery_pids)

        # Save meta information into a json file
        meta = {'name': 'DukeMyGT', 'shot': 'multiple', 'num_cameras': 14,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids)),
            'camstyle': sorted(list(camstyle_pids))
        }]
        write_json(splits, osp.join(self.root, 'splits.json'))

    def load(self, camstyle=False, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

        # Randomly split train / val
        trainval_pids = np.asarray(self.split['trainval'])
        np.random.shuffle(trainval_pids)
        train_pids = sorted(trainval_pids)
        val_pids = sorted([])

        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        self.train = _pluck(identities, train_pids, relabel=True)
        self.val = _pluck(identities, val_pids, relabel=True)
        self.trainval = _pluck(identities, trainval_pids, relabel=True)
        self.query = _pluck(identities, self.split['query'])
        self.gallery = _pluck(identities, self.split['gallery'])
        self.camstyle = _pluck(identities, self.split['camstyle'], relabel=True, relabel_offset=len(trainval_pids))
        self.num_train_ids = len(train_pids) + len(self.split['camstyle']) * camstyle
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(trainval_pids) + len(self.split['camstyle']) * camstyle

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(len(train_pids), len(self.train)))
            print("  val      | {:5d} | {:8d}"
                  .format(len(val_pids), len(self.val)))
            print("  trainval | {:5d} | {:8d}"
                  .format(len(trainval_pids), len(self.trainval)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))
            print("  camstyle | {:5d} | {:8d}"
                  .format(len(self.split['camstyle']), len(self.camstyle)))
