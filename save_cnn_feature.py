from __future__ import print_function, absolute_import

import argparse
import json
import os
import re
import time

import h5py
import numpy as np
import torch
from torch.backends import cudnn

from reid import models
from reid.datasets import *
from reid.feature_extraction import extract_cnn_feature
from reid.utils.my_utils import *
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.meters import AverageMeter
from reid.utils.osutils import mkdir_if_missing


def save_file(lines, args, if_created):
    # write file
    if args.dataset == 'detections':
        folder_name = osp.expanduser('~/Data/DukeMTMC/L0-features/') + "det_features_{}". \
            format(args.l0_name) + '_' + args.det_time
    elif args.dataset == 'gt_test':
        folder_name = osp.abspath(osp.join(working_dir, os.pardir)) + '/DeepCC/experiments/' + args.l0_name
    else:
        folder_name = osp.expanduser('~/Data/DukeMTMC/L0-features/') + "gt_features_{}".format(args.l0_name)
    if args.re:
        folder_name += '_RE'
    if args.crop:
        folder_name += '_CROP'

    mkdir_if_missing(folder_name)
    with open(osp.join(folder_name, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=1)
    for cam in range(8):
        output_fname = folder_name + '/features%d.h5' % (cam + 1)
        mkdir_if_missing(os.path.dirname(output_fname))
        if args.tracking_icams != 0 and cam + 1 != args.tracking_icams:
            continue
        if not lines[cam]:
            continue

        if not if_created[cam]:
            with h5py.File(output_fname, 'w') as f:
                mat_data = np.vstack(lines[cam])
                f.create_dataset('emb', data=mat_data, dtype=float, maxshape=(None, None))
                pass
            if_created[cam] = 1
        else:
            with h5py.File(output_fname, 'a') as f:
                mat_data = np.vstack(lines[cam])
                f['emb'].resize((f['emb'].shape[0] + mat_data.shape[0]), axis=0)
                f['emb'][-mat_data.shape[0]:] = mat_data
                pass

    return if_created


def extract_features(model, data_loader, args, is_detection=True):
    model.eval()
    print_freq = 1000
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # f_names = [[] for _ in range(8)]
    # features = [[] for _ in range(8)]
    if_created = [0 for _ in range(8)]
    lines = [[] for _ in range(8)]

    end = time.time()
    for i, (imgs, fnames, _, _) in enumerate(data_loader):
        outputs = extract_cnn_feature(model, imgs, eval_only=True)
        for fname, output in zip(fnames, outputs):
            if is_detection:
                pattern = re.compile(r'c(\d)_f(\d+)')
                cam, frame = map(int, pattern.search(fname).groups())
                # f_names[cam - 1].append(fname)
                # features[cam - 1].append(output.numpy())
                line = np.concatenate([np.array([cam, frame]), output.numpy()])
            else:
                pattern = re.compile(r'(\d+)_c(\d+)_f(\d+)')
                pid, cam, frame = map(int, pattern.search(fname).groups())
                # line = output.numpy()
                line = np.concatenate([np.array([cam, pid, frame]), output.numpy()])
            lines[cam - 1].append(line)
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

            if_created = save_file(lines, args, if_created)

            lines = [[] for _ in range(8)]

    save_file(lines, args, if_created)
    return


def main(args):
    tic = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file

    if args.tracking_icams != 0:
        tracking_icams = [args.tracking_icams]
    else:
        tracking_icams = list(range(1, 9))

    data_dir = osp.expanduser('~/Data/DukeMTMC/ALL_det_bbox')
    if args.dataset == 'detections':
        type = 'tracking_det'
        dataset_dir = osp.join(data_dir, ('det_bbox_OpenPose_' + args.det_time))
        fps = None
    elif args.dataset == 'gt_test':
        type = 'tracking_gt'
        # dataset_dir = osp.expanduser('~/Data/DukeMTMC/ALL_gt_bbox/train/gt_bbox_1_fps')  # gt @ 1fps
        # dataset_dir = osp.expanduser('~/houyz/open-reid-PCB_n_RPP/data/dukemtmc/dukemtmc/raw/DukeMTMC-reID/bounding_box_test')  # reid
        dataset_dir = None
        fps = 1
    else:
        type = 'tracking_gt'
        # dataset_dir = osp.expanduser('~/Data/DukeMTMC/ALL_gt_bbox/train/gt_bbox_60_fps')
        dataset_dir = None
        fps = 60

    dataset = DukeMTMC(dataset_dir, type=type, iCams=tracking_icams, fps=fps, trainval=args.det_time == 'trainval')

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.crop:  # default: False
        test_transformer = T.Compose([
            T.RandomSizedRectCrop(args.height, args.width),
            T.ToTensor(),
            normalizer,
            T.RandomErasing(EPSILON=args.re), ])
    else:
        test_transformer = T.Compose([
            T.RectScale(args.height, args.width),
            T.ToTensor(),
            normalizer,
            T.RandomErasing(EPSILON=args.re), ])
    data_loader = DataLoader(Preprocessor(dataset.train, root=dataset.train_path, transform=test_transformer),
                             batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=False, pin_memory=True)
    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=0, last_stride=args.last_stride,
                          output_feature=args.output_feature)
    # Load from checkpoint
    model, start_epoch, best_top1 = checkpoint_loader(model, args.resume, eval_only=True)
    print("=> Start epoch {}".format(start_epoch))
    model = nn.DataParallel(model).cuda()
    toc = time.time() - tic
    print('*************** initialization takes time: {:^10.2f} *********************\n'.format(toc))

    tic = time.time()
    extract_features(model, data_loader, args, type == 'tracking_det')
    toc = time.time() - tic
    print('*************** compute features takes time: {:^10.2f} *********************\n'.format(toc))
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-a', '--arch', type=str, default='ide', choices=['ide', 'pcb'])
    parser.add_argument('-d', '--dataset', type=str, default='gt_test', choices=['detections', 'gt_test', 'gt_all'])
    parser.add_argument('-b', '--batch-size', type=int, default=64, help="batch size")
    parser.add_argument('-j', '--num-workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height, default: 256 for resnet*")
    parser.add_argument('--width', type=int, default=128, help="input width, default: 128 for resnet*")
    # model
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--output-feature', type=str, default='None')
    parser.add_argument('-s', '--last_stride', type=int, default=2, choices=[1, 2])
    parser.add_argument('--output_feature', type=str, default='None')
    # misc
    parser.add_argument('--seed', type=int, default=1)
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))
    parser.add_argument('--l0_name', type=str, metavar='PATH')
    parser.add_argument('--det_time', type=str, metavar='PATH', default='val',
                        choices=['trainval_nano', 'trainval', 'train', 'val', 'test_all'])
    parser.add_argument('--tracking_icams', type=int, default=0, help="specify if train on single iCam")
    # data jittering
    parser.add_argument('--re', type=float, default=0, help="random erasing")
    parser.add_argument('--crop', action='store_true',
                        help="resize then crop, default: False")
    main(parser.parse_args())
