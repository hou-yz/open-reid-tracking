from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os

import numpy as np
import time
import json
import random
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid.datasets.det_duke import *
from reid import models
from reid.utils.data import transforms as T
from reid.utils.osutils import mkdir_if_missing
from reid.utils.serialization import load_checkpoint, save_checkpoint

from collections import OrderedDict
from reid.utils.meters import AverageMeter
from reid.feature_extraction import extract_cnn_feature

import h5py
import re


def checkpoint_loader(model, path, eval_only=False):
    checkpoint = load_checkpoint(path)
    pretrained_dict = checkpoint['state_dict']
    if isinstance(model, nn.DataParallel):
        Parallel = 1
        model = model.module.cpu()
    else:
        Parallel = 0
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    if eval_only and 'fc.weight' in pretrained_dict:
        del pretrained_dict['fc.weight']
        del pretrained_dict['fc.bias']
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    start_epoch = checkpoint['epoch']

    if Parallel:
        model = nn.DataParallel(model).cuda()

    return model, start_epoch


def extract_features(model, data_loader, args, print_freq=100, OpenPose_det=True):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # f_names = [[] for _ in range(8)]
    # features = [[] for _ in range(8)]

    if OpenPose_det:
        lines = [[] for _ in range(8)]
    else:
        lines = []

    end = time.time()
    for i, (imgs, fnames) in enumerate(data_loader):
        # data_time.update(time.time() - end)
        # if args.mygt_icams != 0 and OpenPose_det:
        #     pattern = re.compile(r'c(\d)_f(\d+)')
        #     start_cam, _ = map(int, pattern.search(fnames[0]).groups())
        #     end_cam, _ = map(int, pattern.search(fnames[-1]).groups())
        #     if start_cam > args.mygt_icams or end_cam < args.mygt_icams:
        #         continue

        pass

        outputs = extract_cnn_feature(model, imgs, eval_only=True)
        for fname, output in zip(fnames, outputs):
            if OpenPose_det:
                pattern = re.compile(r'c(\d)_f(\d+)')
                cam, frame = map(int, pattern.search(fname).groups())
                # f_names[cam - 1].append(fname)
                # features[cam - 1].append(output.numpy())
                line = np.concatenate([np.array([cam, frame]), output.numpy()])
                lines[cam - 1].append(line)
            else:
                pattern = re.compile(r'(\d+)_c(\d+)_f(\d+)')
                pid, cam, frame = map(int, pattern.search(fname).groups())
                line = output.numpy()
                lines.append(line)
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return lines


def main(args):
    tic = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file

    data_dir = osp.expanduser('~/Data/DukeMTMC/ALL_det_bbox')

    if args.dataset == 'detections':
        dataset_dir = osp.join(data_dir, ('det_bbox_OpenPose_' + args.det_time))
    else:
        dataset_dir = osp.expanduser('~/Data/DukeMTMC/ALL_gt_bbox/gt_bbox_1_fps/allcam')  # gt @ 1fps
        # dataset_dir = osp.expanduser('~/houyz/open-reid-PCB_n_RPP/data/dukemtmc/dukemtmc/raw/DukeMTMC-reID/bounding_box_test')  # reid
    if args.mygt_icams != 0:
        mygt_icams = [args.mygt_icams]
    else:
        mygt_icams = list(range(1, 9))

    if args.dataset == 'detections':
        dataset = DetDuke(dataset_dir, mygt_icams)
    else:
        dataset = DetDuke(dataset_dir)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.RectScale(args.height, args.width, interpolation=3),
        T.ToTensor(),
        normalizer,
    ])
    data_loader = DataLoader(Preprocessor(dataset, root=dataset_dir, transform=test_transformer),
                             batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=False, pin_memory=True)
    # Create model
    model = models.create('ide', num_features=args.features,
                          dropout=args.dropout, num_classes=0, last_stride=args.last_stride,
                          output_feature=args.output_feature)
    # Load from checkpoint
    model, start_epoch = checkpoint_loader(model, args.resume, eval_only=True)
    print("=> Start epoch {}".format(start_epoch))
    model = nn.DataParallel(model).cuda()
    toc = time.time() - tic
    print('*************** initialization takes time: {:^10.2f} *********************\n'.format(toc))

    tic = time.time()
    lines = extract_features(model, data_loader, args, OpenPose_det=(args.dataset == 'detections'))
    toc = time.time() - tic
    print('*************** compute features takes time: {:^10.2f} *********************\n'.format(toc))

    tic = time.time()
    # write file
    if args.dataset == 'detections':
        folder_name = osp.expanduser('~/Data/DukeMTMC/L0-features/') + "det_features_{}". \
            format(args.l0_name) + '_' + args.det_time
        mkdir_if_missing(folder_name)
        with open(osp.join(folder_name, 'args.json'), 'w') as fp:
            json.dump(vars(args), fp, indent=1)
        for cam in range(8):
            output_fname = folder_name + '/features%d.h5' % (cam + 1)
            mkdir_if_missing(os.path.dirname(output_fname))
            if args.mygt_icams != 0 and cam + 1 != args.mygt_icams:
                continue

            with h5py.File(output_fname, 'w') as f:
                mat_data = np.vstack(lines[cam])
                f.create_dataset('emb', data=mat_data, dtype=float)
                pass
    else:
        folder_name = osp.abspath(osp.join(working_dir, os.pardir)) + '/DeepCC/experiments/' + args.l0_name
        mkdir_if_missing(folder_name)
        with open(osp.join(folder_name, 'args.json'), 'w') as fp:
            json.dump(vars(args), fp, indent=1)
        if args.mygt_icams == 0:
            output_fname = folder_name + '/features.h5'
        else:
            output_fname = folder_name + '/features_icam' + str(args.mygt_icams) + '.h5'
        mkdir_if_missing(os.path.dirname(output_fname))

        with h5py.File(output_fname, 'w') as f:
            # asciiList = [n.encode("ascii", "ignore") for n in f_names[cam]]
            # f.create_dataset('f_names', (len(asciiList), 1), 'S10', asciiList)
            # emb = np.vstack(features[cam])
            # f.create_dataset('emb', data=emb, dtype=float)
            mat_data = np.vstack(lines)
            f.create_dataset('emb', data=mat_data, dtype=float)
            pass
    toc = time.time() - tic
    print('*************** write file takes time: {:^10.2f} *********************\n'.format(toc))
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='reid_test',
                        choices=['detections', 'reid_test'])
    parser.add_argument('-b', '--batch-size', type=int, default=64, help="batch size")
    parser.add_argument('-j', '--num-workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*")
    # model
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--output-feature', type=str, default='None')
    parser.add_argument('-s', '--last_stride', type=int, default=2,
                        choices=[1, 2])
    parser.add_argument('--output_feature', type=str, default='None')
    # misc
    parser.add_argument('--seed', type=int, default=1)
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--l0_name', type=str, metavar='PATH',
                        default='ide_2048_')
    parser.add_argument('--det_time', type=str, metavar='PATH',
                        default='trainval_mini')
    parser.add_argument('--mygt_icams', type=int, default=0, help="specify if train on single iCam")
    main(parser.parse_args())
