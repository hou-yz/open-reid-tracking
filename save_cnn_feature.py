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
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.meters import AverageMeter
from reid.utils.my_utils import *
from reid.utils.osutils import mkdir_if_missing


def save_file(lines, args, if_created):
    # write file
    if args.type == 'detections':
        folder_name = osp.expanduser(
            '~/Data/{}/L0-features/'.format('DukeMTMC' if args.dataset == 'duke' else 'AIC19')) \
                      + "det_features_{}".format(args.l0_name) + '_' + args.det_time
        if args.dataset == 'aic':
            folder_name += '_{}'.format(args.det_type)

    elif args.type == 'gt_mini':
        folder_name = osp.abspath(osp.join(working_dir, os.pardir)) + \
                      '/DeepCC/experiments/' + args.l0_name + '_' + args.gt_type + '_' + args.det_time
    elif args.type == 'gt_all':  # only extract ground truth data from 'train' set
        folder_name = osp.expanduser(
            '~/Data/{}/L0-features/'.format('DukeMTMC' if args.dataset == 'duke' else 'AIC19')) \
                      + "gt_features_{}".format(args.l0_name)
    else:  # reid_test
        folder_name = osp.expanduser('~/Data/AIC19-reid/L0-features/') \
                      + "aic_reid_{}_features_{}".format(args.reid_test, args.l0_name)

    if args.re:
        folder_name += '_RE'
    if args.crop:
        folder_name += '_CROP'

    mkdir_if_missing(folder_name)
    with open(osp.join(folder_name, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=1)
    for cam in range(8 if args.dataset == 'duke' else 40):
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


def extract_features(model, data_loader, args, is_detection=True, use_fname=True, gt_type='reid'):
    model.eval()
    print_freq = 1000
    batch_time = AverageMeter()
    data_time = AverageMeter()

    if_created = [0 for _ in range(8 if args.dataset == 'duke' else 40)]
    lines = [[] for _ in range(8 if args.dataset == 'duke' else 40)]

    end = time.time()
    for i, (imgs, fnames, pids, cams) in enumerate(data_loader):
        cams += 1
        outputs = extract_cnn_feature(model, imgs, eval_only=True)
        for fname, output, pid, cam in zip(fnames, outputs, pids, cams):
            if is_detection:
                pattern = re.compile(r'c(\d+)_f(\d+)')
                cam, frame = map(int, pattern.search(fname).groups())
                # f_names[cam - 1].append(fname)
                # features[cam - 1].append(output.numpy())
                line = np.concatenate([np.array([cam, frame]), output.numpy()])
            else:
                pattern = re.compile(r'(\d+)_c(\d+)_f(\d+)')
                if use_fname:
                    pid, cam, frame = map(int, pattern.search(fname).groups())
                else:
                    cam, pid = cam.numpy(), pid.numpy()
                    frame = -1 * np.ones_like(pid)
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

            lines = [[] for _ in range(8 if args.dataset == 'duke' else 40)]

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
        tracking_icams = list(range(1, (8 if args.dataset == 'duke' else 40) + 1))

    data_dir = osp.expanduser('~/Data/{}/ALL_det_bbox'.format('DukeMTMC' if args.dataset == 'duke' else 'AIC19'))
    if args.type == 'detections':
        type = 'tracking_det'
        if args.dataset == 'duke':
            dataset_dir = osp.join(data_dir, 'det_bbox_OpenPose_' + args.det_time)
        else:
            dataset_dir = osp.join(data_dir, args.det_time, args.det_type)
        fps = None
        use_fname = True
    elif args.type == 'gt_mini':
        # args.det_time = 'trainval'
        type = 'reid'
        dataset_dir = None
        fps = 1
        use_fname = False
    elif args.type == 'gt_all':
        if args.dataset == 'aic':
            args.det_time = 'trainval'
        type = 'tracking_gt'
        dataset_dir = None
        fps = 60 if args.dataset == 'duke' else 10
        use_fname = True
    else:  # reid_test
        type = 'reid_test'
        dataset_dir = None
        fps = 1
        use_fname = False

    print(dataset_dir)
    if args.dataset == 'duke':
        dataset = DukeMTMC(dataset_dir, type=type, iCams=tracking_icams, fps=fps, trainval=args.det_time == 'trainval')
    else:  # aic
        dataset = AI_City(dataset_dir, type=type, fps=fps, trainval=args.det_time == 'trainval', gt_type=args.gt_type)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.Resize([args.height, args.width]),
        T.RandomHorizontalFlip(),
        T.Pad(10 * args.crop),
        T.RandomCrop([args.height, args.width]),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=args.re), ])
    # Create model
    if args.arch == 'zju':
        model = models.create(args.arch, num_features=args.features, norm=args.norm,
                              dropout=args.dropout, num_classes=0, last_stride=args.last_stride,
                              output_feature=args.output_feature, backbone=args.backbone, BNneck=args.BNneck)
    else:
        model = models.create(args.arch, num_features=args.features, norm=args.norm,
                              dropout=args.dropout, num_classes=0, last_stride=args.last_stride,
                              output_feature=args.output_feature)
    # Load from checkpoint
    model, start_epoch, best_top1 = checkpoint_loader(model, args.resume, eval_only=True)
    print("=> Start epoch {}".format(start_epoch))
    model = nn.DataParallel(model).cuda()
    model.eval()
    toc = time.time() - tic
    print('*************** initialization takes time: {:^10.2f} *********************\n'.format(toc))

    tic = time.time()
    if args.type == 'reid_test':
        args.reid_test = 'query'
        data_loader = DataLoader(Preprocessor(dataset.query, root=dataset.query_path, transform=test_transformer),
                                 batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        extract_features(model, data_loader, args, is_detection=False, use_fname=use_fname)
        args.reid_test = 'gallery'
        data_loader = DataLoader(Preprocessor(dataset.gallery, root=dataset.gallery_path, transform=test_transformer),
                                 batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        extract_features(model, data_loader, args, is_detection=False, use_fname=use_fname)
    else:
        data_loader = DataLoader(Preprocessor(dataset.train, root=dataset.train_path, transform=test_transformer),
                                 batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        extract_features(model, data_loader, args, is_detection=type == 'tracking_det', use_fname=use_fname)
    toc = time.time() - tic
    print('*************** compute features takes time: {:^10.2f} *********************\n'.format(toc))
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-a', '--arch', type=str, default='ide', choices=['ide', 'pcb', 'zju'])
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'densenet121'],
                        help='architecture for base network')
    parser.add_argument('-d', '--dataset', type=str, default='duke', choices=['duke', 'aic'])
    parser.add_argument('--type', type=str, default='gt_mini', choices=['detections', 'gt_mini', 'gt_all', 'reid_test'])
    parser.add_argument('-b', '--batch-size', type=int, default=64, help="batch size")
    parser.add_argument('-j', '--num-workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height, default: 256 for resnet*")
    parser.add_argument('--width', type=int, default=128, help="input width, default: 128 for resnet*")
    # model
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5, help='0.5 for ide/pcb, 0 for triplet/zju')
    parser.add_argument('-s', '--last_stride', type=int, default=2, choices=[1, 2])
    parser.add_argument('--output_feature', type=str, default='fc', choices=['pool5', 'fc'])
    parser.add_argument('--norm', action='store_true', help="normalize feat, default: False")
    parser.add_argument('--BNneck', action='store_true', help="BN layer, default: False")
    # misc
    parser.add_argument('--seed', type=int, default=1)
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))
    parser.add_argument('--l0_name', type=str, metavar='PATH')
    parser.add_argument('--det_time', type=str, metavar='PATH', default='val',
                        choices=['trainval_nano', 'trainval', 'train', 'val', 'test_all', 'test'])
    parser.add_argument('--det_type', type=str, default='ssd', choices=['ssd', 'yolo'])
    parser.add_argument('--gt_type', type=str, default='gt', choices=['gt', 'labeled'])
    parser.add_argument('--tracking_icams', type=int, default=0, help="specify if train on single iCam")
    # data jittering
    parser.add_argument('--re', type=float, default=0, help="random erasing")
    parser.add_argument('--crop', action='store_true', help="resize then crop, default: False")
    main(parser.parse_args())
