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
from reid.utils.meters import AverageMeter
from reid.utils.get_loaders import *


def save_file(lines, args, root, if_created):
    # write file
    if args.data_type == 'tracking_det':
        folder_name = root + f"/L0-features/det_{args.det_time}_features_{args.model}_{args.resume}"
        if args.dataset == 'aic':
            folder_name += f'_{args.det_type}'
    elif args.data_type == 'reid':
        folder_name = root + f"/L0-features/reid_trainval_features_{args.model}_{args.resume}"
    elif args.data_type == 'tracking_gt':  # only extract ground truth data from 'train' set
        folder_name = root + f"/L0-features/gt_{args.det_time}_features_{args.model}_{args.resume}"
    elif args.data_type == 'reid_test':  # reid_test: query/gallery
        folder_name = root + f"/L0-features/reid_{args.reid_test}_features_{args.model}_{args.resume}"
    else:
        raise Exception

    if args.re:
        folder_name += '_RE'
    if args.crop:
        folder_name += '_CROP'

    os.makedirs(folder_name, exist_ok=True)
    with open(osp.join(folder_name, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=1)
    for cam in range(len(lines)):
        output_fname = folder_name + '/features%d.h5' % (cam + 1)
        if args.tracking_icams != 0 and cam + 1 != args.tracking_icams and args.tracking_icams is not None:
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


def extract_n_save(model, data_loader, args, root, num_cams, is_detection=True, use_fname=True, gt_type='reid'):
    model.eval()
    print_freq = 1000
    batch_time = AverageMeter()
    data_time = AverageMeter()

    if_created = [0 for _ in range(num_cams)]
    lines = [[] for _ in range(num_cams)]

    end = time.time()
    for i, (imgs, fnames, pids, cams) in enumerate(data_loader):
        cams += 1
        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid, cam in zip(fnames, outputs, pids, cams):
            if is_detection:
                pattern = re.compile(r'c(\d+)_f(\d+)')
                cam, frame = map(int, pattern.search(fname).groups())
                # f_names[cam - 1].append(fname)
                # features[cam - 1].append(output.numpy())
                line = np.concatenate([np.array([cam, 0, frame]), output.numpy()])
            else:
                if use_fname:
                    pattern = re.compile(r'(\d+)_c(\d+)_f(\d+)')
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

            if_created = save_file(lines, args, root, if_created)

            lines = [[] for _ in range(num_cams)]

    save_file(lines, args, root, if_created)
    return


def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    tic = time.time()
    if args.tracking_icams:
        tracking_icams = [args.tracking_icams]

    if args.data_type == 'tracking_det':
        if args.dataset == 'duke_tracking':
            dataset_dir = osp.join(args.data_dir, 'DukeMTMC', 'ALL_det_bbox', f'det_bbox_OpenPose_{args.det_time}')
        elif args.dataset == 'aic_tracking':
            dataset_dir = osp.join(args.data_dir, 'AIC19', 'ALL_det_bbox',
                                   f'det_bbox_{args.det_type}_{args.det_time}', )
        fps = None
        use_fname = True
    elif args.data_type == 'reid':
        # args.det_time = 'trainval'
        dataset_dir = None
        fps = 1
        use_fname = False
    elif args.data_type == 'tracking_gt':
        if args.dataset == 'aic':
            args.det_time = 'trainval'
        dataset_dir = None
        fps = 60 if args.dataset == 'duke' else 10
        use_fname = True
    elif args.data_type == 'reid_test':  # reid_test
        dataset_dir = None
        fps = 1
        use_fname = False
    else:
        raise Exception

    print(dataset_dir)
    if args.dataset == 'duke_tracking':
        dataset = DukeMTMC(dataset_dir, data_type=args.data_type, iCams=tracking_icams, fps=fps,
                           trainval=args.det_time == 'trainval')
    elif args.dataset == 'aic_tracking':  # aic
        dataset = AI_City(dataset_dir, data_type=args.data_type, fps=fps, trainval=args.det_time == 'trainval',
                          gt_type=args.gt_type)
    else:
        dataset = datasets.create(args.dataset, args.data_dir)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.Resize([args.height, args.width]),
        T.Pad(10 * args.crop),
        T.RandomCrop([args.height, args.width]),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=args.re), ])
    # Create model
    model = models.create(args.model, feature_dim=args.features, num_classes=0, norm=args.norm,
                          dropout=args.dropout, last_stride=args.last_stride, arch=args.arch)
    # Load from checkpoint
    assert args.resume, 'must provide resume directory'
    resume_fname = osp.join(f'logs/{args.model}/{args.dataset}', args.resume, 'model_best.pth.tar')
    model, start_epoch, best_top1 = checkpoint_loader(model, resume_fname)
    print(f"=> Last epoch {start_epoch}")
    model = nn.DataParallel(model).cuda()
    model.eval()
    toc = time.time() - tic
    print('*************** initialization takes time: {:^10.2f} *********************\n'.format(toc))

    tic = time.time()
    if args.data_type == 'reid_test':
        args.reid_test = 'query'
        data_loader = DataLoader(Preprocessor(dataset.query, root=dataset.query_path, transform=test_transformer),
                                 batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        extract_n_save(model, data_loader, args, dataset.root, dataset.num_cams,
                       is_detection=False, use_fname=use_fname)
        args.reid_test = 'gallery'
        data_loader = DataLoader(Preprocessor(dataset.gallery, root=dataset.gallery_path, transform=test_transformer),
                                 batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        extract_n_save(model, data_loader, args, dataset.root, dataset.num_cams,
                       is_detection=False, use_fname=use_fname)
    else:
        data_loader = DataLoader(Preprocessor(dataset.train, root=dataset.train_path, transform=test_transformer),
                                 batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        extract_n_save(model, data_loader, args, dataset.root, dataset.num_cams,
                       is_detection=args.data_type == 'tracking_det', use_fname=use_fname)
    toc = time.time() - tic
    print('*************** compute features takes time: {:^10.2f} *********************\n'.format(toc))
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save re-ID features")
    # data
    parser.add_argument('--model', type=str, default='ide', choices=models.names())
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=['resnet50', 'densenet121'],
                        help='architecture for base network')
    parser.add_argument('-d', '--dataset', type=str, default='duke', choices=datasets.names())
    parser.add_argument('--data_type', type=str, default='reid',
                        choices=['tracking_det', 'reid', 'tracking_gt', 'reid_test'])
    parser.add_argument('-b', '--batch-size', type=int, default=64, help="batch size")
    parser.add_argument('-j', '--num-workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height, default: 256 for resnet*")
    parser.add_argument('--width', type=int, default=128, help="input width, default: 128 for resnet*")
    # model
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5, help='0.5 for ide/pcb, 0 for triplet/zju')
    parser.add_argument('-s', '--last_stride', type=int, default=2, choices=[1, 2])
    parser.add_argument('--norm', action='store_true', help="normalize feat, default: False")
    # misc
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.expanduser('~/Data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=None)
    parser.add_argument('--det_time', type=str, metavar='PATH', default='val',
                        choices=['trainval_nano', 'trainval', 'train', 'val', 'test_all', 'test'])
    parser.add_argument('--det_type', type=str, default='ssd', choices=['ssd', 'yolo'])
    parser.add_argument('--gt_type', type=str, default='gt', choices=['gt', 'labeled'])
    parser.add_argument('--tracking_icams', type=int, default=None, help="specify if train on single iCam")
    parser.add_argument('--seed', type=int, default=None)
    # data jittering
    parser.add_argument('--re', type=float, default=0, help="random erasing")
    parser.add_argument('--crop', action='store_true', help="resize then crop, default: False")
    main(parser.parse_args())
