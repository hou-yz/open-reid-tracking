from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os

import numpy as np
import time
import random
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid.datasets.det_duke import *
from reid import models
from reid.utils.data import transforms as T
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

from collections import OrderedDict
from reid.utils.meters import AverageMeter
from reid.feature_extraction import extract_cnn_feature

import h5py

if os.name == 'nt':  # windows
    num_workers = 0
    batch_size = 64
    pass
else:  # linux
    num_workers = 8
    batch_size = 256
    os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'


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
    if eval_only:
        del pretrained_dict['fc.weight']
        del pretrained_dict['fc.bias']
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    start_epoch = checkpoint['epoch'] + 1

    if Parallel:
        model = nn.DataParallel(model).cuda()

    return model, start_epoch


def extract_features(model, data_loader, args, print_freq=10):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # f_names = [[] for _ in range(8)]
    # features = [[] for _ in range(8)]
    lines = [[] for _ in range(8)]

    end = time.time()
    for i, (imgs, fnames) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs, eval_only=True, output_feature=args.output_feature)
        for fname, output in zip(fnames, outputs):
            cam, frame = int(fname[1]), int(fname[4:10])
            # f_names[cam - 1].append(fname)
            # features[cam - 1].append(output.numpy())
            line = np.concatenate([np.array([cam, frame]), output.numpy()])
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

    return lines


def main(args):
    tic = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (256, 128)

    dataset_dir = osp.join(args.data_dir, 'det_dataset_OpenPose')
    dataset = DetDuke(dataset_dir)
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.RectScale(256, 128),
        # T.Resize((256, 128), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])
    data_loader = DataLoader(
        Preprocessor(dataset, root=dataset_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True)
    # Create model
    model = models.create('ide', num_features=args.features,
                          dropout=args.dropout, num_classes=1)
    # Load from checkpoint
    model, start_epoch = checkpoint_loader(model, args.resume, eval_only=True)
    print("=> Start epoch {}".format(start_epoch))
    model = nn.DataParallel(model).cuda()
    toc = time.time() - tic
    print('*************** initialization takes time: {:^10.2f} *********************\n'.format(toc))

    tic = time.time()
    lines = extract_features(model, data_loader, args)
    toc = time.time() - tic
    print('*************** compute features takes time: {:^10.2f} *********************\n'.format(toc))

    if not os.path.isdir("det_features_OpenPose"):
        os.mkdir("det_features_OpenPose")
        pass
    tic = time.time()
    for cam in range(8):
        output_fname = 'det_features_OpenPose/features%d.h5' % (cam + 1)
        with h5py.File(output_fname, 'w') as f:
            # asciiList = [n.encode("ascii", "ignore") for n in f_names[cam]]
            # f.create_dataset('f_names', (len(asciiList), 1), 'S10', asciiList)
            # emb = np.vstack(features[cam])
            # f.create_dataset('emb', data=emb, dtype=float)
            mat_data = np.vstack(lines[cam])
            f.create_dataset('emb', data=mat_data, dtype=float)
            pass
        # write file

    toc = time.time() - tic
    print('*************** write file takes time: {:^10.2f} *********************\n'.format(toc))
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    # model
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--output-feature', type=str, default='None')
    # misc
    parser.add_argument('--seed', type=int, default=1)
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
