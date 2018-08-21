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

from reid.loss import TripletLoss
from reid.utils.data.sampler import RandomIdentitySampler
from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

'''
    triplet loss
'''


def get_data(name, split_id, data_dir, height, width, batch_size, num_instances, workers,
             combine_trainval, re=0):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        # T.RandomSizedRectCrop(height, width),
        T.Resize((288, 144), interpolation=3),
        T.RandomCrop((256, 128)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(EPSILON=re),
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    # slimmer & faster query
    indices_eval_query = random.sample(range(len(dataset.query)), int(len(dataset.query) / 5))
    eval_set_query = list(dataset.query[i] for i in indices_eval_query)

    test_loader = DataLoader(
        Preprocessor(list(set(eval_set_query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader, eval_set_query,


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

    start_epoch = checkpoint['epoch']
    best_top1 = checkpoint['best_top1']

    if Parallel:
        model = nn.DataParallel(model).cuda()

    return model, start_epoch, best_top1


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    # Redirect print to both console and log file
    if (not args.evaluate) and args.log:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, 'num_instances should divide batch_size'
    dataset, num_classes, train_loader, val_loader, test_loader, eval_set_query = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.num_instances, args.num_workers,
                 args.combine_trainval, args.re)

    # Create model for triplet (num_classes = 0)
    model = models.create('ide', num_features=args.features,
                          dropout=args.dropout, num_classes=0)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        if args.evaluate:
            model, start_epoch, best_top1 = checkpoint_loader(model, args.resume, eval_only=True)
        else:
            model, start_epoch, best_top1 = checkpoint_loader(model, args.resume)
        print("=> Start epoch {}  best top1_eval {:.1%}"
              .format(start_epoch, best_top1))
    model = nn.DataParallel(model).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        metric.train(model, train_loader)
        # print("Validation:")
        # evaluator.evaluate(val_loader, dataset.val, dataset.val, eval_only=True, output_feature=args.output_feature, metric=metric)
        print("Test:")
        evaluator.evaluate(test_loader, eval_set_query, dataset.gallery,
                           metric=metric, eval_only=True, output_feature=args.output_feature)
        return

    # Criterion
    criterion = TripletLoss(margin=args.margin).cuda()

    if args.train:
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)

        # Trainer
        trainer = Trainer(model, criterion)

        # Schedule learning rate
        def adjust_lr(epoch):
            lr = args.lr if epoch <= 100 else \
                args.lr * (0.001 ** ((epoch - 100) / 50.0))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

        # Start training
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            adjust_lr(epoch)
            trainer.train(epoch, train_loader, optimizer, fixed_bn=True)
            if epoch < args.start_save:
                continue

            if epoch % 5 == 0:
                print("Validation:")
                top1_eval = evaluator.evaluate(val_loader, dataset.val, dataset.val,
                                               metric=metric, eval_only=True, output_feature=args.output_feature)
                # print("Test:")
                # top1_test = evaluator.evaluate(test_loader, eval_set_query, dataset.gallery,
                #                                metric=metric, eval_only=True, output_feature=args.output_feature)

                is_best = top1_eval >= best_top1
                best_top1 = max(top1_eval, best_top1)
                save_checkpoint({
                    'state_dict': model.module.state_dict(),
                    'epoch': epoch + 1,
                    'best_top1': best_top1,
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
                print('\n * Finished epoch {:3d}  top1_eval: {:5.1%}  best_eval: {:5.1%} \n'.
                      format(epoch, top1_eval, best_top1, ' *' if is_best else ''))

            t1 = time.time()
            t_epoch = t1 - t0
            print('*************** Epoch takes time: {:^10.2f} *********************\n'.format(t_epoch))
            pass

        # Final test
        print('Test with best model:')
        model, _, _ = checkpoint_loader(model, osp.join(args.logs_dir, 'model_best.pth.tar'), eval_only=True)

        metric.train(model, train_loader)
        evaluator.evaluate(test_loader, eval_set_query, dataset.gallery,
                           metric=metric, eval_only=True, output_feature=args.output_feature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    parser.add_argument('--log', type=int, default=1)
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('-j', '--num-workers', type=int, default=8)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=0)
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--train', action='store_true',
                        help="train IDE model from start")
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # random erasing
    parser.add_argument('--re', type=float, default=0)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--output_feature', type=str, default='pool5')
    main(parser.parse_args())
