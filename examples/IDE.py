from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp

import numpy as np
import random
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

if os.name == 'nt':  # windows
    num_workers = 0
    batch_size = 64
    pass
else:  # linux
    num_workers = 8
    batch_size = 64
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'


def get_data(name, split_id, data_dir, height, width, batch_size, workers,
             combine_trainval):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        # T.RandomSizedRectCrop(height, width),
        T.RectScale(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)

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


def checkpoint_loader(model, path):
    checkpoint = load_checkpoint(path)
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    start_epoch = checkpoint['epoch'] + 1
    best_top1 = checkpoint['best_top1']

    return model, start_epoch, best_top1


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (384, 128)
    dataset, num_classes, train_loader, val_loader, test_loader, eval_set_query = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, batch_size, num_workers,
                 args.combine_trainval)

    # Create model
    model = models.create('ide', num_features=256,
                          dropout=args.dropout, num_classes=num_classes)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        model, start_epoch, best_top1 = checkpoint_loader(model, args.resume)
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    model = nn.DataParallel(model).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        metric.train(model, train_loader)
        print("Validation:")
        evaluator.evaluate(val_loader, dataset.val, dataset.val, metric)
        print("Test:")
        evaluator.evaluate(test_loader, eval_set_query, dataset.gallery, metric)
        return

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

    if args.train:
        # Optimizer
        if hasattr(model.module, 'base'):  # low learning_rate the base network (aka. ResNet-50)
            base_param_ids = set(map(id, model.module.base.parameters()))
            new_params = [p for p in model.parameters() if
                          id(p) not in base_param_ids]
            param_groups = [
                {'params': model.module.base.parameters(), 'lr_mult': 0.1},
                {'params': new_params, 'lr_mult': 1.0}]
        else:
            param_groups = model.parameters()
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                    # momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    # nesterov=True
                                    )

        # Trainer
        trainer = Trainer(model, criterion)

        # Schedule learning rate
        def adjust_lr(epoch):
            if epoch < args.epochs - 20:
                lr = args.lr
            else:
                lr = args.lr * 0.1
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

        # Start training
        for epoch in range(start_epoch, args.epochs):
            adjust_lr(epoch)
            trainer.train(epoch, train_loader, optimizer)
            if epoch < args.start_save:
                continue
            top1 = evaluator.evaluate(val_loader, dataset.val, dataset.val)

            is_best = top1 >= best_top1
            best_top1 = max(top1, best_top1)
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1,
                'rpp': False,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, top1, best_top1, ' *' if is_best else ''))

        # Final test
        print('Test with best model:')
        model, _, _ = checkpoint_loader(model, osp.join(args.logs_dir, 'model_best.pth.tar'))

        metric.train(model, train_loader)
        evaluator.evaluate(test_loader, eval_set_query, dataset.gallery, metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    # parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--train', action='store_true',
                        help="train IDE model from start")
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
