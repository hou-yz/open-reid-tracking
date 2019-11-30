import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import datetime
import sys
import shutil
from distutils.dir_util import copy_tree
import time
import numpy as np
import torch
from bisect import bisect_right
from reid import models
from reid.evaluators import Evaluator
from reid.loss import *
from reid.trainers import Trainer
from reid.utils.logger import Logger
from reid.utils.my_utils import *
from reid.utils.serialization import save_checkpoint

'''
tricks from ZJU paper

baseline-S          check
warmup              ()
re                  ()
lsr                 ()
s=1                 ()
BNneck              ()
centerloss          skip
'''


def main(args):
    args.step_size = args.step_size.split(',')
    args.step_size = [int(x) for x in args.step_size]
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    if args.logs_dir is None:
        args.logs_dir = osp.join(f'logs/zju/{args.dataset}', datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
    if args.train:
        os.makedirs(args.logs_dir, exist_ok=True)
        copy_tree('./reid', args.logs_dir + '/scripts/reid')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(args.logs_dir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
        sys.stdout = Logger(os.path.join(args.logs_dir, 'log.txt'), )
    print('Settings:')
    print(vars(args))
    print('\n')

    # Create data loaders
    dataset, num_classes, train_loader, query_loader, gallery_loader, camstyle_loader = \
        get_data(args.dataset, args.data_dir, args.height, args.width, args.batch_size, args.num_workers,
                 args.combine_trainval, args.crop, args.tracking_icams, args.tracking_fps, args.re, args.num_instances,
                 camstyle=0, zju=1, colorjitter=args.colorjitter)

    # Create model
    model = models.create('zju', feature_dim=args.feature_dim, norm=args.norm,
                          num_classes=num_classes, last_stride=args.last_stride, arch=args.arch,
                          BNneck=args.BNneck)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        resume_fname = osp.join(f'logs/zju/{args.dataset}', args.resume, 'model_best.pth.tar')
        model, start_epoch, best_top1 = checkpoint_loader(model, resume_fname)
        print("=> Start epoch {}  best top1 {:.1%}".format(start_epoch, best_top1))
    model = nn.DataParallel(model).cuda()

    # Criterion
    criterion = [LSR_loss().cuda() if args.LSR else nn.CrossEntropyLoss().cuda(),
                 TripletLoss(margin=None if args.softmargin else args.margin).cuda()]

    # Optimizer
    if 'aic' in args.dataset:
        # Optimizer
        if hasattr(model.module, 'base'):  # low learning_rate the base network (aka. DenseNet-121)
            base_param_ids = set(map(id, model.module.base.parameters()))
            new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
            param_groups = [{'params': model.module.base.parameters(), 'lr_mult': 1},
                            {'params': new_params, 'lr_mult': 2}]
        else:
            param_groups = model.parameters()
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, )

    # Trainer
    trainer = Trainer(model, criterion)

    # Evaluator
    evaluator = Evaluator(model)

    if args.train:
        # Schedule learning rate
        def adjust_lr(epoch):
            if epoch <= args.warmup:
                alpha = epoch / args.warmup
                warmup_factor = 0.01 * (1 - alpha) + alpha
            else:
                warmup_factor = 1
            lr = args.lr * warmup_factor * (0.1 ** bisect_right(args.step_size, epoch))
            print('Current learning rate: {}'.format(lr))
            for g in optimizer.param_groups:
                if 'aic' in args.dataset:
                    g['lr'] = lr * g.get('lr_mult', 1)
                else:
                    g['lr'] = lr

        # Draw Curve
        epoch_s = []
        loss_s = []
        prec_s = []
        eval_epoch_s = []
        eval_top1_s = []

        # Start training
        for epoch in range(start_epoch + 1, args.epochs + 1):
            t0 = time.time()
            adjust_lr(epoch)
            # train_loss, train_prec = 0, 0
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, fix_bn=args.fix_bn, print_freq=10)

            if epoch < args.start_save:
                continue

            if epoch % 10 == 0:
                top1 = evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)
                eval_epoch_s.append(epoch)
                eval_top1_s.append(top1)
            else:
                top1 = 0

            is_best = top1 >= best_top1
            best_top1 = max(top1, best_top1)
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'best_top1': best_top1,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
            epoch_s.append(epoch)
            loss_s.append(train_loss)
            prec_s.append(train_prec)
            draw_curve(os.path.join(args.logs_dir, 'train_curve.jpg'), epoch_s, loss_s, prec_s,
                       eval_epoch_s, eval_top1_s)

            t1 = time.time()
            t_epoch = t1 - t0
            print('\n * Finished epoch {:3d}  top1: {:5.1%}  best_eval: {:5.1%} {}\n'.
                  format(epoch, top1, best_top1, ' *' if is_best else ''))
            print('*************** Epoch takes time: {:^10.2f} *********************\n'.format(t_epoch))
            pass

        # Final test
        print('Test with best model:')
        model, start_epoch, best_top1 = checkpoint_loader(model, osp.join(args.logs_dir, 'model_best.pth.tar'))
        print("=> Start epoch {}  best top1 {:.1%}".format(start_epoch, best_top1))

        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)
    else:
        print("Test:")
        evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ZJU baseline without center loss")
    parser.add_argument('--log', type=bool, default=1)
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501', choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64, help="batch size")
    parser.add_argument('-j', '--num-workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height, default: 256 for resnet*")
    parser.add_argument('--width', type=int, default=128, help="input width, default: 128 for resnet*")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, val set alone for validation")
    parser.add_argument('--tracking_icams', type=int, default=0, help="specify if train on single iCam")
    parser.add_argument('--tracking_fps', type=int, default=1, help="specify if train on single iCam")
    parser.add_argument('--re', type=float, default=0.5, help="random erasing")
    parser.add_argument('--crop', type=bool, default=1, help="resize then crop, default: True")
    parser.add_argument('--colorjitter', action='store_true', help="resize then crop, default: True")
    # model
    parser.add_argument('--feature_dim', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('-s', '--last_stride', type=int, default=1, choices=[1, 2])
    parser.add_argument('--norm', action='store_true', help="normalize feat, default: False")
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'densenet121'],
                        help='architecture for base network')
    parser.add_argument('--BNneck', type=bool, default=0, help="BN layer, default: True")
    # loss
    parser.add_argument('--margin', type=float, default=0.3, help="margin of the triplet loss, default: 0.3")
    parser.add_argument('--softmargin', action='store_true', help="use softmargin triplet loss, default: false")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, default: 4")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--LSR', type=bool, default=1, help="use label smooth loss, default: True")
    # training configs
    parser.add_argument('--train', action='store_true', help="train IDE model from start")
    parser.add_argument('--fix_bn', type=bool, default=0, help="fix (skip training) BN in base network")
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--step-size', default='30,60,80')
    parser.add_argument('--start_save', type=int, default=0, help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.expanduser('~/Data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=None)
    main(parser.parse_args())
