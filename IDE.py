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
from reid import models
from reid.camstyle_trainer import CamStyleTrainer
from reid.evaluators import Evaluator
from reid.loss import *
from reid.trainers import Trainer
from reid.utils.logger import Logger
from reid.utils.my_utils import *
from reid.utils.serialization import save_checkpoint

'''            
    no crop for duke_tracking by default        check
    RE                                          check
'''


def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    if args.logs_dir is None:
        args.logs_dir = osp.join(f'logs/ide/{args.dataset}', datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
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
                 args.combine_trainval, args.crop, args.tracking_icams, args.tracking_fps, args.re, 0, args.camstyle)

    # Create model
    model = models.create('ide', feature_dim=args.feature_dim, num_classes=num_classes, norm=args.norm,
                          dropout=args.dropout, last_stride=args.last_stride, arch=args.arch)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        resume_fname = osp.join(f'logs/ide/{args.dataset}', args.resume, 'model_best.pth.tar')
        model, start_epoch, best_top1 = checkpoint_loader(model, resume_fname)
        print("=> Start epoch {}  best top1 {:.1%}".format(start_epoch, best_top1))
    model = nn.DataParallel(model).cuda()

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda() if not args.LSR else LSR_loss().cuda()

    # Optimizer
    if hasattr(model.module, 'base'):  # low learning_rate the base network (aka. ResNet-50)
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
        param_groups = [{'params': model.module.base.parameters(), 'lr_mult': 0.1},
                        {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=True)

    # Trainer
    if args.camstyle == 0:
        trainer = Trainer(model, criterion)
    else:
        trainer = CamStyleTrainer(model, criterion, camstyle_loader)

    # Evaluator
    evaluator = Evaluator(model)

    if args.train:
        # Schedule learning rate
        def adjust_lr(epoch):
            step_size = args.step_size
            lr = args.lr * (0.1 ** (epoch // step_size))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

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
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, fix_bn=args.fix_bn)

            if epoch < args.start_save:
                continue

            if epoch % 5 == 0:
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
    parser = argparse.ArgumentParser(description="Softmax loss classification")
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
    parser.add_argument('--re', type=float, default=0, help="random erasing")
    parser.add_argument('--crop', type=bool, default=1, help="resize then crop, default: True")
    # model
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('-s', '--last_stride', type=int, default=2, choices=[1, 2])
    parser.add_argument('--norm', action='store_true', help="normalize feat, default: False")
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'densenet121'],
                        help='architecture for base network')
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--LSR', action='store_true', help="use label smooth loss")
    # training configs
    parser.add_argument('--train', action='store_true', help="train IDE model from start")
    parser.add_argument('--fix_bn', type=bool, default=0, help="fix (skip training) BN in base network")
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--step-size', type=int, default=40)
    parser.add_argument('--start_save', type=int, default=0, help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--print-freq', type=int, default=1)
    # camstyle batchsize
    parser.add_argument('--camstyle', type=int, default=0)
    parser.add_argument('--fake_pooling', type=int, default=1)
    # misc
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.expanduser('~/Data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=None)
    main(parser.parse_args())
