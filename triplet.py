import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import datetime
import sys
import shutil
from distutils.dir_util import copy_tree
import numpy as np
import torch
from reid import models
from reid.evaluators import Evaluator
from reid.loss import TripletLoss
from reid.trainers import Trainer
from reid.utils.logger import Logger
from reid.utils.my_utils import *
from reid.utils.serialization import save_checkpoint

'''
    triplet loss
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
        args.logs_dir = osp.join(f'logs/triplet/{args.dataset}',
                                 datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
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
    assert args.num_instances > 1, "num_instances should be larger than 1"
    assert args.batch_size % args.num_instances == 0, 'num_instances should divide batch_size'
    dataset, num_classes, train_loader, query_loader, gallery_loader, _ = \
        get_data(args.dataset, args.data_dir, args.height, args.width, args.batch_size, args.num_workers,
                 args.combine_trainval, args.crop, args.tracking_icams, args.tracking_fps, args.re, args.num_instances,
                 False)

    # Create model for triplet (num_classes = 0, num_instances > 0)
    model = models.create('ide', feature_dim=args.feature_dim, num_classes=0, norm=args.norm,
                          dropout=args.dropout, last_stride=args.last_stride)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        resume_fname = osp.join(f'logs/triplet/{args.dataset}', args.resume, 'model_best.pth.tar')
        model, start_epoch, best_top1 = checkpoint_loader(model, resume_fname)
        print("=> Last epoch {}  best top1 {:.1%}".format(start_epoch, best_top1))
        start_epoch += 1
    model = nn.DataParallel(model).cuda()

    # Criterion
    criterion = TripletLoss(margin=args.margin).cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Trainer
    trainer = Trainer(model, criterion)

    # Evaluator
    evaluator = Evaluator(model)

    if args.train:
        # Schedule learning rate
        def adjust_lr(epoch):
            if epoch <= args.step_size:
                lr = args.lr
            else:
                lr = args.lr * (0.001 ** (float(epoch - args.step_size) / (args.epochs - args.step_size)))
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
            adjust_lr(epoch)
            # train_loss, train_prec = 0, 0
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, fix_bn=args.fix_bn)

            if epoch < args.start_save:
                continue

            if epoch % 25 == 0:
                top1 = evaluator.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery)
                eval_epoch_s.append(epoch)
                eval_top1_s.append(top1)
            else:
                top1 = 0

            is_best = top1 >= best_top1
            best_top1 = max(top1, best_top1)
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch,
                'best_top1': best_top1,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
            epoch_s.append(epoch)
            loss_s.append(train_loss)
            prec_s.append(train_prec)
            draw_curve(os.path.join(args.logs_dir, 'train_curve.jpg'), epoch_s, loss_s, prec_s,
                       eval_epoch_s, eval_top1_s)
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
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    parser.add_argument('--log', type=bool, default=1)
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501', choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=128, help="batch size")
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
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('-s', '--last_stride', type=int, default=2, choices=[1, 2])
    parser.add_argument('--norm', action='store_true', help="normalize feat, default: False")
    # loss
    parser.add_argument('--margin', type=float, default=0.3, help="margin of the triplet loss, default: 0.3")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # optimizer
    parser.add_argument('--lr', type=float, default=2e-4, help="learning rate of ALL parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--train', action='store_true', help="train IDE model from start")
    parser.add_argument('--fix_bn', type=bool, default=0, help="fix (skip training) BN in base network")
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--step-size', type=int, default=150)
    parser.add_argument('--start_save', type=int, default=0, help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--print-freq', type=int, default=10)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.expanduser('~/Data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=None)
    main(parser.parse_args())
