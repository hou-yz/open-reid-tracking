import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import os.path as osp
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from reid import datasets
from reid.utils.draw_curve import draw_curve
from reid.metric.metric_trainer import CNNTrainer
from reid.metric.reid_feat_dataset import *
from reid.metric.MLP_model import MLP_metric
from reid.metric.metric_evaluate import metric_evaluate


def main(args):
    # dataset path
    if args.logs_dir is None:
        args.logs_dir = osp.join(f'logs/metric/mlp/{args.dataset}')
    else:
        args.logs_dir = osp.join(f'logs/metric/mlp/{args.dataset}')
    os.makedirs(args.logs_dir, exist_ok=True)

    root = osp.expanduser('~/Data')
    if 'duke' in args.dataset:
        if 'tracking' in args.dataset:
            root = osp.join(root, 'DukeMTMC')
        else:
            root = osp.join(root, 'DukeMTMC-reID')
    elif 'aic' in args.dataset:
        if 'tracking' in args.dataset:
            root = osp.join(root, 'AIC19')
        else:
            root = osp.join(root, 'AIC19-reid')
    elif args.dataset == 'market1501':
        root = osp.join(root, 'Market1501')
    elif args.dataset == 'veri':
        root = osp.join(root, 'VeRi')
    else:
        raise Exception
    root += '/L0-features/'
    assert args.data_dir, 'Must provide data directory'
    train_dir = args.data_dir
    query_dir = args.data_dir.replace('trainval', 'query')
    gallery_dir = args.data_dir.replace('trainval', 'gallery')

    feat_trainset = HyperFeat(root + train_dir)
    feat_queryset = HyperFeat(root + query_dir)
    feat_galleryset = HyperFeat(root + gallery_dir)
    siamese_trainset = SiameseHyperFeat(feat_trainset)
    siamese_testset = SiameseHyperFeat(feat_galleryset)

    train_loader = DataLoader(siamese_trainset, batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True, shuffle=True)
    test_loader = DataLoader(siamese_testset, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=True)

    # model
    model = MLP_metric(feature_dim=siamese_trainset.feature_dim, num_class=2).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20, 1)

    trainer = CNNTrainer(model, nn.CrossEntropyLoss(), )

    if args.train:
        # Draw Curve
        x_epoch = []
        train_loss_s = []
        train_prec_s = []
        test_loss_s = []
        test_prec_s = []
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        for epoch in range(1, args.epochs + 1):
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, cyclic_scheduler=scheduler)
            test_loss, test_prec = trainer.test(test_loader, )
            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            train_prec_s.append(train_prec)
            test_loss_s.append(test_loss)
            test_prec_s.append(test_prec)
            draw_curve(args.logs_dir + '/MetricNet.jpg', x_epoch, train_loss_s, train_prec_s,
                       None, test_loss_s, test_prec_s)
            pass
        torch.save({'state_dict': model.state_dict(), }, args.logs_dir + '/model.pth.tar')

    checkpoint = torch.load(args.logs_dir + '/model.pth.tar')
    model_dict = checkpoint['state_dict']
    model.load_state_dict(model_dict)
    trainer.test(test_loader)
    metric_evaluate(model, feat_queryset, feat_galleryset)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Metric learning on top of re-ID features')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'gcn'])
    parser.add_argument('-d', '--dataset', type=str, default='duke_reid', choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-j', '--num-workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=40, metavar='N')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR')
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, val set alone for validation")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--log-interval', type=int, default=300, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--data-dir', type=str, default=None, metavar='PATH')
    parser.add_argument('--logs-dir', type=str, default=None, metavar='PATH')
    args = parser.parse_args()
    main(args)
