#!/usr/bin/env bash
#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2,3 python3 IDE_triplet.py --train -d dukemtmc --logs-dir logs/ide_triplet/fc256/dukemtmc/basis --height 384 -s 1 --features 256 --output_feature fc --mygt_fps 1 --fix_bn 0 --re 0.5

CUDA_VISIBLE_DEVICES=2,3 python3 IDE_triplet.py --train -d duke_my_gt --logs-dir logs/ide_triplet/fc256/duke_my_gt/train/1_fps/basis --height 384 -s 1 --features 256 --output_feature fc --mygt_fps 1 --fix_bn 0 --re 0.5

# reid feat
CUDA_VISIBLE_DEVICES=2,3 python3 save_cnn_feature.py  -a ide -b 128 --resume logs/ide_triplet/fc256/duke_my_gt/train/1_fps/basis/model_best.pth.tar --features 256 --output_feature fc --l0_name ide_triplet_basis_train_1fps -s 1 --height 384 -d gt_test
# det feat
CUDA_VISIBLE_DEVICES=2,3 python3 save_cnn_feature.py  -a ide -b 128 --resume logs/ide_triplet/fc256/duke_my_gt/train/1_fps/basis/model_best.pth.tar --features 256 --output_feature fc --l0_name ide_triplet_basis_train_1fps -s 1 --height 384 -d detections --det_time trainval
# gt feat
CUDA_VISIBLE_DEVICES=2,3 python3 save_cnn_feature.py  -a ide -b 128 --resume logs/ide_triplet/fc256/duke_my_gt/train/1_fps/basis/model_best.pth.tar --features 256 --output_feature fc --l0_name ide_triplet_basis_train_1fps -s 1 --height 384 -d gt_all --det_time trainval
