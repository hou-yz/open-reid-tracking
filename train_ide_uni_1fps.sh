#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=3 python3 IDE.py --epochs 60  --train -d dukemtmc --logs-dir logs/ide_new/256/dukemtmc/raw --combine-trainval
#CUDA_VISIBLE_DEVICES=3 python3 IDE.py --epochs 60  --train -d dukemtmc --logs-dir logs/ide_new/256/dukemtmc/basis --height 384 -s 1 --features 256 --output_feature fc --tracking_fps 1 --re 0.5 --combine-trainval
#CUDA_VISIBLE_DEVICES=3 python3 IDE.py --epochs 60  --train -d duke_tracking --logs-dir logs/ide_new/256/duke_tracking/train/1_fps/basis --height 384 -s 1 --features 256 --output_feature fc --tracking_fps 1 --re 0.5
# reid feat
#CUDA_VISIBLE_DEVICES=3 python3 save_cnn_feature.py  -a ide --resume logs/ide_new/256/duke_tracking/train/1_fps/basis/model_best.pth.tar --features 256 --output_feature fc --l0_name ide_basis_train_1fps -s 1 --height 384 -d gt_test
# det feat
#CUDA_VISIBLE_DEVICES=3 python3 save_cnn_feature.py  -a ide --resume logs/ide_new/256/duke_tracking/train/1_fps/basis/model_best.pth.tar --features 256 --output_feature fc --l0_name ide_basis_train_1fps -s 1 --height 384 -d detections --det_time trainval
# gt feat
CUDA_VISIBLE_DEVICES=2,3 python3 save_cnn_feature.py -b 128 -a ide --resume logs/ide_new/256/duke_tracking/train/1_fps/basis/model_best.pth.tar --features 256 --output_feature fc --l0_name ide_basis_train_1fps -s 1 --height 384 -d gt_all --det_time trainval