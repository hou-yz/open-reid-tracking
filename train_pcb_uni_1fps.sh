#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=6,7 python3 PCB.py --train -d dukemtmc --logs-dir logs/pcb_new/256/dukemtmc/raw --height 384 -s 1 --features 256 --output_feature fc --combine-trainval
#CUDA_VISIBLE_DEVICES=6,7 python3 PCB.py --train -d dukemtmc --logs-dir logs/pcb_new/256/dukemtmc/basis_crop --height 384 -s 1 --features 256 --output_feature fc --re 0.5 --combine-trainval
CUDA_VISIBLE_DEVICES=6,7 python3 PCB.py --train -d dukemtmc --logs-dir logs/pcb_new/64/dukemtmc/basis_crop --height 384 -s 1 --features 64 --output_feature fc --re 0.5 --combine-trainval --crop

CUDA_VISIBLE_DEVICES=6,7 python3 PCB.py --train -d duke_tracking --logs-dir logs/pcb_new/64/duke_tracking/train/1_fps/basis_crop --height 384 -s 1 --features 64 --output_feature fc --tracking_fps 1 --fix_bn 0 --re 0.5 --crop

# reid feat
CUDA_VISIBLE_DEVICES=6,7 python3 save_cnn_feature.py  -a pcb -b 128 --resume logs/pcb_new/64/duke_tracking/train/1_fps/basis_crop/model_best.pth.tar --features 64 --output_feature fc --l0_name pcb_basis_crop_fc64_train_1fps -s 1 --height 384 -d gt_test
# det feat
CUDA_VISIBLE_DEVICES=6,7 python3 save_cnn_feature.py  -a pcb -b 128 --resume logs/pcb_new/64/duke_tracking/train/1_fps/basis_crop/model_best.pth.tar --features 64 --output_feature fc --l0_name pcb_basis_crop_fc64_train_1fps -s 1 --height 384 -d detections --det_time trainval
# gt feat
CUDA_VISIBLE_DEVICES=6,7 python3 save_cnn_feature.py  -a pcb -b 128 --resume logs/pcb_new/64/duke_tracking/train/1_fps/basis_crop/model_best.pth.tar --features 64 --output_feature fc --l0_name pcb_basis_crop_fc64_train_1fps -s 1 --height 384 -d gt_all --det_time trainval
