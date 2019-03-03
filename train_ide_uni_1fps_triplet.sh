#!/usr/bin/env bash
#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=6,7 python3 IDE_triplet.py --train -d dukemtmc --logs-dir logs/ide_triplet/fc256/dukemtmc/raw --features 256 --output_feature fc --combine-trainval
CUDA_VISIBLE_DEVICES=6,7 python3 IDE_triplet.py --train -d dukemtmc --logs-dir logs/ide_triplet/fc256/dukemtmc/basis_crop --height 384 -s 1 --features 256 --output_feature fc --tracking_fps 1 --fix_bn 0 --re 0.5 --combine-trainval --crop
CUDA_VISIBLE_DEVICES=6,7 python3 IDE_triplet.py --train -d duke_tracking --logs-dir logs/ide_triplet/fc256/duke_tracking/train/1_fps/basis_crop --height 384 -s 1 --features 256 --output_feature fc --tracking_fps 1 --fix_bn 0 --re 0.5 --crop

# reid feat
CUDA_VISIBLE_DEVICES=6,7 python3 save_cnn_feature.py  -a ide -b 128 --resume logs/ide_triplet/fc256/duke_tracking/train/1_fps/basis_crop/model_best.pth.tar --features 256 --output_feature fc --l0_name ide_triplet_basis_crop_train_1fps -s 1 --height 384 -d gt_test
# det feat
CUDA_VISIBLE_DEVICES=6,7 python3 save_cnn_feature.py  -a ide -b 128 --resume logs/ide_triplet/fc256/duke_tracking/train/1_fps/basis_crop/model_best.pth.tar --features 256 --output_feature fc --l0_name ide_triplet_basis_crop_train_1fps -s 1 --height 384 -d detections --det_time trainval
# gt feat
CUDA_VISIBLE_DEVICES=6,7 python3 save_cnn_feature.py  -a ide -b 128 --resume logs/ide_triplet/fc256/duke_tracking/train/1_fps/basis_crop/model_best.pth.tar --features 256 --output_feature fc --l0_name ide_triplet_basis_crop_train_1fps -s 1 --height 384 -d gt_all --det_time trainval
