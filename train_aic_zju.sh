#!/usr/bin/env bash
# train
CUDA_VISIBLE_DEVICES=0,1 python3 ZJU_baseline.py --train -d aic_reid --logs-dir logs/ZJU/1024/aic_reid/lr001_3steps_hw256_warmup10_lsr_densenet121_feat1024_s1_batch64 --height 256 --width 256 --lr 0.01 --step-size 30,60,80 --warmup 10 --LSR --backbone densenet121 --features 1024 --BNneck -s 1 -b 64 --epochs 120
CUDA_VISIBLE_DEVICES=0,1 python3 ZJU_baseline.py --train -d aic_reid --logs-dir logs/ZJU/1024/aic_reid/lr001_softmargin --softmargin  --height 256 --width 256 --lr 0.01 --step-size 30,60,80 --warmup 10 --LSR --backbone densenet121 --features 1024 --BNneck -s 1 -b 64 --epochs 120
CUDA_VISIBLE_DEVICES=0,1 python3 ZJU_baseline.py --train -d aic_reid --logs-dir logs/ZJU/1024/aic_reid/lr001_colorjitter --colorjitter  --height 256 --width 256 --lr 0.01 --step-size 30,60,80 --warmup 10 --LSR --backbone densenet121 --features 1024 --BNneck -s 1 -b 64 --epochs 120

# reid feat
CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/1024/aic_reid/lr001_3steps_hw256_warmup10_lsr_densenet121_feat1024_s1_batch64/model_best.pth.tar --features 1024 --height 256 --width 256 --l0_name zju_lr001 --BNneck -s 1 -d aic --type gt_mini -b 64
CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/1024/aic_reid/lr001_softmargin/model_best.pth.tar --features 1024 --height 256 --width 256 --l0_name zju_lr001_softmargin --BNneck -s 1 -d aic --type gt_mini -b 64
CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/1024/aic_reid/lr001_colorjitter/model_best.pth.tar --features 1024 --height 256 --width 256 --l0_name zju_lr001_colorjitter --BNneck -s 1 -d aic --type gt_mini -b 64

# det feat
CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/1024/aic_reid/lr001_3steps_hw256_warmup10_lsr_densenet121_feat1024_s1_batch64/model_best.pth.tar --features 1024 --height 256 --width 256 --l0_name zju_lr001 --BNneck -s 1 -d aic --type detections --det_time trainval -b 64
CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/1024/aic_reid/lr001_3steps_hw256_warmup10_lsr_densenet121_feat1024_s1_batch64/model_best.pth.tar --features 1024 --height 256 --width 256 --l0_name zju_lr001 --BNneck -s 1 -d aic --type detections --det_time test -b 64
CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/1024/aic_reid/lr001_softmargin/model_best.pth.tar --features 1024 --height 256 --width 256 --l0_name zju_lr001_softmargin --BNneck -s 1 -d aic --type detections --det_time trainval -b 64
CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/1024/aic_reid/lr001_softmargin/model_best.pth.tar --features 1024 --height 256 --width 256 --l0_name zju_lr001_softmargin --BNneck -s 1 -d aic --type detections --det_time test -b 64
CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/1024/aic_reid/lr001_colorjitter/model_best.pth.tar --features 1024 --height 256 --width 256 --l0_name zju_lr001_colorjitter --BNneck -s 1 -d aic --type detections --det_time trainval -b 64
CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/1024/aic_reid/lr001_colorjitter/model_best.pth.tar --features 1024 --height 256 --width 256 --l0_name zju_lr001_colorjitter --BNneck -s 1 -d aic --type detections --det_time test -b 64

# ensemble
python3 reid/prepare/ensemble.py