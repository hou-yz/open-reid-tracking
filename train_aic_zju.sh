#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=1 python3 ZJU_baseline.py --train -d aic_reid --logs-dir logs/ZJU/1024/aic_reid/lr0001_3steps_hw256_warmup10_lsr_densenet121_feat1024_s1_batch32_model-3-1_epoch80 --height 256 --width 256 --lr 0.001 --step-size 40,60,80 --warmup 10 --LSR --backbone densenet121 --features 1024 --BNneck -s 1 -b 32 --epochs 80

# reid feat
#CUDA_VISIBLE_DEVICES=1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/1024/aic_reid/best_lr0001_3steps_hw256_warmup10_lsr_densenet121_feat1024_s1_batch32/model_best.pth.tar --features 1024 --height 256 --width 256 --l0_name zju_best --BNneck -s 1 -d aic --type gt_mini -b 32
# det feat
CUDA_VISIBLE_DEVICES=1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/1024/aic_reid/best_lr0001_3steps_hw256_warmup10_lsr_densenet121_feat1024_s1_batch32/model_best.pth.tar --features 1024 --height 256 --width 256 --l0_name zju_best --BNneck -s 1 -d aic --type detections --det_time val -b 32