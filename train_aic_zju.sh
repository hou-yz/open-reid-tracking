#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python3 ZJU_baseline.py --train -d aic_reid --logs-dir logs/ZJU/1024/aic_reid/lr0001_3steps_hw256_warmup10_lsr_densenet121_feat1024_s1_batch32 --height 256 --width 256 --lr 0.001 --step-size 40,60,80 --warmup 10 --LSR --arch densenet121 --features 1024 --BNneck -s 1 -b 32
