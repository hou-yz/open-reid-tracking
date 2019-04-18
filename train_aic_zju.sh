#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python3 ZJU_baseline.py --train -d aic_reid --logs-dir logs/ZJU/2048/aic_reid/basis_S --height 224 --width 224 -s 1 --lr 0.001
