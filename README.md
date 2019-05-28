# Open-ReID-tracking

This repo is based on Cysu's [open-reid](https://github.com/Cysu/open-reid), which is a great re-ID library. For performance, we implemented some other baseline models on top of it. For utility, we add some function for the tracking-by-detection workflow in tracking works. 

# AI-City 2019 update

Please use this repo alongside with our flavor of [DeepCC](https://github.com/hou-yz/DeepCC_aic) tracker for tracking. 

### Model
We adopt a new baseline from Hao Luo \[[git](https://github.com/michuanhaohao/reid-strong-baseline), [paper](https://arxiv.org/abs/1903.07071)\] and make a few modification on top of it.

### Data
AI-City datasets (track-1, track-2) and VeRi dataset are needed for the following steps.
They should be stored in a file structure like this:
```
~
└───Data
    └───AIC19
    │   │ track-1 data
    │   │ ...
    │
    └───AIC19-reid
    │   │ track-2 data
    │   │ ...
    │
    └───VeRi
        │ VeRi data
        │ ...
```


### Training & Tracking Pre-requisite
We describe the workflow for a simple model. For the full ensemble model, please check 

First, please use the following to extract detection bounding boxes from videos.
```angular2html
cd open-reid
python3 reid/prepare/extract_bbox.py
```

Next, train the baseline on re-ID data from AI-City 2019 (track-2). 
```angular2html
# train
CUDA_VISIBLE_DEVICES=0,1 python3 ZJU_baseline.py --train -d aic_reid --logs-dir logs/ZJU/256/aic_reid/lr001_colorjitter --colorjitter  --height 256 --width 256 --lr 0.01 --step-size 30,60,80 --warmup 10 --LSR --backbone densenet121 --features 256 --BNneck -s 1 -b 64 --epochs 120
```
Then, the detection bounding box feature are computed. 
```angular2html
# gt feat (optional)
# CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/256/aic_reid/lr001_colorjitter/model_best.pth.tar --features 256 --height 256 --width 256 --l0_name zju_lr001_colorjitter_256 --BNneck -s 1 -d aic --type gt_all -b 64
# reid feat (parameter tuning, see DeepCC_aic)
CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/256/aic_reid/lr001_colorjitter/model_best.pth.tar --features 256 --height 256 --width 256 --l0_name zju_lr001_colorjitter_256 --BNneck -s 1 -d aic --type gt_mini -b 64
# det feat (tracking pre-requisite, see DeepCC_aic)
CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/256/aic_reid/lr001_colorjitter/model_best.pth.tar --features 256 --height 256 --width 256 --l0_name zju_lr001_colorjitter_256 --BNneck -s 1 -d aic --type detections --det_time trainval -b 64
CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/256/aic_reid/lr001_colorjitter/model_best.pth.tar --features 256 --height 256 --width 256 --l0_name zju_lr001_colorjitter_256 --BNneck -s 1 -d aic --type detections --det_time test -b 64
```


# Our Baselines and Performance

### IDE baseline
training IDE from scratch
```angular2html
cd open-reid
python3 IDE.py --train -d dukemtmc  --combine-trainval --logs-dir logs/ide/dukemtmc/raw
```


testing & evaluating
```angular2html
python3 IDE.py --evaluate -d dukemtmc --resume logs/ide/dukemtmc/raw/model_best.pth.tar > eval_IDE.log
```


### PCB Model

Added PCB model support.

See `PCB.py` and `reid/models/PCB_model.py`. also modified `reid/trainers.py` for PCB training and  `reid/feature_extraction/cnn.py` for PCB evaluating.

training PCB from scratch
```angular2html
cd open-reid
python3 PCB.py --train -d dukemtmc  --combine-trainval --logs-dir logs/pcb/dukemtmc/raw
```

testing & evaluating
```angular2html
python3 PCB.py --evaluate -d dukemtmc --resume logs/pcb/dukemtmc/raw/model_best.pth.tar > eval_PCB.log
```


## Current Results

Cross-entropy loss:
- `batch_size = 64`.
- `learning rate = 0.1`, step decay after 40 epochs. Train for 60 epochs in total.
- 0.1x learning rate for `resnet-50` base.
- `weight decay = 5e-4`.
- SGD optimizer, `momentum = 0.9`, `nestrov = true`.

Triplet loss:
- `margin=0.3`.
- `ims_per_id = 4`, `ids_per_batch = 32`.
- `learning rate = 2e-4`, exponentially decay after 150 epochs. Train for 300 epochs in total.
- unifide learning rate for `resnet-50` base and `fc` feature layer.
- `weight decay = 5e-4`.
- Adam optimizer.


`Raw` settings:
- `stride = 2` in last conv block.
- `im_w x im_h = 128 x 256`.
- only horizontal flipping used for data augmentation.

`Basis` settings:
- `stride = 1` in last conv block.
- `im_w x im_h = 128 x 384`.
- horizontal flipping and Random Erasing with `re = 0.5` used for data augmentation.

~~PCB model normalizes feature to unit length.~~

The results are as follows. 

| dataset | model | loss | settings                        | mAP (%) | Rank-1 (%) |
| ---     | ---   | ---  | ---                             | :---: | :---: |
| Duke|IDE|Triplet|Raw                                     | 59.76 | 76.26 |
| Duke|IDE|Triplet|Basis w/ crop                           | 63.50 | 78.19 |
| Duke|IDE|Triplet|Basis                                   | 66.44 | 81.33 |
| Duke|IDE|CrossEntropy|Raw                                | 51.65 | 71.10 |
| Duke|IDE|CrossEntropy|Basis w/ crop                      | 58.05 | 75.63 |
| Duke|IDE|CrossEntropy|Basis                              | 62.93 | 79.67 |
| Duke|PCB|CrossEntropy|Raw' (Basis w/o RE)                | 68.41 | 83.12 |
| Duke|PCB|CrossEntropy|Raw' + fc64                        | 68.06 | 82.76 |
| Duke|PCB|CrossEntropy|Raw' + NOT normalizing stripes     | 66.01 | 83.17 |
| Duke|PCB|CrossEntropy|Basis                              | 68.70 | 82.81 |
| Duke|PCB|CrossEntropy|Basis + fc64                       | 68.59 | 82.85 |


**`stride = 1` (higher spatial resolution before global pooling) higher performance than `stride = 2` (original ResNet). This is from paper [Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349).**
