# Open-ReID-tracking

This repo is based on Cysu's [open-reid](https://github.com/Cysu/open-reid), which is a great re-ID library. For performance, we implemented some other baseline models on top of it. For utility, we add some function for the tracking-by-detection workflow in tracking works. 

- update all models for performance & readability. 
- add ```data/README.md```. check for folder structure & dataset download. 
- add ```requirements.txt```. use ```conda install --file requirements.txt``` to install. 
- add BN after feature layer in ```reid/models/IDE_model.py``` for separation. This introduces a higher performance.
- fix high cpu usage via adding ```os.environ['OMP_NUM_THREADS'] = '1'``` in runable files. 
- NEW: We adopt a baseline from Hao Luo \[[git](https://github.com/michuanhaohao/reid-strong-baseline), [paper](https://arxiv.org/abs/1903.07071)\]. See ```ZJU.py```. We achieve competitive performance with the same `IDE_model.py`. 

Please use this repo alongside with our flavor of [DeepCC](https://github.com/hou-yz/DeepCC_aic) tracker for tracking. 

## Model
- IDE \[[paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)\]
- Triplet \[[paper](https://arxiv.org/abs/1703.07737)\]
- PCB \[[git](https://github.com/syfafterzy/PCB_RPP_for_reID), [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yifan_Sun_Beyond_Part_Models_ECCV_2018_paper.pdf)\]
- ZJU \[[git](https://github.com/michuanhaohao/reid-strong-baseline), [paper](https://arxiv.org/abs/1903.07071)\]


## Data
The re-ID datasets should be stored in a file structure like this:
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
    │   │ ...
    │
    └───DukeMTMC-reID
    │   │ ...
    │
    └───Market-1501-v15.09.15
        │ ...
```


## Usage
### Re-ID
training from scratch
```shell script
CUDA_VISIBLE_DEVICES=0 python3 IDE.py -d market1501 --train
```
this will automatically save your logs at `./logs/ide/market1501/YYYY-MM-DD_HH-MM-SS`, where `YYYY-MM-DD_HH-MM-SS` is the time stamp when the training started. 

resume & evaluate
```shell script
CUDA_VISIBLE_DEVICES=0 python3 IDE.py -d market1501 --resume YYYY-MM-DD_HH-MM-SS
```

### Feature Extraction for Tracking (to be updated)
We describe the workflow for a simple model. For the full ensemble model, please check 

First, please use the following to extract detection bounding boxes from videos.
```shell script
python3 reid/prepare/extract_bbox.py
```

Next, train the baseline on re-ID data from AI-City 2019 (track-2). 
```shell script
# train
CUDA_VISIBLE_DEVICES=0,1 python3 ZJU.py --train -d aic_reid --logs-dir logs/ZJU/256/aic_reid/lr001_colorjitter --colorjitter  --height 256 --width 256 --lr 0.01 --step-size 30,60,80 --warmup 10 --LSR --backbone densenet121 --features 256 --BNneck -s 1 -b 64 --epochs 120
```
Then, the detection bounding box feature are computed. 
```shell script
# gt feat (optional)
# CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/256/aic_reid/lr001_colorjitter/model_best.pth.tar --features 256 --height 256 --width 256 --l0_name zju_lr001_colorjitter_256 --BNneck -s 1 -d aic --type gt_all -b 64
# reid feat (parameter tuning, see DeepCC_aic)
CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/256/aic_reid/lr001_colorjitter/model_best.pth.tar --features 256 --height 256 --width 256 --l0_name zju_lr001_colorjitter_256 --BNneck -s 1 -d aic --type gt_mini -b 64
# det feat (tracking pre-requisite, see DeepCC_aic)
CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/256/aic_reid/lr001_colorjitter/model_best.pth.tar --features 256 --height 256 --width 256 --l0_name zju_lr001_colorjitter_256 --BNneck -s 1 -d aic --type detections --det_time trainval -b 64
CUDA_VISIBLE_DEVICES=0,1 python3 save_cnn_feature.py -a zju --backbone densenet121 --resume logs/ZJU/256/aic_reid/lr001_colorjitter/model_best.pth.tar --features 256 --height 256 --width 256 --l0_name zju_lr001_colorjitter_256 --BNneck -s 1 -d aic --type detections --det_time test -b 64
```

## Implementation details

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


`Default` Settings:
- IDE 
  - `stride = 2` in last conv block.
  - `h x w = 256 x 128`.
  - random horizontal flip + random crop.
- Triplet
  - `stride = 2` in last conv block.
  - `h x w = 256 x 128`.
  - random horizontal flip + random crop.
- PCB
  - `stride = 1` in last conv block.
  - `h x w = 384 x 128`.
  - random horizontal flip.
- ZJU
  - cross entropy + triplet.
  - `ims_per_id = 4`, `ids_per_batch = 16`.
  - `h x w = 256 x 128`.
  - warmup for 10 epochs.
  - random horizontal flip + pad 10 pixel then random crop + random erasing with `re = 0.5`.
  - label smooth.
  - `stride = 1` in last conv block.
  - ~~BNneck.~~
  - ~~center loss.~~

`Tracking` settings for IDE, Triplet, and PCB:
- `stride = 1` in last conv block.
- `h x w = 384 x 128`.
- horizontal flipping + Random Erasing with `re = 0.5`.

`Raw` setting for ZJU:
  - cross entropy + triplet.
  - `ims_per_id = 4`, `ids_per_batch = 16`.
  - `h x w = 256 x 128`.
  - random horizontal flip + pad 10 pixel then random crop.



## Experiment Results

| dataset | model  | settings                        | mAP (%) | Rank-1 (%) |
| ---     | ---    | ---                             | :---: | :---: |
| Duke|IDE|Default                                   | 58.70 | 77.56 |
| Duke|Triplet|Default                               | 62.40 | 78.19 |
| Duke|PCB|Default                                   | 68.72 | 83.12 |
| Duke|ZJU|Default                                   | 75.20 | 86.71 |
| Market|IDE|Default                                   | 69.34 | 86.58 |
| Market|Triplet|Default                               | 72.42 | 86.55 |
| Market|PCB|Default                                   | 77.53 | 92.52 |
| Market|ZJU|Default                                   | 85.37 | 93.79 |

<!---
| Duke|IDE|Default                                | 51.65 | 71.10 |
| Duke|IDE|Tracking w/ crop                      | 58.05 | 75.63 |
| Duke|IDE|Tracking                              | 62.93 | 79.67 |
| Duke|Triplet|Default                                     | 59.76 | 76.26 |
| Duke|Triplet|Tracking w/ crop                           | 63.50 | 78.19 |
| Duke|Triplet|Tracking                                   | 66.44 | 81.33 |
| Duke|PCB|Default' (Tracking w/o RE)                | 68.41 | 83.12 |
| Duke|PCB|Default' + fc64                        | 68.06 | 82.76 |
| Duke|PCB|Default' + NOT normalizing stripes     | 66.01 | 83.17 |
| Duke|PCB|Tracking                              | 68.70 | 82.81 |
| Duke|PCB|Tracking + fc64                       | 68.59 | 82.85 |
-->
