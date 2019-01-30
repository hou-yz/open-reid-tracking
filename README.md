# Open-ReID

Open-ReID is a lightweight library of person re-identification for research
purpose. It aims to provide a uniform interface for different datasets, a full
set of models and evaluation metrics, as well as examples to reproduce (near)
state-of-the-art results.

## IDE baseline
training IDE from scratch
```angular2html
cd open-reid
python3 IDE.py --train -d market1501  --combine-trainval --logs-dir logs/ide/market1501
```


testing & evaluating
```angular2html
python3 IDE.py --evaluate -d market1501 --resume logs/ide/market1501/model_best.pth.tar --combine-trainval > eval_IDE.log
```


## PCB Model

Added PCB model support with faster evaluation.

See `/PCB_n_RPP.py` and `/reid/models/PCB_model.py`. also modified `/reid/trainers.py` for PCB training and  `/reid/feature_extraction/cnn.py` for PCB evaluating.

training PCB from scratch
```angular2html
cd open-reid
python3 PCB_n_RPP.py --train-PCB -d market1501  --combine-trainval --logs-dir logs/pcb/market1501
```

training RPP based on trained PCB model
```angular2html
python3 PCB_n_RPP.py --train-RPP -d market1501 --resume logs/pcb/market1501/model_best.pth.tar --combine-trainval --logs-dir logs/pcb_n_rpp/market1501
```

testing & evaluating
```angular2html
python3 PCB_n_RPP.py --evaluate -d market1501 --resume logs/pcb/market1501/model_best.pth.tar --combine-trainval > eval_PCB.log
python3 PCB_n_RPP.py --evaluate -d market1501 --resume logs/pcb_n_rpp/market1501/model_best.pth.tar --combine-trainval > eval_RPP.log
```


# Current Results

IDE-Raw with settings:
- IDE model, cross-entropy loss
- `stride = 2` in last conv block, NOT normalizing feature to unit length
- `im_w x im_h = 128 x 256`, `batch_size = 64`
- Only horizontal flipping used for data augmentation
- SGD optimizer, 0.1x learning rate for `resnet-50` base, base learning rate 0.1, `momentum = 0.9`, `nestrov = true`, step decaying after 40 epochs. Train for 60 epochs in total.


`Basis` settings:
- `stride = 1` in last conv block, NOT normalizing feature to unit length
- `im_w x im_h = 128 x 384`
- Horizontal flipping and Random Erasing with `re = 0.5` used for data augmentation

Triplet loss with settings:
- IDE model, `stride = 2` or `stride = 1` in last conv block
- NOT normalizing feature to unit length, with margin 0.3
- Only horizontal flipping used for data augmentation
- `im_w x im_h = 128 x 256`, `ims_per_id = 4`, `ids_per_batch = 32`
- Adam optimizer, unifide learning rate for `resnet-50` base and `fc` feature layer , base learning rate 2e-4, decaying exponentially after 150 epochs. Train for 300 epochs in total.

The results are as follows. 

|                       | mAP (%) | Rank-1 (%) |
| ---                   | :---: | :---: |
| Duke-Triplet-S2       | 59.76 | 76.26 |
| Duke-Triplet-S1       | 63.58 | 78.10 |
| Duke-Triplet-Basis    | 66.44 | 81.33 |
| Duke-IDE-Raw          | 51.65 | 71.10 |
| Duke-IDE-Basis        | 62.93 | 79.67 |

**We see that `stride = 1` (higher spatial resolution before global pooling) has obvious improvement over `stride = 2` (original ResNet). I tried this inspired by paper [Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349).**
