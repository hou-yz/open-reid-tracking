# Open-ReID

Open-ReID is a lightweight library of person re-identification for research
purpose. It aims to provide a uniform interface for different datasets, a full
set of models and evaluation metrics, as well as examples to reproduce (near)
state-of-the-art results.

## PCB Model

Added PCB model support with faster evaluation.

See `/examples/PCB_n_RPP.py` and `/reid/models/PCB_model.py`. also modified `/reid/trainers.py` for PCB training and  `/reid/feature_extraction/cnn.py` for PCB evaluating.

training PCB from scratch
```angular2html
cd open-reid
python3 ./examples/PCB_n_RPP.py --train-PCB -d market1501  --combine-trainval --logs-dir logs/pcb/market1501
```

training RPP based on trained PCB model
```angular2html
python3 ./examples/PCB_n_RPP.py --train-RPP -d market1501 --resume logs/pcb/market1501/model_best.pth.tar --combine-trainval --logs-dir logs/pcb_n_rpp/market1501
```

testing & evaluating
```angular2html
python3 ./examples/PCB_n_RPP.py --evaluate -d market1501 --resume logs/pcb/market1501/model_best.pth.tar --combine-trainval --logs-dir logs/pcb/market1501
python3 ./examples/PCB_n_RPP.py --evaluate -d market1501 --resume logs/pcb_n_rpp/market1501/model_best.pth.tar --combine-trainval --logs-dir logs/pcb_n_rpp/market1501
```


## Installation

Install [PyTorch](http://pytorch.org/) (version >= 0.2.0). Although we support
both python2 and python3, we recommend python3 for better performance.

```shell
git clone https://github.com/Cysu/open-reid.git
cd open-reid
python setup.py install
```

## Examples

```shell
python examples/softmax_loss.py -d viper -b 64 -j 2 -a resnet50 --logs-dir logs/softmax-loss/viper-resnet50
```

This is just a quick example. VIPeR dataset may not be large enough to train a deep neural network.

Check about more [examples](https://cysu.github.io/open-reid/examples/training_id.html)
and [benchmarks](https://cysu.github.io/open-reid/examples/benchmarks.html).
