## About

This repository contains a PyTorch Lightning implementation for a cloud classification task.
The goal is to classify images of clouds into seven different categories using deep learning techniques.

## Todos
- [x] find labels
- [x] fix data load
- [x] test training (pipeline)
- [ ] improve accuracy / double check validation

## Install

install pytorch with cuda support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118     

## Run

python train.py --epochs 10 --tag 'test1'

## Data

dataset: https://www.kaggle.com/competitions/cloud-type-classification2/data

seven sky conditions are considered, namely, cirriform clouds, high cumuliform clouds, stratocumulus clouds, cumulus clouds, cumulonimbus clouds, stratiform clouds, and clear sky.

 file descriptions:
- train.csv - the training set
- test.csv - the test set
- submit.csv - a sample submission file in the correct format

data fields:
- id - an image name
- label - a label of the image
- predict - a result of the image prediction

## Modules

### Lightning

1. Install `conda install lightning -c conda-forge`
2. [Get started](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)

### TensorBoard

By default, Lightning automatically uses TensorBoard logger, and stores the logs to a 'lightning_logs/' directory.
Install `conda install conda-forge::tensorboard`.
Then run this on your commandline `tensorboard --logdir=lightning_logs/` and and open your browser to http://localhost:6006/.
By default, Lightning logs every 50 steps. Use Trainer flags to Control Logging Frequency.

## Ressources

- https://docs.pytorch.org/vision/main/datasets.html
- https://lightning.ai/docs/pytorch/stable/starter/introduction.html
- https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html
- https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
- https://lightning.ai/docs/pytorch/stable/common/early_stopping.html
- https://lightning.ai/docs/pytorch/stable/visualize/logging_basic.html
- https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html
- https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html
- https://www.comet.com/site/