## About

This project aims training and deploying a machine learning model to classify clouds.

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

1. Install `conda install conda-forge::tensorboard`
2. Visualize training
If you have tensorboard installed, you can use it for visualizing experiments.
Run this on your commandline and open your browser to http://localhost:6006/
`tensorboard --logdir`.

## Ressources

- https://lightning.ai/docs/pytorch/stable/starter/introduction.html