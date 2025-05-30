## About


## Todos

- [ ] find labels
- [ ] fix data load
- [ ] test training

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

- https://lightning.ai/docs/pytorch/stable/starter/introduction.html
- https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html
- https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
- https://lightning.ai/docs/pytorch/stable/common/early_stopping.html
- https://lightning.ai/docs/pytorch/stable/visualize/logging_basic.html
- https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html
- https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html
- https://www.comet.com/site/