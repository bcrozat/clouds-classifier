# TODO: install & use lightning for training

# Import dependencies
import argparse # Allows to provide input parameters (parse arguments) from the command line instead of hardcoding them
import time
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# Import custom modules
from model import LitCloudNet
from dataset import dataloader_train, dataloader_test
#from utils import save_model, save_acc_plot, save_loss_plot
import lightning as L

# Set up argument parser
parser = argparse.ArgumentParser() # Initialize argument parser
parser.add_argument('-e', '--epochs', type=int, default=10,
    help='number of epochs to train the model for')
parser.add_argument('-t', '--tag', type=str, default=10,
    help='model tag to save')
args = vars(parser.parse_args())

# Parameters
learning_rate = 1e-3
epochs = args['epochs']
model = LitCloudNet()
tag = args['tag']
print(model)
print(f'Tag: {tag}')

# Initialize the model


# Train the model (using Lightning)
trainer = L.Trainer(limit_train_batches=100,
                    max_epochs=1,
                    accelerator='gpu'
                    )
trainer.fit(model=model, train_dataloaders=train_loader)