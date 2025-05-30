# Import dependencies
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from tqdm.auto import tqdm
import argparse # Allows to provide input parameters (parse arguments) from the command line instead of hardcoding them
import lightning as L
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Import custom modules
from model import LitCloudNet
from dataset import train_dataloader, test_dataloader

# Start timer
start_time = time.time()

# Create timestamp
timestamp = time.strftime('%Y-%m-%d-%H%M%S', time.localtime(start_time))

# Print start message
print('# Training started')

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
tag = args['tag']

# Initialize the model
model = LitCloudNet()
print(f'Model: {model}')

# Train the model (using Lightning)
trainer = L.Trainer(default_root_dir='logs',
                    limit_train_batches=100,
                    max_epochs=1,
                    accelerator='gpu',
                    callbacks=[EarlyStopping(monitor='acc_loss', mode='max', patience=3)] # Early stopping after 3 iterations without improvement
                    )
trainer.fit(model=model, train_dataloaders=train_dataloader)

# Print log message
print('# Training complete!')

# Print log message
print('# Testing started!')

# Test the model
trainer.test(model=model, train_dataloaders=test_dataloader)

# Print log message
print('# Testing complete!')
