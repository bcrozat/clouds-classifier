# Import libraries
import torch.nn as nn
import torch.optim as optim
import lightning as L
import torch
from torchmetrics import Accuracy

# Parameters
learning_rate = 1e-3
num_classes = 7

# Define the LightningModule
class LitCloudNet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64 * 16 * 16, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        z = self.feature_extractor(x)
        x_hat = self.classifier(z)
        train_loss = self.criterion(x_hat, x) # nn.CrossEntropyLoss() for multi-class classification, nn.BCELoss() for binary classification
        train_acc = self.acc()
        preds = torch.argmax(x_hat, dim=1)
        train_acc.update(preds, y)
        self.log_dict({'train_loss': train_loss, 'train_acc': self.train_acc.compute()}, prog_bar=True) # Logging to TensorBoard
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        z = self.feature_extractor(x)
        x_hat = self.classifier(z)
        val_loss = self.criterion(x_hat, x)
        val_acc = self.acc()
        preds = torch.argmax(x_hat, dim=1)
        val_acc.update(preds, y)
        self.log_dict({'val_loss': val_loss, 'val_acc': self.val_acc.compute()}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer