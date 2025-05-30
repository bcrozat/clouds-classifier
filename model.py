# Import libraries
import torch.nn as nn
import torch.optim as optim
import lightning as L

# Parameters
learning_rate = 1e-3
num_classes = 7

# Define the LightningModule
class LitCloudNet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
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

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.feature_extractor(x)
        x_hat = self.classifier(z)
        loss = nn.BCELoss(x_hat, x) # nn.CrossEntropyLoss() for multi-class classification, nn.BCELoss() for binary classification
        self.log('train_loss', loss) # Logging to TensorBoard
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.feature_extractor(x)
        x_hat = self.classifier(z)
        test_loss = nn.BCELoss(x_hat, x)
        self.log('test_loss', test_loss)

    # def validation_step(self, batch, batch_idx):
    #     x, _ = batch
    #     x = x.view(x.size(0), -1)
    #     z = self.feature_extractor(x)
    #     x_hat = self.classifier(z)
    #     val_loss = nn.BCELoss(x_hat, x)
    #     self.log('val_loss', val_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer