# Import dependencies
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision import transforms

# Print start message
print('# Loading data...')

# Define transformations
train_transforms = transforms.Compose([
    # transforms.RandomHorizontalFlip(), # Simulate different viewpoints
    # transforms.RandomRotation(45), # Simulate different angles of cloud formations
    # transforms.RandomAutocontrast(), # Simulate different lighting conditions
    # Maybe change color to simulate different time & light reflections
    # transforms.ToTensor(), # No need to convert to tensor since read_image already returns one
    transforms.Resize((128, 128)),
])

test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
])

# Define custom dataset
class CloudDataset(Dataset):
    def __init__(self, annotations, root, transform=None, target_transform=None):
        self.labels = pd.read_csv(annotations)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = os.path.join(self.root, self.labels.iloc[index, 0])
        image = read_image(path) # Returns a tensor
        label = self.labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Load dataset
train_dataset = CloudDataset(
    annotations=r'data/train.csv',
    root='data/train',
    transform=train_transforms
)

# Create validation dataset
train_dataset_size = int(len(train_dataset) * 0.8) # 20% of train dataset
val_dataset_size = len(train_dataset) - train_dataset_size
seed = torch.Generator().manual_seed(1)
train_dataset, val_dataset = random_split(train_dataset, [train_dataset_size, val_dataset_size], generator=seed)

# train_dataset = ImageFolder(
#     root=r'data/train',
#     train=True,
#     transform=train_transforms
# )

# test_dataset = ImageFolder(
#     root=r'data/test',
#     train=False,
#     transform=test_transforms
# )

# Load data
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=64)

# Check dataset
image, label = next(iter(train_dataloader))
print(image.shape)  # torch.Size([batch_size, 3, 128, 128]) with batch_size=64
image = image[0].permute(1, 2, 0) # Selects first image to test and permutes dimensions to (H, W, C)
print(image.shape)  # torch.Size([128, 128, 3])
plt.imshow(image)

# Print end message
print('# Data loaded.')
