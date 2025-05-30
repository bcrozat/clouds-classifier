# Import dependencies
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Print start message
print('# Loading data...')

# Define transformations
train_transforms = transforms.Compose([
    # transforms.RandomHorizontalFlip(), # Simulate different viewpoints
    # transforms.RandomRotation(45), # Simulate different angles of cloud formations
    # transforms.RandomAutocontrast(), # Simulate different lighting conditions
    # Maybe change color to simulate different time & light reflections
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
])

# Load dataset
train_dataset = ImageFolder(
    root=r'D:\Data\cloud-type-classification2\images\train',
    transform=train_transforms,
)

test_dataset = ImageFolder(
    root=r'D:\Data\cloud-type-classification2\images\test',
    transform=test_transforms,
)

# Create validation dataset (20% of the train dataset)
# train_dataset_size = int(len(train_dataset) * 0.8)
# val_dataset_size = len(train_dataset) - train_dataset_size
# seed = torch.Generator().manual_seed(1)
# train_dataset, val_dataset = random_split(train_dataset, [train_dataset_size, val_dataset_size], generator=seed)

# Load data
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=64)

# Check dataset
# image, label = next(iter(train_dataloader))
# print(image.shape)  # torch.Size([1, 3, 128, 128])
# image = image.squeeze().permute(1, 2, 0)
# print(image.shape)  # torch.Size([128, 128, 3])
# plt.imshow(image)
# plt.show()

# Print end message
print('# Data loaded.')
