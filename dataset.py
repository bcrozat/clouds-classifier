# Import dependencies
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

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
dataset_train = ImageFolder(
    root=r'D:\Data\cloud-type-classification2\images\train',
    transform=train_transforms,
)

dataset_test = ImageFolder(
    root=r'D:\Data\cloud-type-classification2\images\test',
    transform=test_transforms,
)

# Load data
dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=64)
dataloader_test = DataLoader(dataset_test, shuffle=True, batch_size=64)
image, label = next(iter(dataloader_train))
print(image.shape)  # torch.Size([1, 3, 128, 128])
image = image.squeeze().permute(1, 2, 0)
print(image.shape)  # torch.Size([128, 128, 3])
plt.imshow(image)
plt.show()

# Indicate end
print('End.')
