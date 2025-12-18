import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import random

# Define augmentation transforms (light)
transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.ToTensor()
])

# Load FashionMNIST test set (no normalization for visualization)
dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=None
)

# Sample 100 random indices
indices = random.sample(range(len(dataset)), 100)

# Prepare transforms
aug_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.ToTensor()
])
to_tensor = transforms.ToTensor()

# Collect both original and augmented images
imgs_orig = []
imgs_aug = []
for idx in indices:
    img, _ = dataset[idx]
    imgs_orig.append(to_tensor(img))
    imgs_aug.append(aug_transform(img))
imgs_tensor_orig = torch.stack(imgs_orig)  # (100, 1, 28, 28)
imgs_tensor_aug = torch.stack(imgs_aug)    # (100, 1, 28, 28)

# Save both grids
save_image(imgs_tensor_orig, "fashionmnist_grid_original.png", nrow=10, padding=2, normalize=True)
save_image(imgs_tensor_aug, "fashionmnist_grid_augmented.png", nrow=10, padding=2, normalize=True)
