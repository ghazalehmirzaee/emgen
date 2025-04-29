import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    def __init__(self, train=True, normalize_for_gan=False):
        """
        MNIST dataset implementation.

        Args:
            train (bool): Whether to use training or test set
            normalize_for_gan (bool): If True, normalize to [-1,1] for GAN training
        """
        if normalize_for_gan:
            # Transform for GAN training ([-1, 1] range)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            # Original transform (0-1 range)
            transform = transforms.ToTensor()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.mnist = datasets.MNIST(
            root=os.path.join(current_dir, "data"),
            train=train,
            transform=transform,
            download=True
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, _ = self.mnist[idx]
        return image

