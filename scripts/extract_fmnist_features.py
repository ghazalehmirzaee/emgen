import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from fashion_cnn import FashionCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load 10k random indices
indices = np.load('indices.npy')

# Load FashionMNIST full training set (60k images)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2860], std=[0.3530])
])
dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)

# Sample 10k images using indices
images = []
for idx in indices:
    img, _ = dataset[idx]
    images.append(img)
images = torch.stack(images)  # shape: (10000, 1, 28, 28)

# Load FashionCNN
model = FashionCNN(num_classes=10)
model.load_state_dict(torch.load('fmnist_feature/fashion_cnn_heavy_best.pth', map_location=device))
model = model.to(device)
model.eval()

# Compute features in batches to avoid OOM
features = []
batch_size = 256
print("Extracting features...")
with torch.no_grad():
    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i:i+batch_size].to(device)
        _, feats = model(batch, return_features=True)
        features.append(feats['pooled'].cpu().numpy())
features = np.concatenate(features, axis=0)  # shape: (10000, feature_dim)

np.save('fmnist_features.npy', features)



