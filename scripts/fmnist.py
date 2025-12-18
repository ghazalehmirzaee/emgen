import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import argparse
from torchvision.utils import save_image
import os
from torchvision.models import ResNet18_Weights, Inception_V3_Weights
from tqdm import tqdm
from fashion_cnn import FashionCNN  # Import your custom model

# Parameters
num_classes = 10
num_samples_per_class = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['resnet18', 'inceptionv3', 'fashioncnn'], default='resnet18')
parser.add_argument('--rotation', type=float, default=15, help='Random rotation degree')
parser.add_argument('--affine_translate', type=float, default=0.1, help='Random affine translation (fraction)')
parser.add_argument('--perspective', type=float, default=0.2, help='Random perspective distortion scale')
parser.add_argument('--fashioncnn_weights', type=str, default='fmnist_feature/fashion_cnn_heavy_best.pth', help='Path to FashionCNN weights')
args = parser.parse_args()


# 2. Preprocessing for feature extractor with augmentations
if args.model == 'inceptionv3':
    input_size = 299
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
elif args.model == 'resnet18':
    input_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
else: # for fashioncnn
    input_size = 28
    normalize = transforms.Normalize(mean=[0.2860], std=[0.3530])

# Transforms
transform_list = [
    transforms.RandomRotation(degrees=args.rotation),
    transforms.RandomAffine(degrees=0, translate=(args.affine_translate, args.affine_translate)),
    transforms.RandomPerspective(distortion_scale=args.perspective, p=0.5),
    transforms.Resize(input_size),
]
if args.model != 'fashioncnn':
    transform_list.append(transforms.Grayscale(num_output_channels=3))
transform_list.extend([
    transforms.ToTensor(),
    normalize
])
transform = transforms.Compose(transform_list)

# 3. Load FashionMNIST
dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

os.makedirs("grids", exist_ok=True)

# 5. Load pre-trained model and remove classifier
if args.model == 'resnet18':
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # extract features from penultimate layer
    feature_type = "ResNet18"
elif args.model == 'inceptionv3':
    model = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # extract features from penultimate layer
    model.aux_logits = False  # disable auxiliary outputs for inference
    feature_type = "InceptionV3"
elif args.model == 'fashioncnn':
    model = FashionCNN(num_classes=10)
    model.load_state_dict(torch.load(args.fashioncnn_weights, map_location=device))
    model = model.to(device)
    model.eval()
    feature_type = "FashionCNN"
else:
    raise ValueError("Unknown model")

model = model.to(device)
model.eval()

def extract_features(loader):
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            if args.model == 'fashioncnn':
                # FashionCNN expects [N, 1, 28, 28] and returns dict of features
                # Undo normalization for FashionCNN, then apply its normalization
                _, features = model(imgs, return_features=True)
                feats = features['pooled']
            else:
                feats = model(imgs)
                if isinstance(feats, tuple):  # Inception may return (features, aux)
                    feats = feats[0]
            return feats.cpu().numpy()

# Compute L2 distances
def pairwise_l2(feats1, feats2):
    dists = np.linalg.norm(feats1[:, None, :] - feats2[None, :, :], axis=-1)
    return dists


# Calculate within-class and between-class feature distances for all 10 classes, then create and print a confusion matrix of mean L2 distances.
all_indices = []
for c in range(num_classes):
    all_indices.append([i for i, (_, label) in enumerate(dataset) if label == c])

# Sample 50 indices per class
sampled_indices = [random.sample(idxs, num_samples_per_class) for idxs in all_indices]
subsets = [Subset(dataset, idxs) for idxs in sampled_indices]
loaders = [DataLoader(subset, batch_size=num_samples_per_class, shuffle=False) for subset in subsets]

# Extract features for all classes
features = []
for loader in tqdm(loaders, desc="Extracting features for all classes"):
    features.append(extract_features(loader))
features = np.stack(features)  # shape: (10, 50, feature_dim)

# Compute confusion matrix of mean L2 distances
confusion = np.zeros((num_classes, num_classes))
for i in range(num_classes):
    for j in range(num_classes):
        dists = np.linalg.norm(features[i][:, None, :] - features[j][None, :, :], axis=-1)
        if i == j:
            # Exclude self-distances (diagonal) for within-class
            mask = ~np.eye(num_samples_per_class, dtype=bool)
            confusion[i, j] = dists[mask].mean()
        else:
            confusion[i, j] = dists.mean()

print("Confusion matrix of mean L2 feature distances (rows: true class, cols: compared class):")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8,6))
    sns.heatmap(
        confusion,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        # vmin=12,
        # vmax=20
    )
    plt.xlabel("Class")
    plt.ylabel("Class")
    # Add model and augmentation info to the title
    aug_info = (
        f"Rotation: {args.rotation}°, "
        f"Affine translation: ±{int(args.affine_translate*100)}%, "
        f"Perspective distortion: {args.perspective}"
    )
    model_name = feature_type
    plt.title(f"Feature Distance ({model_name})\n{aug_info}")
    plt.tight_layout()
    # Compose filename with extractor and distortion info
    extractor = args.model
    rot = int(args.rotation)
    aff = int(args.affine_translate * 100)
    persp = int(args.perspective * 10)
    fname = f"grids/{extractor}_{rot}_{aff}_{persp}.png"
    plt.savefig(fname)
    plt.close()
except ImportError:
    pass

# Save 10x10 grid of first 10 samples from each class (row: class, col: sample)
def save_fmnist_grid(dataset, sampled_indices, fname):
    imgs = []
    for class_indices in sampled_indices:
        # Take first 10 indices for this class
        for idx in class_indices[:10]:
            img, _ = dataset[idx]
            imgs.append(img)
    imgs_tensor = torch.stack(imgs)  # shape: (100, C, H, W)
    save_image(imgs_tensor, fname, nrow=10, padding=2, normalize=True)

fmnist_grid_fname = fname.replace('.png', '_fmnist_grid.png')
save_fmnist_grid(dataset, sampled_indices, fmnist_grid_fname)
