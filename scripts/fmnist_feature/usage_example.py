import torch
import numpy as np
from model_architecture import FashionCNN
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model (choose: light, medium, or heavy)
model = FashionCNN(num_classes=10)
model.load_state_dict(torch.load('fmnist_feature/fashion_cnn_medium_best.pth', map_location=device))
model = model.to(device)
model.eval()

# Extract features from your generated images
# Input: images shape [N, 28, 28] or [N, 1, 28, 28]
def extract_features(images):
    # Preprocess
    if images.ndim == 3:
        images = images[:, np.newaxis, :, :]
    images = (images - 0.2860) / 0.3530  # Fashion-MNIST normalization
    
    images_tensor = torch.FloatTensor(images).to(device)
    
    with torch.no_grad():
        _, features = model(images_tensor, return_features=True)
        return features['pooled'].cpu().numpy()  # [N, 512]

# Load FashionMNIST test set
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)
loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)

# Get a batch of images
images, labels = next(iter(loader))  # images: [1000, 1, 28, 28]
images_np = images.numpy().squeeze(1)  # shape: [1000, 28, 28]

features = extract_features(images_np)
print(features.shape)  # (1000, 512)
