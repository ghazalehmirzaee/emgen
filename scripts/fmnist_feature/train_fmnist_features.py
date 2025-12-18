import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm.auto import tqdm
import time
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


#! My Imports
from fashion_cnn import FashionCNN
from early_stopping import EarlyStopping


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("WARNING: GPU not available, using CPU (training will be slow)")

# Create output directories
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("features").mkdir(exist_ok=True)

print("Defining augmentation strategies...\n")

# Strategy 1: LIGHT AUGMENTATION (Conservative)
light_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),  # Rotate ±15 degrees
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)  # Shift up to 10% in any direction
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST mean and std
])

# Strategy 2: MEDIUM AUGMENTATION (Moderate)
medium_transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),  # Rotate ±30 degrees
    transforms.RandomAffine(
        degrees=0,
        translate=(0.15, 0.15),  # Shift up to 15%
        scale=(0.85, 1.15)  # Scale between 85% and 115%
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # 50% chance of perspective warp
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# Strategy 3: HEAVY AUGMENTATION (Aggressive)
heavy_transform = transforms.Compose([
    transforms.RandomRotation(degrees=45),  # Rotate ±45 degrees
    transforms.RandomAffine(
        degrees=0,
        translate=(0.2, 0.2),  # Shift up to 20%
        scale=(0.8, 1.2),  # Scale between 80% and 120%
        shear=10  # Shear transformation
    ),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.RandomHorizontalFlip(p=0.3),  # 30% chance to flip
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# TEST TRANSFORM (No augmentation - just normalization)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# Store in dictionary for easy iteration
augmentation_strategies = {
    'light': light_transform,
    'medium': medium_transform,
    'heavy': heavy_transform
}

print("Testing model architecture...")
model = FashionCNN(num_classes=10).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f" Model created successfully!")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")

# Test forward pass
test_input = torch.randn(2, 1, 28, 28).to(device)
test_output, test_features = model(test_input, return_features=True)
print(f"\n   Input shape: {test_input.shape}")
print(f"   Output shape: {test_output.shape}")
print(f"   Feature layers available: {list(test_features.keys())}")
for layer_name, feat in test_features.items():
    print(f"      {layer_name}: {feat.shape}")

del test_input, test_output, test_features  # Free memory

# ============================================================================
# Define Training and Validation Functions
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


# ============================================================================
# Complete Training Loop
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs, device,
                model_name, patience=10):
    """
    Complete training loop with:
    - Loss function with label smoothing
    - AdamW optimizer with weight decay
    - Learning rate scheduler
    - Early stopping
    """

    # Loss function with label smoothing (prevents overconfidence)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer: AdamW with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Learning rate scheduler: reduce LR when validation accuracy plateaus
    # FIXED: Removed 'verbose' parameter
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    model_path = f'models/{model_name}_best.pth'
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}\n")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 70)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Get current learning rate BEFORE scheduler step
        current_lr = optimizer.param_groups[0]['lr']

        # Update learning rate based on validation accuracy
        old_lr = current_lr
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']

        # Manually print if LR changed (since we removed verbose)
        if new_lr != old_lr:
            print(f'   📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}')

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(new_lr)

        # Print results
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {new_lr:.6f}')

        # Early stopping check
        early_stopping(val_acc, model, model_path)
        if early_stopping.early_stop:
            print("\n Early stopping triggered!")
            break

        print()

    training_time = time.time() - start_time
    print(f"{'='*70}")
    print(f"Training completed in {training_time/60:.2f} minutes")
    print(f" Best validation accuracy: {early_stopping.val_acc_max:.2f}%")
    print(f"{'='*70}\n")

    # Load best model
    model.load_state_dict(torch.load(model_path))

    # Save history
    with open(f'results/{model_name}_history.json', 'w') as f:
        json.dump(history, f, indent=4)

    return model, history


# ============================================================================
# Define Visualization Functions
# ============================================================================

def plot_training_history(history, model_name):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=4)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2, marker='o', markersize=4)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2, marker='s', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Learning rate plot
    axes[2].plot(history['lr'], linewidth=2, color='green', marker='o', markersize=4)
    axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'results/{model_name}_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model, test_loader, device, model_name):
    """Comprehensive model evaluation with confusion matrix"""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate accuracy
    accuracy = 100 * np.mean(all_preds == all_labels)
    print(f"\n Test Accuracy: {accuracy:.2f}%")

    # Class names for Fashion-MNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 10})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {model_name}\nTest Accuracy: {accuracy:.2f}%',
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return accuracy, cm


