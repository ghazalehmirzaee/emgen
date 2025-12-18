import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    This allows gradients to flow better during training (prevents vanishing gradient).
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity mapping or projection)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Add skip connection
        out = self.relu(out)

        return out


class FashionCNN(nn.Module):
    """
    Professional CNN with residual connections for Fashion-MNIST.
    Architecture: 4 residual layers (64 -> 128 -> 256 -> 512 channels)
    """
    def __init__(self, num_classes=10):
        super(FashionCNN, self).__init__()

        # Initial convolution (1 channel -> 64 channels)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)    # 28x28 -> 28x28
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)   # 28x28 -> 14x14
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)  # 14x14 -> 7x7
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)  # 7x7 -> 3x3

        # Global average pooling (3x3 -> 1x1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Initialize weights properly
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Helper function to create a sequence of residual blocks"""
        layers = []
        # First block may have stride > 1 (downsampling)
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # Remaining blocks have stride = 1
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization (good for ReLU)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        """
        Forward pass through the network.
        If return_features=True, also returns intermediate feature maps.
        """
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Residual layers
        x1 = self.layer1(x)    # 64 channels
        x2 = self.layer2(x1)   # 128 channels
        x3 = self.layer3(x2)   # 256 channels
        x4 = self.layer4(x3)   # 512 channels

        # Global pooling
        x_pool = self.avgpool(x4)
        x_flat = torch.flatten(x_pool, 1)

        # Classification
        out = self.fc(x_flat)

        if return_features:
            # Return features from multiple layers for semantic analysis
            return out, {
                'layer1': x1,
                'layer2': x2,
                'layer3': x3,
                'layer4': x4,
                'pooled': x_flat
            }

        return out
