import torch
from torch import nn
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)


class DCGANGenerator(nn.Module):
    """
    DCGAN Generator architecture for generating images from noise vectors.
    Uses transposed convolutions to upsample from latent space to image space.
    """

    def __init__(
            self,
            device: str = 'cpu',
            latent_dim: int = 100,
            out_channels: int = 1,
            hidden_dim: int = 64,
            img_size: int = 28,
            weights: str = None
    ):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim

        # Determine the initial size based on the target image size
        if img_size == 28:  # MNIST size
            initial_size = 7  # 7x7
        elif img_size == 32:  # CIFAR size
            initial_size = 8  # 8x8
        else:
            # For other sizes, calculate the appropriate initial size
            initial_size = img_size // 4

        # Define generator layers
        self.layers = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, hidden_dim * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            # state size: (hidden_dim*4) x 4 x 4
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            # state size: (hidden_dim*2) x 8 x 8
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            # state size: (hidden_dim) x 16 x 16
            nn.ConvTranspose2d(hidden_dim, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (out_channels) x 32 x 32 or (out_channels) x 28 x 28 after cropping
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Load pre-trained weights if provided
        if weights is not None:
            state_dict = torch.load(weights, map_location=self.device)
            self.load_state_dict(state_dict)
            log.info(f"Loaded weights from {weights} into DCGANGenerator")

    def _init_weights(self, m):
        """Initialize weights with mean=0, std=0.02"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, z):
        """
        Forward pass of the generator.

        Args:
            z (torch.Tensor): Batch of latent vectors [B, latent_dim]

        Returns:
            torch.Tensor: Generated images [B, C, H, W]
        """
        # Reshape the input
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)  # [B, latent_dim, 1, 1]
        return self.layers(z)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator architecture for classifying real vs. fake images.
    Uses strided convolutions to downsample from image space to a single classification output.
    """

    def __init__(
            self,
            device: str = 'cpu',
            in_channels: int = 1,
            hidden_dim: int = 64,
            weights: str = None
    ):
        super().__init__()
        self.device = device

        # Define discriminator layers
        self.layers = nn.Sequential(
            # input is (in_channels) x 28 x 28
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (hidden_dim) x 14 x 14
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (hidden_dim*2) x 7 x 7
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (hidden_dim*4) x 3 x 3
            nn.Conv2d(hidden_dim * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size: 1 x 1 x 1
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Load pre-trained weights if provided
        if weights is not None:
            state_dict = torch.load(weights, map_location=self.device)
            self.load_state_dict(state_dict)
            log.info(f"Loaded weights from {weights} into DCGANDiscriminator")

    def _init_weights(self, m):
        """Initialize weights with mean=0, std=0.02"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Batch of images [B, C, H, W]

        Returns:
            torch.Tensor: Classification scores [B, 1]
        """
        return self.layers(x).view(-1, 1).squeeze(1)


class WGANGenerator(nn.Module):
    """
    WGAN Generator architecture with similar structure to DCGAN but optimized
    for Wasserstein loss and improved stability.
    """

    def __init__(
            self,
            device: str = 'cpu',
            latent_dim: int = 100,
            out_channels: int = 1,
            hidden_dim: int = 64,
            img_size: int = 28,
            weights: str = None
    ):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim

        # Define generator layers (same as DCGAN but with different initialization)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, hidden_dim * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            # state size: (hidden_dim*4) x 4 x 4
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            # state size: (hidden_dim*2) x 8 x 8
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            # state size: (hidden_dim) x 16 x 16
            nn.ConvTranspose2d(hidden_dim, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (out_channels) x 32 x 32 or adjusted to 28 x 28
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Load pre-trained weights if provided
        if weights is not None:
            state_dict = torch.load(weights, map_location=self.device)
            self.load_state_dict(state_dict)
            log.info(f"Loaded weights from {weights} into WGANGenerator")

    def _init_weights(self, m):
        """Initialize weights for improved Wasserstein GAN stability"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, z):
        """
        Forward pass of the generator.

        Args:
            z (torch.Tensor): Batch of latent vectors [B, latent_dim]

        Returns:
            torch.Tensor: Generated images [B, C, H, W]
        """
        # Reshape the input
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)  # [B, latent_dim, 1, 1]
        return self.main(z)


class WGANDiscriminator(nn.Module):
    """
    WGAN Critic (Discriminator) architecture for estimating Wasserstein distance.
    No sigmoid activation at the end since we're not predicting probabilities.
    """

    def __init__(
            self,
            device: str = 'cpu',
            in_channels: int = 1,
            hidden_dim: int = 64,
            weights: str = None
    ):
        super().__init__()
        self.device = device

        # Define critic layers (no sigmoid at the end)
        self.main = nn.Sequential(
            # input is (in_channels) x 28 x 28
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (hidden_dim) x 14 x 14
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (hidden_dim*2) x 7 x 7
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (hidden_dim*4) x 3 x 3
            nn.Conv2d(hidden_dim * 4, 1, 4, 1, 0, bias=False)
            # No sigmoid for Wasserstein GAN
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Load pre-trained weights if provided
        if weights is not None:
            state_dict = torch.load(weights, map_location=self.device)
            self.load_state_dict(state_dict)
            log.info(f"Loaded weights from {weights} into WGANDiscriminator")

    def _init_weights(self, m):
        """Initialize weights for improved Wasserstein GAN stability"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Batch of images [B, C, H, W]

        Returns:
            torch.Tensor: Critic scores [B]
        """
        return self.main(x).view(-1)

