import torch
from torch import nn
import logging

from emgen.utils.positional_embeddings import PositionalEmbedding
from emgen.model.simple_linear import SimpleLinear
from emgen.model.unet import DownBlock, Bottleneck, UpBlock

log = logging.getLogger(__name__)


class LinearArch(nn.Module):
    """
    For given xt, it predicts x0 or the noise.
    """
    def __init__(
        self, 
        device: str = 'cpu',
        data_dim: int = 2,
        emb_size: int = 128,
        input_emb: str = "sinusoidal",
        time_emb: str = "sinusoidal",
        hidden_size: int = 128, 
        hidden_layers: int = 3,
        weights: str = None
    ):
        super().__init__()
        self.device = device

        #! Positional Embeddings
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0, device=self.device)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0, device=self.device)
        self.time_mlp   = PositionalEmbedding(emb_size, time_emb, device=self.device)

        #! Main Backbone
        input_size = len(self.input_mlp1.layer) + len(self.input_mlp2.layer) + len(self.time_mlp.layer)
        self.joint_mlp = SimpleLinear(input_size, hidden_size, hidden_layers, data_dim)
        
        if weights is not None:
            state_dict = torch.load(weights, map_location=self.device)
            self.load_state_dict(state_dict)
            log.info(f"Loaded weights from {weights} into LinearArch")


    def forward(self, x, t):
        """
        B = batch size, d = dimension
        
        Args:
            x (torch.Tensor): [B, d] Noisy sample
            t (torch.Tensor): [B,] Time step
        Returns:
            torch.Tensor: [b, d] Noise or x0 sample
        """
        #! Positional Embedding
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        
        #! Input for the backbone
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        
        #! Backbone
        x = self.joint_mlp(x)
        
        return x
   

class UNetArch(nn.Module):
    def __init__(self,
        device: str = 'cpu',
        emb_size=128, 
        time_emb: str = "sinusoidal",
        in_channels=3, 
        base_channels=64, 
        num_down=2,
        weights: str = None
    ):
        """
        Args:
            in_channels (int): Number of input (and output) channels.
            base_channels (int): Number of channels at input and output of UNet
            num_down (int): Number of downsampling steps.
        """
        super().__init__()
        self.device = device
        
        #! Time embedding
        self.time_mlp = PositionalEmbedding(emb_size, time_emb, scale=1.0, device=self.device)
        
        #! Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        #! Encoder
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for _ in range(num_down):
            self.down_blocks.append(DownBlock(channels, channels * 2, emb_size))
            channels *= 2
        
        #! Bottleneck block
        self.bottleneck = Bottleneck(channels, channels, emb_size)
        
        #! Decoder
        self.up_blocks = nn.ModuleList()
        for _ in range(num_down):
            self.up_blocks.append(UpBlock(channels, channels // 2, emb_size))
            channels //= 2
        
        #! Final convolution
        self.final_conv = nn.Conv2d(channels, in_channels, kernel_size=1)
       
        #! Load weights if available 
        if weights is not None:
            state_dict = torch.load(weights, map_location=self.device)
            self.load_state_dict(state_dict)
            log.info(f"Loaded weights from {weights} into LinearArch")
   
    
    def forward(self, x, t):
        """
        B = batch size, C = image channels, H = height, W = width
        
        Args:
            x (torch.Tensor): [B, C, H, W] Noisy Image.
            t (torch.Tensor): [B,] Time step.
        Returns:
            torch.Tensor: [B, C, H, W] Noise or x0 sample.
        """
        #! Create time embedding for conditioning
        t_emb = self.time_mlp(t)  # t_emb shape: [B, emb_size]
        
        #! Initial convolution
        x = self.init_conv(x)
        
        #! Downsample
        skip_connections = []
        for down in self.down_blocks:
            skip, x = down(x, t_emb)
            skip_connections.append(skip)
        
        #! Bottleneck
        x = self.bottleneck(x, t_emb)
        
        #! Upsample
        for up in self.up_blocks:
            skip = skip_connections.pop()
            x = up(x, skip, t_emb)
        
        #! Final projection
        return self.final_conv(x)
     

class UNet1DArch(nn.Module):
    """
    UNet architecture for 1D data.
    Uses 1D convolutions and appropriate pooling/upsampling operations.
    """

    def __init__(
            self,
            device: str = 'cpu',
            emb_size: int = 128,
            time_emb: str = "sinusoidal",
            in_channels: int = 1,
            base_channels: int = 64,
            num_down: int = 2,
            weights: str = None
    ):
        super().__init__()
        self.device = device

        # Time embedding
        self.time_mlp = PositionalEmbedding(emb_size, time_emb, scale=1.0, device=self.device)

        # Initial 1D convolution
        self.init_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for _ in range(num_down):
            self.down_blocks.append(Down1DBlock(channels, channels * 2, emb_size))
            channels *= 2

        # Bottleneck block
        self.bottleneck = Bottleneck1D(channels, channels, emb_size)

        # Decoder
        self.up_blocks = nn.ModuleList()
        for _ in range(num_down):
            self.up_blocks.append(Up1DBlock(channels, channels // 2, emb_size))
            channels //= 2

        # Final convolution
        self.final_conv = nn.Conv1d(channels, in_channels, kernel_size=1)

        # Load weights if available
        if weights is not None:
            state_dict = torch.load(weights, map_location=self.device)
            self.load_state_dict(state_dict)
            logging.info(f"Loaded weights from {weights} into UNet1DArch")

    def forward(self, x, t):
        """
        B = batch size, C = channels, L = sequence length

        Args:
            x (torch.Tensor): [B, C, L] Noisy 1D data
            t (torch.Tensor): [B,] Time step
        Returns:
            torch.Tensor: [B, C, L] Noise or x0 prediction
        """
        # Time embedding
        t_emb = self.time_mlp(t)  # [B, emb_size]

        # Initial convolution
        x = self.init_conv(x)

        # Downsample path
        skip_connections = []
        for down in self.down_blocks:
            skip, x = down(x, t_emb)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x, t_emb)

        # Upsample path
        for up in self.up_blocks:
            skip = skip_connections.pop()
            x = up(x, skip, t_emb)

        # Final projection
        return self.final_conv(x)


class Down1DBlock(nn.Module):
    """Downsampling block for 1D UNet"""

    def __init__(self, in_channels, out_channels, emb_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                                    stride=2, padding=1)
        self.time_mlp = nn.Linear(emb_size, out_channels)

    def forward(self, x, t_emb):
        x = self.conv(x)
        t_emb = self.time_mlp(t_emb).view(x.shape[0], -1, 1)
        x += t_emb
        x_downsampled = self.downsample(x)
        return x, x_downsampled


class Bottleneck1D(nn.Module):
    """Bottleneck block for 1D UNet"""

    def __init__(self, in_channels, out_channels, emb_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.time_mlp = nn.Linear(emb_size, out_channels)

    def forward(self, x, t_emb):
        h = self.conv(x)
        t_bias = self.time_mlp(t_emb).view(h.shape[0], -1, 1)
        return h + t_bias


class Up1DBlock(nn.Module):
    """Upsampling block for 1D UNet"""

    def __init__(self, in_channels, out_channels, emb_size):
        super().__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, in_channels,
                                         kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.time_mlp = nn.Linear(emb_size, out_channels)

    def forward(self, x, skip, t_emb):
        x = self.upconv(x)
        # Handle potential dimension mismatch
        if x.shape[2] != skip.shape[2]:
            diff = skip.shape[2] - x.shape[2]
            x = F.pad(x, (diff // 2, diff - diff // 2))
        # Concatenate along channel dimension (skip connection)
        x = torch.cat([x, skip], dim=1)
        h = self.conv(x)
        t_bias = self.time_mlp(t_emb).view(h.shape[0], -1, 1)
        return h + t_bias

