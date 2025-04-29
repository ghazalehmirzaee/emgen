import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from hydra.core.hydra_config import HydraConfig

# ! emgen imports
from emgen.utils.visualization import plot_2d_intermediate_samples, plot_images_intermediate_samples
from emgen.generative_model.gan.gan_model_arch import DCGANGenerator, DCGANDiscriminator, WGANGenerator, \
    WGANDiscriminator
from emgen.dataset.toy_dataset import ToyDataset
from emgen.utils.config_schema import TrainConfig

log = logging.getLogger(__name__)


class GANModel():
    """
    Train and Sample from a GAN Model
    """

    def __init__(
            self,
            device: str = 'cpu',
            generator=None,
            discriminator=None,
            dataset=None,
            train: TrainConfig = TrainConfig(),
            gan_type: str = 'dcgan',
            latent_dim: int = 100,
            clip_value: float = 0.01,  # For WGAN
    ):
        # ! Inputs
        self.device = device
        self.generator = generator.to(self.device) if generator else None
        self.discriminator = discriminator.to(self.device) if discriminator else None
        self.train_config = train
        self.gan_type = gan_type
        self.latent_dim = latent_dim
        self.clip_value = clip_value

        self.out_dir = Path(HydraConfig.get().runtime.output_dir)

        if dataset:
            # ! Get data dimensions
            sample = dataset[0]
            if isinstance(sample, torch.Tensor):
                self.data_dim = sample.shape
            else:
                # For toy datasets that return numpy arrays
                self.data_dim = (1, *sample.shape)

            # ! DataLoader
            self.train_loader = DataLoader(
                dataset,
                batch_size=self.train_config.train_batch_size,
                shuffle=True,
                drop_last=True
            )

            # ! Create generator and discriminator if not provided
            if not self.generator or not self.discriminator:
                self._create_models()

        # ! Optimizers
        if self.generator and self.discriminator:
            if self.gan_type == 'wgan':
                # WGAN typically uses RMSprop
                self.gen_optimizer = torch.optim.RMSprop(
                    self.generator.parameters(),
                    lr=self.train_config.learning_rate
                )
                self.disc_optimizer = torch.optim.RMSprop(
                    self.discriminator.parameters(),
                    lr=self.train_config.learning_rate
                )
            else:
                # DCGAN typically uses Adam with beta1=0.5
                self.gen_optimizer = torch.optim.Adam(
                    self.generator.parameters(),
                    lr=self.train_config.learning_rate,
                    betas=(0.5, 0.999)
                )
                self.disc_optimizer = torch.optim.Adam(
                    self.discriminator.parameters(),
                    lr=self.train_config.learning_rate,
                    betas=(0.5, 0.999)
                )

        self.frames = []

    def _create_models(self):
        """Create generator and discriminator models based on data dimensions and GAN type"""
        if len(self.data_dim) == 1:  # 1D data (toy dataset)
            # Not implemented for now
            raise NotImplementedError("GAN is not implemented for 1D data yet")

        elif len(self.data_dim) == 2:  # 2D data (toy dataset)
            # Create generator and discriminator for 2D data
            if self.gan_type == 'dcgan':
                self.generator = DCGANGenerator(
                    device=self.device,
                    latent_dim=self.latent_dim,
                    out_channels=1,
                    hidden_dim=64
                )
                self.discriminator = DCGANDiscriminator(
                    device=self.device,
                    in_channels=1,
                    hidden_dim=64
                )
            elif self.gan_type == 'wgan':
                self.generator = WGANGenerator(
                    device=self.device,
                    latent_dim=self.latent_dim,
                    out_channels=1,
                    hidden_dim=64
                )
                self.discriminator = WGANDiscriminator(
                    device=self.device,
                    in_channels=1,
                    hidden_dim=64
                )

        elif len(self.data_dim) == 3:  # Image data (MNIST, FashionMNIST)
            channels, height, width = self.data_dim
            if self.gan_type == 'dcgan':
                self.generator = DCGANGenerator(
                    device=self.device,
                    latent_dim=self.latent_dim,
                    out_channels=channels,
                    hidden_dim=64,
                    img_size=height
                )
                self.discriminator = DCGANDiscriminator(
                    device=self.device,
                    in_channels=channels,
                    hidden_dim=64
                )
            elif self.gan_type == 'wgan':
                self.generator = WGANGenerator(
                    device=self.device,
                    latent_dim=self.latent_dim,
                    out_channels=channels,
                    hidden_dim=64,
                    img_size=height
                )
                self.discriminator = WGANDiscriminator(
                    device=self.device,
                    in_channels=channels,
                    hidden_dim=64
                )

        else:
            raise ValueError(f"Unsupported data dimensions: {self.data_dim}")

    def train(self):
        log.info(f"Training the {self.gan_type.upper()} Model")
        self.generator.train()
        self.discriminator.train()

        gen_losses = []
        disc_losses = []

        for epoch in tqdm(range(self.train_config.num_epochs)):
            gen_epoch_loss = 0.0
            disc_epoch_loss = 0.0
            batch_count = 0

            for batch in tqdm(self.train_loader):
                if isinstance(batch, list) or isinstance(batch, tuple):
                    # For datasets that return (data, label) pairs
                    real_data = batch[0].to(self.device)
                else:
                    # For datasets that return just data
                    real_data = batch.to(self.device)

                batch_size = real_data.size(0)
                batch_count += 1

                # Create real and fake labels for the loss function
                real_labels = torch.ones(batch_size, device=self.device)
                fake_labels = torch.zeros(batch_size, device=self.device)

                # ---------------------
                # Train Discriminator
                # ---------------------
                self.disc_optimizer.zero_grad()

                # Train with real data
                real_output = self.discriminator(real_data)

                # Generate fake data
                z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
                fake_data = self.generator(z)
                fake_output = self.discriminator(fake_data.detach())

                # Calculate loss based on GAN type
                if self.gan_type == 'wgan':
                    # WGAN loss
                    disc_loss = -(torch.mean(real_output) - torch.mean(fake_output))
                else:
                    # DCGAN loss (binary cross entropy)
                    disc_real_loss = F.binary_cross_entropy(real_output, real_labels)
                    disc_fake_loss = F.binary_cross_entropy(fake_output, fake_labels)
                    disc_loss = disc_real_loss + disc_fake_loss

                # Update discriminator
                disc_loss.backward()
                self.disc_optimizer.step()

                # Clip weights (for WGAN)
                if self.gan_type == 'wgan':
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-self.clip_value, self.clip_value)

                # ---------------------
                # Train Generator
                # ---------------------
                self.gen_optimizer.zero_grad()

                # Generate fake data
                z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
                fake_data = self.generator(z)
                fake_output = self.discriminator(fake_data)

                # Calculate loss based on GAN type
                if self.gan_type == 'wgan':
                    # WGAN loss
                    gen_loss = -torch.mean(fake_output)
                else:
                    # DCGAN loss (try to fool discriminator)
                    gen_loss = F.binary_cross_entropy(fake_output, real_labels)

                # Update generator
                gen_loss.backward()
                self.gen_optimizer.step()

                # Record losses
                gen_epoch_loss += gen_loss.item()
                disc_epoch_loss += disc_loss.item()

            # Calculate average epoch losses
            gen_epoch_loss /= batch_count
            disc_epoch_loss /= batch_count
            gen_losses.append(gen_epoch_loss)
            disc_losses.append(disc_epoch_loss)

            log.info(
                f"Epoch [{epoch + 1}/{self.train_config.num_epochs}] - Gen Loss: {gen_epoch_loss:.4f}, Disc Loss: {disc_epoch_loss:.4f}")

            # Save samples and model periodically
            if epoch % self.train_config.save_images_step == 0 or epoch == self.train_config.num_epochs - 1:
                sample_out = self.sample()
                sample_out_dir = self.out_dir / f"train_epoch_{epoch:02d}"
                sample_out_dir.mkdir(parents=True, exist_ok=True)

                # Save generated samples
                if len(self.data_dim) == 3:  # Image data
                    # Intermediate samples aren't available in the same way as diffusion,
                    # but we can create a sequence of different noise vectors to show progress
                    plot_images_intermediate_samples(
                        np.expand_dims(sample_out['generated_samples'], axis=0),
                        sample_out_dir,
                        1,
                    )
                elif len(self.data_dim) == 2:  # 2D data
                    plot_2d_intermediate_samples(
                        np.expand_dims(sample_out['generated_samples'], axis=0),
                        sample_out_dir,
                        1,
                    )

                # Save models
                torch.save(self.generator.state_dict(), sample_out_dir / f"generator.pth")
                torch.save(self.discriminator.state_dict(), sample_out_dir / f"discriminator.pth")

                # Save loss plot
                plt.figure(figsize=(10, 5))
                plt.plot(gen_losses, label='Generator Loss')
                plt.plot(disc_losses, label='Discriminator Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'{self.gan_type.upper()} Training Loss')
                plt.legend()
                plt.savefig(sample_out_dir / 'loss_plot.png')
                plt.close()

        return {
            'gen_losses': gen_losses,
            'disc_losses': disc_losses
        }

    def sample(self, num_samples=None):
        """
        Sample from the trained generator.

        Args:
            num_samples (int, optional): Number of samples to generate. Defaults to eval_batch_size.

        Returns:
            dict: Dictionary containing generated samples.
        """
        log.info("Sampling from the GAN Model")
        self.generator.eval()

        # Set number of samples
        if num_samples is None:
            num_samples = self.train_config.eval_batch_size

        # Generate random noise
        z = torch.randn(num_samples, self.latent_dim, 1, 1, device=self.device)

        # Generate samples
        with torch.no_grad():
            samples = self.generator(z)

        # Convert to numpy arrays
        generated_samples = samples.cpu().numpy()

        return {
            'generated_samples': generated_samples,
            'latent_vectors': z.cpu().numpy()
        }

    def generate_samples(self, num_samples=1000, output_dir=None):
        """
        Generate and save a large number of samples.

        Args:
            num_samples (int): Number of samples to generate.
            output_dir (str): Directory to save samples.

        Returns:
            numpy.ndarray: Generated samples.
        """
        if output_dir is None:
            output_dir = self.out_dir / 'generated_samples'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Generating {num_samples} samples from the GAN Model")
        self.generator.eval()

        batch_size = min(100, num_samples)  # Process in batches to avoid memory issues
        remaining = num_samples
        all_samples = []

        with torch.no_grad():
            while remaining > 0:
                current_batch_size = min(batch_size, remaining)
                z = torch.randn(current_batch_size, self.latent_dim, 1, 1, device=self.device)
                samples = self.generator(z)
                all_samples.append(samples.cpu().numpy())
                remaining -= current_batch_size

        # Combine all batches
        all_samples = np.concatenate(all_samples, axis=0)

        # Save all samples as a numpy array
        np.save(output_dir / 'all_samples.npy', all_samples)

        # Save a grid visualization of some samples
        num_vis_samples = min(100, num_samples)
        vis_samples = all_samples[:num_vis_samples]

        if len(self.data_dim) == 3:  # Image data
            # Reshape for visualization
            grid_size = int(np.ceil(np.sqrt(num_vis_samples)))
            h, w = self.data_dim[1], self.data_dim[2]
            channels = self.data_dim[0]
            grid = np.zeros((channels, grid_size * h, grid_size * w))

            for i in range(num_vis_samples):
                row = i // grid_size
                col = i % grid_size
                grid[:, row * h:(row + 1) * h, col * w:(col + 1) * w] = vis_samples[i]

            # Save as image
            if channels == 1:
                plt.figure(figsize=(10, 10))
                plt.imshow(grid[0], cmap='gray')
                plt.axis('off')
                plt.savefig(output_dir / 'sample_grid.png')
                plt.close()
            else:
                # For RGB images
                plt.figure(figsize=(10, 10))
                plt.imshow(np.transpose(grid, (1, 2, 0)))
                plt.axis('off')
                plt.savefig(output_dir / 'sample_grid.png')
                plt.close()

        elif len(self.data_dim) == 2:  # 2D data
            plt.figure(figsize=(10, 10))
            plt.scatter(vis_samples[:, 0], vis_samples[:, 1], alpha=0.6, s=2)
            plt.title(f"Generated 2D Distribution ({num_vis_samples} samples)")
            plt.savefig(output_dir / 'sample_scatter.png')
            plt.close()

        return all_samples

