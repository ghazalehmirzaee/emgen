import torch
import os
import numpy as np
from emgen.generative_model.diffusion.diffusion_model import DiffusionModel
import cv2
from tqdm import tqdm

import hydra
from hydra.utils import instantiate


def save_grid(images, out_path, nrow=10):
    import torchvision.utils as vutils
    import torch
    # images: numpy array (N, C, H, W) or (N, H, W)
    imgs = images[:nrow*nrow]
    if imgs.ndim == 3:  # (N, H, W)
        imgs = imgs[:, None, :, :]
    imgs_tensor = torch.from_numpy(imgs).float()
    imgs_tensor = imgs_tensor / 1.0  # already in [0,1]
    vutils.save_image(imgs_tensor, out_path, nrow=nrow, padding=2, normalize=True)


@hydra.main(version_base=None, config_path="../emgen_config", config_name="sample")
def main(cfg) -> None:
    print("Instantiating diffusion model...")
    diffusion_model: DiffusionModel = instantiate(cfg)
    diffusion_model.diffusion_arch.eval()

    total_batches = 157 # sample 157*32=5024 images

    # Loop over epochs 00 to 99
    for epoch in range(100):
        weights = f'/home/wvuirl/ws/rashik/emgen/scripts/outputs/2025-12-15/00-12-27/train_epoch_{epoch:02d}/diffusion_arch.pth'
        if not os.path.exists(weights):
            print(f"Skipping epoch {epoch:02d}, weights not found.")
            continue
        state_dict = torch.load(weights, map_location='cuda')
        diffusion_model.diffusion_arch.load_state_dict(state_dict)
        print(f'Loaded weights for epoch {epoch:02d}')

        # Sample
        all_samples = []
        for _ in tqdm(range(total_batches), desc=f"Sampling epoch {epoch:02d}"):
            output = diffusion_model.sample(get_intermediate_samples=True)
            samples = output['generated_sample']  # shape: (10, C, H, W) or (10, H, W)
            all_samples.append(samples)
        all_samples = np.concatenate(all_samples, axis=0)  # shape: (N, C, H, W) or (N, H, W)

        # Save all samples as npy
        np.save(f"samples/all_samples_{epoch:02d}.npy", all_samples)
        save_grid(all_samples, f"samples/sample_grid_{epoch:02d}.png", nrow=10)


if __name__ == "__main__":
    main()