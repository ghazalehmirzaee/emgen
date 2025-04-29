import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import os
from pathlib import Path
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from emgen.utils.memorization import (
    compute_memorization_l2,
    compute_memorization_ssim,
    plot_memorization_results
)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../emgen_config", config_name="my_config")
def main(cfg: DictConfig) -> None:
    log.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")

    # Set device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Instantiate dataset
    dataset = instantiate(cfg.generative_model.dataset)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.generative_model.train.train_batch_size,
        shuffle=False
    )

    # Collect training samples
    log.info("Collecting training samples...")
    all_training_samples = []
    for batch in tqdm(train_loader):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]  # Get data if it's a tuple (data, label)
        all_training_samples.append(batch)

    training_samples = torch.cat(all_training_samples, dim=0)
    log.info(f"Collected {len(training_samples)} training samples")

    # Instantiate diffusion model
    log.info("Instantiating diffusion model...")
    diffusion_model = instantiate(cfg.generative_model)

    # Generate samples
    log.info("Generating samples from diffusion model...")
    sample_out = diffusion_model.sample(get_intermediate_samples=True)
    generated_samples = torch.from_numpy(sample_out["generated_sample"]).to(device)
    log.info(f"Generated {len(generated_samples)} samples")

    # Set parameters for memorization metrics
    k = getattr(cfg, "k", 50)
    alpha = getattr(cfg, "alpha", 0.5)

    # Calculate L2-based memorization
    log.info(f"Computing L2-based memorization (k={k}, alpha={alpha})...")
    l2_results = compute_memorization_l2(
        generated_samples.cpu().numpy(),
        training_samples.cpu().numpy(),
        k=k,
        alpha=alpha
    )

    # Calculate SSIM-based memorization
    log.info(f"Computing SSIM-based memorization (k={k}, alpha={alpha})...")
    ssim_results = compute_memorization_ssim(
        generated_samples.cpu().numpy(),
        training_samples.cpu().numpy(),
        k=k,
        alpha=alpha
    )

    # Save results
    output_dir = Path(os.getcwd()) / "memorization_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Plotting and saving results to {output_dir}...")

    # Plot L2 memorization results
    plot_memorization_results(
        generated_samples.cpu().numpy(),
        training_samples.cpu().numpy(),
        l2_results,
        "l2",
        output_dir,
        top_n=10
    )

    # Plot SSIM memorization results
    plot_memorization_results(
        generated_samples.cpu().numpy(),
        training_samples.cpu().numpy(),
        ssim_results,
        "ssim",
        output_dir,
        top_n=10
    )

    # Print summary statistics
    log.info(f"L2 Memorization Score Statistics:")
    log.info(f"  Min: {float(min(l2_results['memorization_scores'])):.4f}")
    log.info(f"  Max: {float(max(l2_results['memorization_scores'])):.4f}")
    log.info(f"  Mean: {float(np.mean(l2_results['memorization_scores'])):.4f}")
    log.info(f"  Median: {float(np.median(l2_results['memorization_scores'])):.4f}")

    log.info(f"SSIM Memorization Score Statistics:")
    log.info(f"  Min: {float(min(ssim_results['memorization_scores'])):.4f}")
    log.info(f"  Max: {float(max(ssim_results['memorization_scores'])):.4f}")
    log.info(f"  Mean: {float(np.mean(ssim_results['memorization_scores'])):.4f}")
    log.info(f"  Median: {float(np.median(ssim_results['memorization_scores'])):.4f}")


    # Save numeric results
    np.savez(
        output_dir / "memorization_results.npz",
        l2_scores=l2_results['memorization_scores'],
        l2_nearest=l2_results['nearest_neighbors'],
        ssim_scores=ssim_results['memorization_scores'],
        ssim_nearest=ssim_results['nearest_neighbors']
    )

    log.info("Memorization evaluation complete!")


if __name__ == "__main__":
    main()

