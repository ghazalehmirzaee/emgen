import hydra
from hydra.utils import instantiate
import os
from omegaconf import DictConfig, OmegaConf
import logging
import torch
from pathlib import Path

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../emgen_config", config_name="my_config")
def main(cfg: DictConfig) -> None:
    log.info(f"Configuration: \n {OmegaConf.to_yaml(cfg)}")

    # Set random seed for reproducibility
    if 'seed' in cfg:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    # Instantiate the generative model
    model = instantiate(cfg.generative_model)

    # Create output directory
    output_dir = Path(os.getcwd()) / "generated_samples"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Number of samples to generate
    num_samples = 1000
    if hasattr(cfg, 'num_samples'):
        num_samples = cfg.num_samples

    log.info(f"Generating {num_samples} samples using {cfg.generative_model._target_}")

    # Generate samples
    samples = model.generate_samples(num_samples=num_samples, output_dir=output_dir)

    log.info(f"Generated {len(samples)} samples")
    log.info(f"Samples saved to {output_dir}")


if __name__ == "__main__":
    main()

