import hydra
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="../emgen_config", config_name="my_config")
def main(cfg) -> None:
    emgen = instantiate(cfg)
    emgen.generative_model.train()

if __name__ == "__main__":
    main()

