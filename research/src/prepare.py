import os
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from data_manager import prepare_dataset


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    cwd = hydra.utils.get_original_cwd()
    load_dotenv(os.path.join(cwd, ".env"))

    raw_dir = os.path.join(cwd, cfg.data.raw_dir)
    prepared_dir = os.path.join(cwd, cfg.data.prepared_dir)

    if not os.path.exists(raw_dir):
        raise FileNotFoundError(
            f"raw_dir не знайдено: {raw_dir}\n"
            f"Спочатку: uv run src/download.py [pose|ramp]"
        )

    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    prepare_dataset(raw_dir, prepared_dir, data_config)


if __name__ == "__main__":
    main()
