import os
import shutil
import random
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv


def _prepare(raw_dir: str, prepared_dir: str, data_config: dict) -> None:
    raw_path = Path(raw_dir)
    images_dir = raw_path / "images"
    labels_dir = raw_path / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Не знайдено: {images_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in exts)
    if not all_images:
        raise ValueError(f"Зображень не знайдено в {images_dir}")

    val_hives = list(data_config.get("val_hives") or [])
    val_ratio = float(data_config.get("val_ratio", 0.2))

    if val_hives:
        val_set = set(val_hives)
        train_images = [f for f in all_images if f.stem[:9] not in val_set]
        val_images = [f for f in all_images if f.stem[:9] in val_set]
        print(f"INFO: Hive-based split → train: {len(train_images)}, val: {len(val_images)}")
        if not val_images:
            print("WARNING: Жодного зображення для вказаних val_hives.")
    else:
        rng = random.Random(42)
        shuffled = list(all_images)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_images = shuffled[:n_val]
        train_images = shuffled[n_val:]
        print(f"INFO: Random split (val_ratio={val_ratio}) → train: {len(train_images)}, val: {len(val_images)}")

    prepared = Path(prepared_dir)
    for split_name, split_images in [("train", train_images), ("val", val_images)]:
        (prepared / split_name / "images").mkdir(parents=True, exist_ok=True)
        (prepared / split_name / "labels").mkdir(parents=True, exist_ok=True)
        for img_path in split_images:
            shutil.copy2(img_path, prepared / split_name / "images" / img_path.name)
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy2(label_path, prepared / split_name / "labels" / label_path.name)

    yaml_content: dict = {
        "path": str(prepared.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": int(data_config.get("nc", 1)),
        "names": list(data_config.get("names", ["object"])),
    }
    kpt_shape = data_config.get("kpt_shape")
    if kpt_shape:
        yaml_content["kpt_shape"] = list(kpt_shape)

    yaml_path = prepared / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    print(f"INFO: data.yaml → {yaml_path}")


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

    _prepare(raw_dir, prepared_dir, OmegaConf.to_container(cfg.data, resolve=True))


if __name__ == "__main__":
    main()
