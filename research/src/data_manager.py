import os
import shutil
import random
from pathlib import Path


def _kaggle_token() -> str:
    key = os.getenv("KAGGLE_KEY", "")
    if key.upper().startswith("KGAT") and not os.getenv("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = key
    return os.getenv("KAGGLE_API_TOKEN", "")


def _find_data_root(base: Path) -> Path:
    if (base / "images").exists():
        return base
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and (sub / "images").exists():
            return sub
    raise FileNotFoundError(f"Не знайдено images/ в {base}")


def download_dataset(kaggle_id: str, raw_dir: str | None = None) -> Path:
    if raw_dir:
        local = Path(raw_dir)
        if local.exists() and any(local.iterdir()):
            print(f"INFO: Дані вже є → {local}")
            return local

    import kagglehub
    _kaggle_token()
    print(f"INFO: Завантаження '{kaggle_id}'...")
    downloaded = Path(kagglehub.dataset_download(kaggle_id))
    root = _find_data_root(downloaded)

    if raw_dir:
        local = Path(raw_dir)
        local.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(root), str(local), dirs_exist_ok=True)
        print(f"INFO: Збережено → {local}")
        return local

    print(f"INFO: Дані → {root}")
    return root


def prepare_dataset(raw_dir: Path | str, prepared_dir: str, data_config: dict) -> None:
    import yaml

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
