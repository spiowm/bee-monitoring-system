import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv
import kagglehub

_DATASETS = {
    "pose": ("spiowm/bee-monitoring-pose", "datasets/raw/pose"),
    "ramp": ("spiowm/bee-monitoring-ramp-detection", "datasets/raw/ramp"),
}

def _find_data_root(base: Path) -> Path:
    if (base / "images").exists():
        return base
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and (sub / "images").exists():
            return sub
    raise FileNotFoundError(f"Не знайдено images/ в {base}")


def download(kaggle_id: str, raw_dir: str) -> None:
    local = Path(raw_dir)
    if local.exists() and any(local.iterdir()):
        print(f"INFO: Дані вже є → {local}")
        return

    
    print(f"INFO: Завантаження '{kaggle_id}'...")
    root = _find_data_root(Path(kagglehub.dataset_download(kaggle_id)))
    local.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(root), str(local), dirs_exist_ok=True)
    print(f"INFO: Збережено → {local}")


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent.parent / ".env")
    keys = sys.argv[1:] or list(_DATASETS.keys())
    for key in keys:
        if key not in _DATASETS:
            print(f"ERROR: невідомий датасет '{key}'. Доступні: {list(_DATASETS.keys())}")
            sys.exit(1)
        download(*_DATASETS[key])
