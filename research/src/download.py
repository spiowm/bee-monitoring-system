import sys
from pathlib import Path
from dotenv import load_dotenv
from data_manager import download_dataset

_DATASETS = {
    "pose": ("spiowm/bee-monitoring-pose", "datasets/raw/pose"),
    "ramp": ("spiowm/bee-monitoring-ramp-detection", "datasets/raw/ramp"),
}

if __name__ == "__main__":
    load_dotenv(Path(__file__).parent.parent / ".env")
    keys = sys.argv[1:] or list(_DATASETS.keys())
    for key in keys:
        if key not in _DATASETS:
            print(f"ERROR: невідомий датасет '{key}'. Доступні: {list(_DATASETS.keys())}")
            sys.exit(1)
        kaggle_id, raw_dir = _DATASETS[key]
        download_dataset(kaggle_id, raw_dir)
