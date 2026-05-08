"""Завантаження ground-truth анотацій з датасету tracking_and_behavior."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

from config import settings

GT_COLUMNS = [
    "frame", "track_id",
    "cx", "cy", "w", "h",
    "arrival", "defensive", "fanning", "unused",
]


def gt_root() -> Path:
    return Path(settings.GT_DATASET_PATH)


def gt_paths(basename: str) -> dict:
    """Повертає шляхи до файлів GT для заданого basename (нова структура підпапок)."""
    root = gt_root() / basename
    return {
        "video": root / "video.mp4",
        "tracks": root / "tracks_and_behavior.txt",
        "entrance_zone": root / "entrance_zone.txt",
    }


def load_gt_tracks(path: Path) -> pd.DataFrame:
    """Парсить tracks_and_behavior_classes_*.txt → DataFrame з нормалізованими CXCYWH."""
    df = pd.read_csv(path, header=None, names=GT_COLUMNS)
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    return df


def load_entrance_zone(path: Path) -> np.ndarray:
    """Парсить рядок виду `polygon = np.array([[x1,y1], ...])` → (4, 2) numpy array у пікселях."""
    text = path.read_text()
    nums = re.findall(r"-?\d+\.?\d*", text)
    if len(nums) < 8:
        raise ValueError(f"entrance_zone файл {path} має <8 чисел: {text!r}")
    coords = np.array([float(n) for n in nums[:8]], dtype=np.float32).reshape(4, 2)
    return coords


def denormalize(df: pd.DataFrame, width: int, height: int) -> pd.DataFrame:
    """Додає колонки x1,y1,x2,y2,cx_px,cy_px у пікселях.
    Увага: у GT файлах колонки 3 і 4 (cx, cy) насправді є top-left x та top-left y (нормалізовані).
    """
    df = df.copy()
    df["x1"] = df["cx"] * width
    df["y1"] = df["cy"] * height
    w_px = df["w"] * width
    h_px = df["h"] * height
    
    df["x2"] = df["x1"] + w_px
    df["y2"] = df["y1"] + h_px
    df["cx_px"] = df["x1"] + w_px * 0.5
    df["cy_px"] = df["y1"] + h_px * 0.5
    return df


def list_available_pairs() -> list[dict]:
    """Сканує GT-директорію (нова структура підпапок) → список доступних відео+GT пар."""
    root = gt_root()
    if not root.exists():
        return []
    pairs = []
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        video = subdir / "video.mp4"
        tracks = subdir / "tracks_and_behavior.txt"
        zone = subdir / "entrance_zone.txt"
        if video.exists() and tracks.exists() and zone.exists():
            pairs.append({
                "basename": subdir.name,
                "video_filename": f"{subdir.name}/video.mp4",
                "size_mb": round(video.stat().st_size / (1024 * 1024), 1),
            })
    return pairs
