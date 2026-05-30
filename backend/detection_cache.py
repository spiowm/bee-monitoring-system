"""Кеш сирих YOLO-детекцій для швидкої ітерації над behavior/defense логікою.

YOLO-інференс — найдорожча частина пайплайну (~7 хв на повне відео).
Детекції + keypoints НЕ залежать від behavior/defense/stitch параметрів,
тому їх можна обчислити один раз і перевикористовувати для десятків
експериментів над класифікацією (секунди замість хвилин).

CachedResult мімікрує мінімальний інтерфейс ultralytics Results, який
використовує DetectionStage, тож кешований шлях ідентичний повному
(окрім самого виклику моделі).
"""
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

CACHE_DIR = Path(__file__).resolve().parent / "data" / "eval_cache"


class _Arr:
    """Мімікрує torch-тензор: .cpu().numpy() повертає numpy-масив."""
    __slots__ = ("_a",)

    def __init__(self, a):
        self._a = a

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Boxes:
    __slots__ = ("xyxy", "conf", "cls")

    def __init__(self, xyxy, conf, cls):
        self.xyxy = _Arr(xyxy)
        self.conf = _Arr(conf)
        self.cls = _Arr(cls)

    def __len__(self):
        return len(self.xyxy._a)


class _Kpts:
    __slots__ = ("xy",)

    def __init__(self, xy):
        self.xy = _Arr(xy)


class CachedResult:
    """Стенд-ін для ultralytics Results — лише поля, які читає DetectionStage."""
    __slots__ = ("boxes", "keypoints")

    def __init__(self, xyxy, conf, cls, kpts):
        self.boxes = _Boxes(xyxy, conf, cls)
        self.keypoints = _Kpts(kpts) if kpts is not None else None


def cache_path(pair: str) -> Path:
    return CACHE_DIR / f"{pair}.pkl"


def _meta(conf, iou, max_det, imgsz, frames, model_path) -> dict:
    mp = Path(model_path)
    return {
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),
        "imgsz": int(imgsz),
        "frames": int(frames),
        "model_path": str(model_path),
        "model_mtime": mp.stat().st_mtime if mp.exists() else 0.0,
    }


def is_valid(pair: str, conf, iou, max_det, imgsz, frames, model_path) -> bool:
    p = cache_path(pair)
    if not p.exists():
        return False
    try:
        with open(p, "rb") as f:
            blob = pickle.load(f)
    except Exception:
        return False
    want = _meta(conf, iou, max_det, imgsz, frames, model_path)
    have = blob.get("meta", {})
    # frames у кеші має бути >= потрібного (можемо взяти префікс)
    if have.get("frames", 0) < want["frames"]:
        return False
    for k in ("conf", "iou", "max_det", "imgsz", "model_path", "model_mtime"):
        if have.get(k) != want[k]:
            return False
    return True


def build_cache(pair: str, video_path: str, model, *, conf, iou, max_det, imgsz,
                frames, device, model_path, progress=True) -> Path:
    """Прогоняє YOLO по відео один раз, зберігає сирі детекції покадрово."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    per_frame = []
    t0 = time.perf_counter()
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_num >= frames:
            break
        frame_num += 1
        res = model(frame, verbose=False, conf=conf, iou=iou, max_det=max_det,
                    imgsz=imgsz, device=device)
        r = res[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            cf = r.boxes.conf.cpu().numpy().astype(np.float32)
            cl = r.boxes.cls.cpu().numpy().astype(np.float32)
            kp = (r.keypoints.xy.cpu().numpy().astype(np.float32)
                  if r.keypoints is not None else None)
        else:
            xyxy = np.zeros((0, 4), np.float32)
            cf = np.zeros((0,), np.float32)
            cl = np.zeros((0,), np.float32)
            kp = None
        per_frame.append((xyxy, cf, cl, kp))

        if progress and frame_num % 200 == 0:
            el = time.perf_counter() - t0
            fps = frame_num / el if el > 0 else 0
            print(f"\r  Кешування YOLO {pair}... {frame_num}/{frames} ({fps:.1f} fps)",
                  end="", flush=True)
    cap.release()
    if progress:
        el = time.perf_counter() - t0
        print(f"\r  Кеш {pair}: {frame_num} кадрів за {el:.1f}с"
              f" ({frame_num/el:.1f} fps){' '*20}")

    blob = {
        "meta": _meta(conf, iou, max_det, imgsz, frame_num, model_path),
        "frames": per_frame,
    }
    p = cache_path(pair)
    with open(p, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_cache(pair: str) -> list:
    """Повертає список CachedResult (index = frame_num - 1)."""
    with open(cache_path(pair), "rb") as f:
        blob = pickle.load(f)
    out = []
    for (xyxy, cf, cl, kp) in blob["frames"]:
        out.append(CachedResult(xyxy, cf, cl, kp))
    return out
