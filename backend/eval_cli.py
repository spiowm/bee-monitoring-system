"""CLI для запуску behavior evaluation без Mongo і фронтенду.

Приклади:
    uv run python eval_cli.py --pair 20230609b-def
    uv run python eval_cli.py --pair 20230711a-fan --frames 500 --warmup 80
    uv run python eval_cli.py --pair 20230609b-def --json /tmp/result.json
    uv run python eval_cli.py --pair 20230711a-fan --skip-video --frames 250
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_cli")

# Пригнічуємо зайвий шум від ultralytics/supervision під час CLI-запуску
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("supervision").setLevel(logging.ERROR)


def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def _fmt_pct(v: float) -> str:
    pct = v * 100
    color = "32" if pct >= 70 else "33" if pct >= 40 else "31"
    return _color(f"{pct:5.1f}%", color)


def _print_perclass_table(per_class: dict, title: str = "Per-class metrics"):
    print(f"\n  {_color(title, '1;36')}")
    # detR = detection recall (знайдено бджолу), clsR = classification recall (== R)
    # розрив detR - clsR = бджоли задетектовані, але класифіковані неправильно/unknown
    header = (f"  {'Клас':<16} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6} {'FN':>6}  "
              f"{'P':>7} {'R':>7} {'F1':>7}  {'detR':>7} {'→misc':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cls, m in per_class.items():
        det_r = m.get('detection_recall', 0.0)
        # частка задетектованих, але неправильно класифікованих
        misc = max(0.0, det_r - m['recall'])
        print(
            f"  {cls:<16} {m['gt_count']:>6} {m['pred_count']:>6} "
            f"{m['tp']:>6} {m['fp']:>6} {m['fn']:>6}  "
            f"{_fmt_pct(m['precision'])} {_fmt_pct(m['recall'])} {_fmt_pct(m['f1'])}  "
            f"{_fmt_pct(det_r)} {_fmt_pct(misc)}"
        )


def _print_foraging(foraging_events: dict | None):
    if foraging_events is None:
        print(f"\n  {_color('Foraging', '1;33')}: мітки відсутні у цьому кліпі")
        return
    m = foraging_events
    print(f"\n  {_color('Foraging (подієва метрика — counting events)', '1;33')}")
    print(
        f"  GT-подій: {m['gt_count']}  Pred-подій: {m['pred_count']}  "
        f"TP:{m['tp']} FP:{m['fp']} FN:{m['fn']}"
    )
    print(
        f"  Precision: {_fmt_pct(m['precision'])}  "
        f"Recall: {_fmt_pct(m['recall'])}  "
        f"F1: {_fmt_pct(m['f1'])}"
    )


def _print_confusion(cm: dict, eval_classes: list[str]):
    if not cm or not eval_classes:
        return
    all_cols = list(next(iter(cm.values())).keys())
    print(f"\n  {_color('Confusion matrix', '1;35')} (рядки=GT, стовпці=Pred)")
    header = f"  {'GT\\Pred':<14}" + "".join(f"{c[:10]:>11}" for c in all_cols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for gt_cls in eval_classes:
        row_str = f"  {gt_cls:<14}"
        for pred_cls in all_cols:
            v = cm.get(gt_cls, {}).get(pred_cls, 0)
            diag = gt_cls == pred_cls
            cell = _color(f"{v:>11}", "32" if (diag and v > 0) else "31" if (not diag and v > 0) else "")
            row_str += cell
        print(row_str)


def run(args):
    # Імпортуємо після налаштування логування
    from config import settings
    from services.evaluation.gt_loader import gt_paths, load_gt_behaviors, denormalize, load_entrance_zone
    from services.evaluation.behavior_eval import build_behavior_evaluation
    from services.pipeline import VideoPipeline
    from ultralytics import YOLO

    paths = gt_paths(args.pair)
    if not paths["video"].exists():
        sys.exit(f"[error] Відео не знайдено: {paths['video']}")
    if not paths["tracks"].exists():
        sys.exit(f"[error] GT-файл не знайдено: {paths['tracks']}")
    if not paths["entrance_zone"].exists():
        sys.exit(f"[error] entrance_zone не знайдено: {paths['entrance_zone']}")

    video_path = str(paths["video"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[error] Не вдалося відкрити відео: {video_path}")

    total_frames_cap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    limit = args.frames if args.frames and args.frames > 0 else total_frames_cap
    limit = min(limit, total_frames_cap)

    print(f"\n{_color('BuzzTrack Eval CLI', '1;34')}  pair={args.pair}  frames={limit}/{total_frames_cap}  fps={fps:.0f}  {width}×{height}")
    print(f"  warmup={args.warmup}  model={settings.MODEL_PATH}")

    # Завантаження GT
    zone = load_entrance_zone(paths["entrance_zone"])
    gt_df_full = denormalize(load_gt_behaviors(paths["tracks"]), width, height)
    gt_df = gt_df_full[gt_df_full["frame"] <= limit].copy()

    # Завантаження моделі
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"  Завантаження моделі (device={device})...", end=" ", flush=True)
    model = YOLO(settings.MODEL_PATH)
    print(_color("OK", "32"))

    # Конфіг без Mongo
    config = {
        "tracker_name": "bytetrack",
        "approach": "A",
        "line_position": 0.0,
        "conf_threshold": 0.2,
        "iou_threshold": 0.8,
        "max_detections": 1000,
        "kp_conf_threshold": 0.5,
        "track_tail_length": 30,
        "ramp_detect_interval": 30,
        "behavior_fanning_duration_min": 1.0,
        "behavior_eval_warmup_frames": args.warmup,
    }

    pipeline = VideoPipeline(model, config, {}, gt_entrance_zone=zone)

    print(f"  Обробка відео...", end=" ", flush=True)
    t0 = time.perf_counter()

    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_num >= limit:
            break
        frame_num += 1
        pipeline.process_frame(frame, frame_num, fps)

        if frame_num % 200 == 0:
            elapsed = time.perf_counter() - t0
            fps_proc = frame_num / elapsed if elapsed > 0 else 0
            print(f"\r  Обробка відео... {frame_num}/{limit} ({fps_proc:.1f} fps)", end="", flush=True)

    cap.release()
    elapsed = time.perf_counter() - t0
    fps_proc = frame_num / elapsed if elapsed > 0 else 0
    print(f"\r  Оброблено {frame_num} кадрів за {elapsed:.1f}с ({fps_proc:.1f} fps процесингу) {_color('✓', '32')}")

    result = pipeline.get_result(frame_num, elapsed)
    pred_per_frame = result.get("per_frame_behaviors", {})
    pred_events = result.get("events", []) or []

    # Evaluation
    doc = build_behavior_evaluation(
        gt_df, pred_per_frame, pred_events, zone, fps,
        warmup_frames=args.warmup,
    )

    # --- Звіт ---
    print(f"\n{'='*60}")
    print(f"  Загальна точність (per-frame fanning/wash/defense): {_fmt_pct(doc['overall_accuracy'])}")
    print(f"  GT-мічених рядків (після warm-up): {doc['total_gt_labeled']}")
    print(f"  Зіставлено через IoU: {doc['total_matched']}")
    print(f"  Виключено multi-label рядків: {doc['excluded_multilabel']}")
    print(f"  Warm-up: {doc['warmup_frames']} кадрів")
    print(f"  Pred-подій counting: {len(pred_events)}")

    _print_perclass_table(doc["per_class"])
    _print_foraging(doc["foraging_events"])
    _print_confusion(doc["confusion_matrix"], doc["eval_classes"])
    print(f"{'='*60}\n")

    if args.json:
        out_path = Path(args.json)
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        print(f"  JSON збережено: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Behavior evaluation без Mongo — запускає пайплайн і рахує метрики",
    )
    parser.add_argument("--pair", required=True, help="Basename датасету (напр. 20230609b-def)")
    parser.add_argument("--frames", type=int, default=0, help="Обмежити кількість кадрів (0=всі)")
    parser.add_argument("--warmup", type=int, default=80, help="Кадрів warm-up для fanning (default: 80)")
    parser.add_argument("--json", default="", help="Шлях для збереження JSON-результату")
    parser.add_argument("--verbose", action="store_true", help="Детальний лог")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("eval_cli").setLevel(logging.DEBUG)

    run(args)


if __name__ == "__main__":
    main()
