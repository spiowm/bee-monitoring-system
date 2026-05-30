"""Швидкий behavior-eval через кеш YOLO-детекцій.

Проганяє пайплайн (Tracking → Stitch → Behavior → Defense → Counting) із
закешованих детекцій для одного або кількох відео і друкує зведену таблицю.
YOLO викликається лише раз на відео (для побудови кешу); далі — секунди.

Приклади:
    uv run python eval_fast.py                       # всі 3, повні, дефолт-конфіг
    uv run python eval_fast.py --config config/eval_config.yaml
    uv run python eval_fast.py --pairs 20230711a-fan --frames 2000
    uv run python eval_fast.py --rebuild             # перебудувати кеш
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
import yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("supervision").setLevel(logging.ERROR)

from eval_cli import (_color, _fmt_pct, _print_perclass_table, _print_foraging,
                      _print_confusion, _print_frame_presence)

DEFAULT_PAIRS = ["20230609b-def", "20230711a-fan", "20230711b-fan"]


def load_yaml_config(path: str) -> dict:
    if path and Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def base_config(warmup: int) -> dict:
    return {
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
        "behavior_eval_warmup_frames": warmup,
        "skip_annotation": True,
    }


def eval_one(pair, model, config, warmup, frames_arg, rebuild, device):
    """Повертає (doc, meta) для одного відео — будує/вантажить кеш за потреби."""
    from config import settings
    from services.evaluation.gt_loader import gt_paths, load_gt_behaviors, denormalize, load_entrance_zone
    from services.evaluation.behavior_eval import build_behavior_evaluation
    from services.pipeline import VideoPipeline
    import detection_cache as dc

    paths = gt_paths(pair)
    video_path = str(paths["video"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"не вдалося відкрити відео: {video_path}")
    total_frames_cap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    limit = frames_arg if frames_arg and frames_arg > 0 else total_frames_cap
    limit = min(limit, total_frames_cap)

    conf = config["conf_threshold"]
    iou = config["iou_threshold"]
    max_det = config["max_detections"]
    imgsz = config.get("imgsz") or (getattr(model, "bee_meta", None) or {}).get("imgsz") or 640

    # Кеш детекцій
    valid = dc.is_valid(pair, conf, iou, max_det, imgsz, limit, settings.MODEL_PATH)
    if rebuild or not valid:
        if model is None:
            raise RuntimeError("потрібна модель для побудови кешу, але вона не завантажена")
        dc.build_cache(pair, video_path, model, conf=conf, iou=iou, max_det=max_det,
                       imgsz=imgsz, frames=total_frames_cap, device=device,
                       model_path=settings.MODEL_PATH)
    cached = dc.load_cache(pair)

    zone = load_entrance_zone(paths["entrance_zone"])
    gt_df_full = denormalize(load_gt_behaviors(paths["tracks"]), width, height)
    gt_df = gt_df_full[gt_df_full["frame"] <= limit].copy()

    pipeline = VideoPipeline(model, config, {}, gt_entrance_zone=zone)

    t0 = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_num >= limit:
            break
        det = cached[frame_num] if frame_num < len(cached) else None
        frame_num += 1
        pipeline.process_frame(frame, frame_num, fps, detection_result=det)
    cap.release()
    elapsed = time.perf_counter() - t0

    result = pipeline.get_result(frame_num, elapsed)
    pred_per_frame = result.get("per_frame_behaviors", {})
    pred_events = result.get("events", []) or []

    doc = build_behavior_evaluation(gt_df, pred_per_frame, pred_events, zone, fps,
                                    warmup_frames=warmup, total_frames=frame_num)
    meta = {"pair": pair, "frames": frame_num, "elapsed": elapsed,
            "fps_proc": frame_num / elapsed if elapsed > 0 else 0,
            "pred_events": len(pred_events)}
    return doc, meta


def _f1(d):
    return d.get("f1", 0.0) if d else 0.0


def print_scoreboard(results):
    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    # ── СТРОГА МЕТРИКА (per-bee IoU) ────────────────────────────────────────
    print(f"\n{_color('='*80, '1;34')}")
    print(f"  {_color('СТРОГА МЕТРИКА  (per-bee IoU matching — наш стандарт)', '1;34')}")
    print(f"{_color('='*80, '1;34')}")
    hdr = (f"  {'Відео':<18}{'acc':>7}{'fanF1':>8}{'fanP':>7}{'fanR':>7}"
           f"{'defF1(ev)':>11}{'forF1(ev)':>11}{'washF1':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    agg = {"fan_f1": [], "fan_r": [], "def_f1": [], "for_f1": [], "wash_f1": []}
    for pair, doc in results.items():
        pc = doc.get("per_class", {})
        fan = pc.get("fanning", {})
        wash = pc.get("washboarding", {})
        deff = doc.get("defense_events")
        forr = doc.get("foraging_events")
        acc = doc.get("overall_accuracy", 0.0)
        fan_f1 = fan.get("f1", 0.0)
        fan_p = fan.get("precision", 0.0)
        fan_r = fan.get("recall", 0.0)
        def_f1 = _f1(deff)
        for_f1 = _f1(forr)
        wash_f1 = wash.get("f1", 0.0)
        print(f"  {pair:<18}{_fmt_pct(acc)}{_fmt_pct(fan_f1)}{_fmt_pct(fan_p)}"
              f"{_fmt_pct(fan_r)}  {_fmt_pct(def_f1)}  {_fmt_pct(for_f1)}{_fmt_pct(wash_f1)}")
        if fan.get("gt_count", 0) > 0:
            agg["fan_f1"].append(fan_f1)
            agg["fan_r"].append(fan_r)
        if forr and forr.get("gt_count", 0) > 0:
            agg["for_f1"].append(for_f1)
        if deff and deff.get("gt_count", 0) > 0:
            agg["def_f1"].append(def_f1)
        if wash.get("gt_count", 0) > 0:
            agg["wash_f1"].append(wash_f1)

    mean_fan = _mean(agg["fan_f1"])
    mean_def = _mean(agg["def_f1"])
    mean_for = _mean(agg["for_f1"])
    combined = _mean([mean_fan, mean_def, mean_for])
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {_color('СЕРЕДНЄ', '1;36'):<27}"
          f"fanF1={_fmt_pct(mean_fan)}  defF1={_fmt_pct(mean_def)}  "
          f"forF1={_fmt_pct(mean_for)}")
    print(f"  {_color('COMBINED SCORE', '1;32')}: {_fmt_pct(combined)}  "
          f"(mean fanF1+defF1+forF1)")

    # ── FRAME-PRESENCE МЕТРИКА (як у статті Sledevič 2025) ──────────────────
    fp_data = {pair: doc.get("frame_presence") for pair, doc in results.items()
               if doc.get("frame_presence")}
    if fp_data:
        print(f"\n{_color('='*80, '1;35')}")
        print(f"  {_color('FRAME-PRESENCE МЕТРИКА  (multi-label frame-level — як Sledevič 2025)', '1;35')}")
        print(f"{_color('='*80, '1;35')}")
        fp_hdr = (f"  {'Відео':<18}{'forF1':>8}{'fanF1':>8}{'washF1':>8}"
                  f"{'defF1':>8}{'bgF1':>8}{'macroF1':>9}")
        print(fp_hdr)
        print("  " + "-" * (len(fp_hdr) - 2))
        all_macro = []
        for pair, fp in fp_data.items():
            pc = fp["per_class"]
            forf = pc.get("foraging", {}).get("f1", 0)
            fanf = pc.get("fanning", {}).get("f1", 0)
            washf = pc.get("washboarding", {}).get("f1", 0)
            deff = pc.get("defense", {}).get("f1", 0)
            bgf = pc.get("background", {}).get("f1", 0)
            macro = fp["macro_f1_behavior"]
            all_macro.append(macro)
            print(f"  {pair:<18}{_fmt_pct(forf)}{_fmt_pct(fanf)}{_fmt_pct(washf)}"
                  f"{_fmt_pct(deff)}{_fmt_pct(bgf)}{_fmt_pct(macro)}")
        print("  " + "-" * (len(fp_hdr) - 2))
        grand_macro = _mean(all_macro)
        print(f"  {_color('MACRO-F1 (4 поведінки)', '1;35')}: {_fmt_pct(grand_macro)}"
              f"  {_color('← порівнянно з 87% у статті', '2;35')}")
    print(f"{_color('='*80, '1;34')}\n")
    return {"combined": combined, "fan_f1": mean_fan, "def_f1": mean_def,
            "for_f1": mean_for, "fan_r": _mean(agg["fan_r"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=",".join(DEFAULT_PAIRS),
                    help="кома-список пар (default: всі 3)")
    ap.add_argument("--frames", type=int, default=0, help="ліміт кадрів (0=всі)")
    ap.add_argument("--warmup", type=int, default=80)
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override конфігу (можна кілька): --set behavior_fanning_max_disp=60")
    ap.add_argument("--rebuild", action="store_true", help="перебудувати кеш детекцій")
    ap.add_argument("--json", default="", help="зберегти summary JSON")
    ap.add_argument("--tag", default="", help="мітка запуску для логів")
    ap.add_argument("--quiet", action="store_true", help="лише зведена таблиця")
    args = ap.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    yaml_cfg = load_yaml_config(args.config)

    def _coerce(v: str):
        low = v.lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            return v

    for kv in args.set:
        k, _, v = kv.partition("=")
        yaml_cfg[k.strip()] = _coerce(v.strip())

    config = base_config(args.warmup)
    config.update(yaml_cfg)

    device = 0 if torch.cuda.is_available() else "cpu"

    tag = f" [{args.tag}]" if args.tag else ""
    print(f"\n{_color('BuzzTrack Fast-Eval', '1;34')}{tag}  pairs={pairs}  "
          f"frames={'all' if not args.frames else args.frames}  config={args.config}")
    if yaml_cfg:
        print(f"  YAML overrides: {yaml_cfg}")

    # Модель потрібна лише якщо доведеться будувати кеш — завантажуємо лениво
    model = None
    from config import settings
    import detection_cache as dc
    need_model = args.rebuild
    if not need_model:
        for pair in pairs:
            from services.evaluation.gt_loader import gt_paths
            v = str(gt_paths(pair)["video"])
            cap = cv2.VideoCapture(v)
            tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            lim = args.frames if args.frames and args.frames > 0 else tot
            imgsz = config.get("imgsz") or 640
            if not dc.is_valid(pair, config["conf_threshold"], config["iou_threshold"],
                               config["max_detections"], imgsz, min(lim, tot), settings.MODEL_PATH):
                need_model = True
                break
    if need_model:
        from ultralytics import YOLO
        print(f"  Завантаження моделі (device={device})...", end=" ", flush=True)
        model = YOLO(settings.MODEL_PATH)
        print(_color("OK", "32"))

    results = {}
    metas = {}
    for pair in pairs:
        try:
            doc, meta = eval_one(pair, model, config, args.warmup, args.frames,
                                 args.rebuild, device)
            results[pair] = doc
            metas[pair] = meta
            if not args.quiet:
                print(f"\n{_color('▶ '+pair, '1;33')}  "
                      f"{meta['frames']} кадрів за {meta['elapsed']:.1f}с "
                      f"({meta['fps_proc']:.0f} fps)")
                _print_perclass_table(doc["per_class"])
                _print_foraging(doc["foraging_events"])
                if doc.get("defense_events"):
                    m = doc["defense_events"]
                    print(f"\n  {_color('Defense (подієва метрика)', '1;33')}")
                    print(f"  GT:{m['gt_count']} Pred:{m['pred_count']} "
                          f"TP:{m['tp']} FP:{m['fp']} FN:{m['fn']}  "
                          f"P:{_fmt_pct(m['precision'])} R:{_fmt_pct(m['recall'])} "
                          f"F1:{_fmt_pct(m['f1'])}")
                _print_confusion(doc["confusion_matrix"], doc["eval_classes"])
                _print_frame_presence(doc.get("frame_presence"))
        except Exception as e:
            import traceback
            print(f"\n  {_color('ERROR', '31')} {pair}: {e}")
            traceback.print_exc()

    summary = print_scoreboard(results)

    if args.json:
        out = {"tag": args.tag, "config": yaml_cfg, "summary": summary,
               "metas": metas, "results": results}
        Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"  Summary → {args.json}")


if __name__ == "__main__":
    main()
