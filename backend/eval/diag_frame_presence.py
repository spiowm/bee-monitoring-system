"""Переоцінка НАШИХ передбачень за МЕТРИКОЮ СТАТТІ (frame-presence, multi-label).

Стаття рахує поведінку на рівні КАДРУ: чи присутній клас у кадрі взагалі
(multi-label, + клас background). Наша звичайна метрика — per-bee IoU F1 (значно
суворіша). Цей скрипт рахує наші ж передбачення їхнім способом для чесного
порівняння з їхніми 87% / 91%.
"""
import argparse
from collections import defaultdict

import cv2
import numpy as np

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

CLASSES = ["foraging", "fanning", "washboarding", "defense", "background"]
BEHAV = {"foraging", "fanning", "washboarding", "defense"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="20230609b-def,20230711a-fan,20230711b-fan")
    args = ap.parse_args()

    from pathlib import Path as _P
    from config import settings
    from services.evaluation.gt_loader import gt_paths, load_gt_behaviors, denormalize, load_entrance_zone
    from services.pipeline import VideoPipeline
    from eval import detection_cache as dc
    from eval.eval_fast import base_config, load_yaml_config

    config = base_config(80)
    config.update(load_yaml_config(str(_P(__file__).resolve().parent.parent / "config" / "eval_config.yaml")))

    grand = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for pair in args.pairs.split(","):
        pair = pair.strip()
        paths = gt_paths(pair)
        video_path = str(paths["video"])
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        cached = dc.load_cache(pair)
        zone = load_entrance_zone(paths["entrance_zone"])
        gt_df = denormalize(load_gt_behaviors(paths["tracks"]), w, h)
        gt_df = gt_df[gt_df["frame"] <= total]
        # GT behaviors present per frame
        gt_present = defaultdict(set)
        for fr, grp in gt_df.groupby("frame"):
            s = {b for b in grp["gt_behavior"].unique() if b in BEHAV}
            gt_present[int(fr)] = s

        pipeline = VideoPipeline(None, config, {}, gt_entrance_zone=zone)
        cap = cv2.VideoCapture(video_path)
        pred_present = defaultdict(set)
        fn = 0
        while True:
            ret, frame = cap.read()
            if not ret or fn >= total:
                break
            det = cached[fn] if fn < len(cached) else None
            fn += 1
            pipeline.process_frame(frame, fn, fps, detection_result=det)
        cap.release()
        pfb = pipeline.pipeline_state.get("per_frame_behaviors", {})
        for f_num, bees in pfb.items():
            s = {b["behavior"] for b in bees.values() if b["behavior"] in BEHAV}
            pred_present[f_num] = s

        # Per-frame multi-label presence over всі кадри відео
        stat = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for f_num in range(1, total + 1):
            g = gt_present.get(f_num, set())
            p = pred_present.get(f_num, set())
            if not g:
                g = {"background"}
            if not p:
                p = {"background"}
            for c in CLASSES:
                in_g, in_p = c in g, c in p
                if in_g and in_p: stat[c]["tp"] += 1
                elif in_p and not in_g: stat[c]["fp"] += 1
                elif in_g and not in_p: stat[c]["fn"] += 1

        print(f"\n=== {pair} (frame-presence, метрика статті) ===")
        print(f"  {'клас':<13}{'P':>7}{'R':>7}{'F1':>7}{'TP':>8}{'FP':>8}{'FN':>8}")
        for c in CLASSES:
            t, fp, fnn = stat[c]["tp"], stat[c]["fp"], stat[c]["fn"]
            pr = t / (t + fp) if t + fp else 0
            rc = t / (t + fnn) if t + fnn else 0
            f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
            print(f"  {c:<13}{pr*100:6.1f}%{rc*100:6.1f}%{f1*100:6.1f}%{t:>8}{fp:>8}{fnn:>8}")
            for k in ("tp", "fp", "fn"):
                grand[c][k] += stat[c][k]

    print("\n=== УСІ 3 ВІДЕО разом (frame-presence) ===")
    print(f"  {'клас':<13}{'P':>7}{'R':>7}{'F1':>7}")
    f1s = []
    for c in CLASSES:
        t, fp, fnn = grand[c]["tp"], grand[c]["fp"], grand[c]["fn"]
        pr = t / (t + fp) if t + fp else 0
        rc = t / (t + fnn) if t + fnn else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        f1s.append(f1)
        print(f"  {c:<13}{pr*100:6.1f}%{rc*100:6.1f}%{f1*100:6.1f}%")
    print(f"  macro-F1 (5 класів) = {sum(f1s)/len(f1s)*100:.1f}%")


if __name__ == "__main__":
    main()
