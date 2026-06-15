"""Balanced frame-presence evaluation — відтворення методики Sledevič et al.

Для кожного відео відбираємо по N=2000 кадрів на клас (balanced sampling):
  - foraging : кадри де хоча б одна бджола має arrival=1 (class_for=1)
  - fanning  : кадри де хоча б одна бджола має fanning=1 (class_fan=1)
  - defense  : кадри де хоча б одна бджола має defensive=1 (class_def=1)
  - background: кадри де ВСІ бджоли мають всі 4 флаги = 0

Запуск (з теки backend/):
    uv run python -m eval.balanced_fp_eval
    uv run python -m eval.balanced_fp_eval --n 2000 --seed 42
    uv run python -m eval.balanced_fp_eval --json /tmp/balanced_result.json
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("supervision").setLevel(logging.ERROR)

DEFAULT_PAIRS = ["20230711a-fan", "20230711b-fan", "20230609b-def"]
CLASSES = ["foraging", "fanning", "defense", "background"]
BEHAV_CLASSES = {"foraging", "fanning", "defense"}

# GT column names → behavior label
GT_COL_MAP = {
    "arrival":    "foraging",
    "fanning":    "fanning",
    "defensive":  "defense",
    "washboarding": "washboarding",
}


def _fmt(v: float, w: int = 6) -> str:
    return f"{v*100:{w}.1f}%"


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def collect_frame_pools(gt_df, total_frames: int) -> dict[str, list[int]]:
    """Для кожного класу зібрати список кадрів де клас присутній."""
    # Для кожного кадру обчислити які класи GT-присутні (без пріоритету, просто наявність флагу)
    frame_flags: dict[int, dict[str, bool]] = {}
    for fr, grp in gt_df.groupby("frame"):
        flags = {
            "foraging":  bool((grp["arrival"] == 1).any()),
            "fanning":   bool((grp["fanning"] == 1).any()),
            "defense":   bool((grp["defensive"] == 1).any()),
            "all_zero":  bool(
                ((grp["arrival"] == 0) & (grp["fanning"] == 0) &
                 (grp["defensive"] == 0) & (grp["washboarding"] == 0)).all()
            ),
        }
        frame_flags[int(fr)] = flags

    pools: dict[str, list[int]] = {c: [] for c in CLASSES}

    for f in range(1, total_frames + 1):
        flags = frame_flags.get(f)
        if flags is None:
            # Кадр без жодних GT-анотацій → може бути background
            pools["background"].append(f)
            continue
        if flags["foraging"]:
            pools["foraging"].append(f)
        if flags["fanning"]:
            pools["fanning"].append(f)
        if flags["defense"]:
            pools["defense"].append(f)
        if flags["all_zero"]:
            pools["background"].append(f)

    return pools


def sample_frames(pools: dict[str, list[int]], n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Рандомно відібрати до N кадрів на клас."""
    sampled: dict[str, np.ndarray] = {}
    for cls, frames in pools.items():
        arr = np.array(frames, dtype=np.int64)
        if len(arr) > n:
            idx = rng.choice(len(arr), size=n, replace=False)
            arr = arr[idx]
        sampled[cls] = arr
    return sampled


def run_pipeline_for_video(pair: str, config: dict, total_frames: int):
    """Запускає pipeline із кешу для одного відео. Повертає pred_per_frame dict."""
    from config import settings
    from services.evaluation.gt_loader import gt_paths, load_entrance_zone
    from services.pipeline import VideoPipeline
    from eval import detection_cache as dc

    paths = gt_paths(pair)
    video_path = str(paths["video"])
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    cap.release()

    cached = dc.load_cache(pair)
    zone = load_entrance_zone(paths["entrance_zone"])

    pipeline = VideoPipeline(None, config, {}, gt_entrance_zone=zone)

    cap = cv2.VideoCapture(video_path)
    fn = 0
    while True:
        ret, frame = cap.read()
        if not ret or fn >= total_frames:
            break
        det = cached[fn] if fn < len(cached) else None
        fn += 1
        pipeline.process_frame(frame, fn, fps, detection_result=det)
    cap.release()

    pred_per_frame = pipeline.pipeline_state.get("per_frame_behaviors", {})
    return pred_per_frame, fps


def compute_balanced_fp_metrics(
    sampled: dict[str, np.ndarray],
    pred_per_frame: dict,
) -> dict:
    """
    Для кожного відібраного кадру перевірити frame-presence:
    GT-клас даного семплу присутній у pred чи ні.

    Повертає confusion_matrix та per_class P/R/F1.
    """
    # Pred: набір класів у кожному кадрі
    pred_present: dict[int, set] = {}
    for f_num, bees in pred_per_frame.items():
        s = {b["behavior"] for b in bees.values() if b.get("behavior") in BEHAV_CLASSES}
        if not s:
            s = {"background"}
        pred_present[int(f_num)] = s

    # Для balanced evaluation: кожен семпл(кадр) має ОДИН GT-клас (той, за яким він семплований)
    # Але у pred може бути кілька класів → перевіряємо чи GT-клас є серед pred

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    # confusion: gt_cls → pred_cls → count
    # Спрощена: для кожного кадру GT=cls, pred=set → якщо cls in pred → TP, інакше FN
    # FP для pred-класу cls = кількість разів коли pred містить cls, але GT != cls
    # (підраховуємо з усіх семплів)

    # Будуємо per-frame GT-класи для семплів
    # Кожен кадр може з'явитись у кількох пулах (напр. foraging і background одночасно)
    # Тому підраховуємо окремо для кожного семплу

    # Збираємо всі (f, gt_cls) пари
    sample_list: list[tuple[int, str]] = []
    for cls, frames in sampled.items():
        for f in frames:
            sample_list.append((int(f), cls))

    for f, gt_cls in sample_list:
        p = pred_present.get(f, {"background"})
        if gt_cls in p:
            tp[gt_cls] += 1
        else:
            fn[gt_cls] += 1

    # FP для кожного класу = кількість семплів де pred містить cls але GT != cls
    for f, gt_cls in sample_list:
        p = pred_present.get(f, {"background"})
        for pred_cls in p:
            if pred_cls in CLASSES and pred_cls != gt_cls:
                fp[pred_cls] += 1

    per_class = {}
    f1s = []
    for cls in CLASSES:
        t, fpp, fnn = tp[cls], fp[cls], fn[cls]
        prec, rec, f1 = _prf(t, fpp, fnn)
        per_class[cls] = {
            "tp": t, "fp": fpp, "fn": fnn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "n_samples": len(sampled.get(cls, [])),
        }
        if cls in BEHAV_CLASSES:
            f1s.append(f1)

    macro_f1_behav = sum(f1s) / len(f1s) if f1s else 0.0
    macro_f1_all = sum(per_class[c]["f1"] for c in CLASSES) / len(CLASSES)

    return {
        "per_class": per_class,
        "macro_f1_behavior": round(macro_f1_behav, 4),
        "macro_f1_all": round(macro_f1_all, 4),
    }


def print_pool_stats(pair: str, pools: dict[str, list[int]]):
    print(f"\n  [{pair}] Доступно кадрів перед семплюванням:")
    for cls in CLASSES:
        n = len(pools[cls])
        flag = "  ⚠ < 2000" if n < 2000 else ""
        print(f"    {cls:<12}: {n:>6}{flag}")


def print_metrics_table(label: str, result: dict, sampled: dict | None = None):
    print(f"\n  {label}")
    hdr = f"  {'Клас':<14} {'N':>6} {'TP':>6} {'FP':>6} {'FN':>6}   {'P':>7} {'R':>7} {'F1':>7}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cls in CLASSES:
        m = result["per_class"][cls]
        n = m["n_samples"]
        print(f"  {cls:<14} {n:>6} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6}   "
              f"{_fmt(m['precision'])} {_fmt(m['recall'])} {_fmt(m['f1'])}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'macro-F1 (3 behav)':<14}                              "
          f"               {_fmt(result['macro_f1_behavior'])}")
    print(f"  {'macro-F1 (all 4)':<14}                              "
          f"               {_fmt(result['macro_f1_all'])}")


def main():
    ap = argparse.ArgumentParser(description="Balanced frame-presence evaluation (Sledevič et al. method)")
    ap.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    ap.add_argument("--n", type=int, default=2000, help="Max кадрів на клас (default=2000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=80)
    ap.add_argument("--json", default="", help="Зберегти JSON-результат")
    args = ap.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    rng = np.random.default_rng(args.seed)

    # --- Конфіг pipeline ---
    from eval.eval_fast import base_config, load_yaml_config
    DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "config" / "eval_config.yaml")
    config = base_config(args.warmup)
    config.update(load_yaml_config(DEFAULT_CONFIG))

    # Завантаження GT і відео-мета
    from config import settings
    from services.evaluation.gt_loader import gt_paths, load_gt_behaviors, denormalize

    print(f"\n{'='*72}")
    print(f"  BALANCED FRAME-PRESENCE EVAL  (Sledevič et al. method)")
    print(f"  seed={args.seed}  N_max={args.n}  pairs={pairs}")
    print(f"{'='*72}")

    all_results = {}
    grand_pools: dict[str, list[int]] = {c: [] for c in CLASSES}
    grand_sampled: dict[str, list[int]] = {c: [] for c in CLASSES}
    grand_pred: dict[int, dict] = {}  # глобальний не потрібен (різні відео)

    # Для агрегованих метрик зберігаємо tp/fp/fn по парах
    grand_tp = defaultdict(int)
    grand_fp = defaultdict(int)
    grand_fn = defaultdict(int)

    # === КРОК 1: Статистика пулів ===
    print(f"\n{'─'*72}")
    print(f"  1. СТАТИСТИКА ПУЛІВ (кадрів доступно перед семплюванням)")
    print(f"{'─'*72}")

    pair_meta = {}
    for pair in pairs:
        paths = gt_paths(pair)
        cap = cv2.VideoCapture(str(paths["video"]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        gt_df = denormalize(load_gt_behaviors(paths["tracks"]), w, h)
        pools = collect_frame_pools(gt_df, total)
        print_pool_stats(pair, pools)

        pair_meta[pair] = {
            "gt_df": gt_df,
            "total": total,
            "pools": pools,
        }

    # === КРОК 2: Семплювання + запуск pipeline ===
    print(f"\n{'─'*72}")
    print(f"  2. BALANCED FRAME-PRESENCE (seed={args.seed}, N_max={args.n})")
    print(f"{'─'*72}")

    for pair in pairs:
        meta = pair_meta[pair]
        pools = meta["pools"]
        gt_df = meta["gt_df"]
        total = meta["total"]

        # Семплювання
        sampled = sample_frames(pools, args.n, rng)

        print(f"\n  [{pair}] Семпли (після відбору):")
        for cls in CLASSES:
            print(f"    {cls:<12}: {len(sampled[cls]):>5} кадрів")

        # Запуск pipeline
        print(f"  [{pair}] Запуск pipeline (з кешу)...", end=" ", flush=True)
        pred_per_frame, fps = run_pipeline_for_video(pair, config, total)
        print("OK")

        # Метрики
        result = compute_balanced_fp_metrics(sampled, pred_per_frame)
        all_results[pair] = result

        print_metrics_table(f"[{pair}] Balanced frame-presence metrics:", result, sampled)

        # Агрегат
        for cls in CLASSES:
            m = result["per_class"][cls]
            grand_tp[cls] += m["tp"]
            grand_fp[cls] += m["fp"]
            grand_fn[cls] += m["fn"]

    # === КРОК 3: Агрегована таблиця ===
    print(f"\n{'─'*72}")
    print(f"  3. АГРЕГОВАНА ЗВЕДЕНА ТАБЛИЦЯ")
    print(f"{'─'*72}")

    # Balanced FP агрегат
    agg_per_class = {}
    f1s_behav = []
    f1s_all = []
    for cls in CLASSES:
        t, fpp, fnn = grand_tp[cls], grand_fp[cls], grand_fn[cls]
        prec, rec, f1 = _prf(t, fpp, fnn)
        agg_per_class[cls] = {
            "tp": t, "fp": fpp, "fn": fnn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "n_samples": t + fnn,
        }
        if cls in BEHAV_CLASSES:
            f1s_behav.append(f1)
        f1s_all.append(f1)

    agg_result = {
        "per_class": agg_per_class,
        "macro_f1_behavior": round(sum(f1s_behav) / len(f1s_behav), 4) if f1s_behav else 0.0,
        "macro_f1_all": round(sum(f1s_all) / len(f1s_all), 4),
    }

    print_metrics_table("АГРЕГАТ (всі 3 відео):", agg_result)

    # Порівняння з existing eval_fast (full-video frame-presence)
    print(f"\n{'─'*72}")
    print(f"  4. ПОРІВНЯЛЬНА ТАБЛИЦЯ (markdown)")
    print(f"{'─'*72}")

    print()
    print("### Balanced Frame-Presence (Sledevič et al. methodology, N=2000/class)")
    print()
    print(f"| Відео | Клас | N | P | R | F1 |")
    print(f"|-------|------|---|---|---|-----|")
    for pair in pairs:
        r = all_results[pair]
        for cls in CLASSES:
            m = r["per_class"][cls]
            print(f"| {pair} | {cls} | {m['n_samples']} | "
                  f"{m['precision']*100:.1f}% | {m['recall']*100:.1f}% | {m['f1']*100:.1f}% |")
        macro = r['macro_f1_behavior']
        print(f"| {pair} | **macro-F1 (behav)** | — | — | — | **{macro*100:.1f}%** |")

    print()
    print("#### Агреговано (всі 3 відео):")
    print()
    print(f"| Клас | TP | FP | FN | P | R | F1 |")
    print(f"|------|----|----|----|---|---|-----|")
    for cls in CLASSES:
        m = agg_result["per_class"][cls]
        print(f"| {cls} | {m['tp']} | {m['fp']} | {m['fn']} | "
              f"{m['precision']*100:.1f}% | {m['recall']*100:.1f}% | {m['f1']*100:.1f}% |")
    print(f"| **macro-F1 (3 behav)** | | | | | | "
          f"**{agg_result['macro_f1_behavior']*100:.1f}%** |")
    print(f"| **macro-F1 (all 4)** | | | | | | "
          f"**{agg_result['macro_f1_all']*100:.1f}%** |")

    print()
    print("### Порівняння: full-video frame-presence (eval_fast, без балансування)")
    print()
    print("*(ці числа беруться з eval_fast.py — потрібно запустити eval_fast щоб мати актуальні)*")
    print()
    print(f"| Відео | forF1 | fanF1 | defF1 | bgF1 | macroF1 |")
    print(f"|-------|-------|-------|-------|------|---------|")
    print("| 20230711a-fan | — | — | — | — | — |")
    print("| 20230711b-fan | — | — | — | — | — |")
    print("| 20230609b-def | — | — | — | — | — |")
    print("| **mean** | | | | | |")
    print()
    print("*Запустіть: `uv run python -m eval.eval_fast --json /tmp/fast_result.json` для отримання чисел*")

    if args.json:
        out = {
            "config": {"n": args.n, "seed": args.seed, "pairs": pairs},
            "per_video": all_results,
            "aggregate": agg_result,
        }
        Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n  JSON → {args.json}")

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()
