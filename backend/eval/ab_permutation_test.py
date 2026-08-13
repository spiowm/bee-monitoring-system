"""Перестановочний тест: чи відрізняється pose-фільтр від випадкового відсіву.

Нульова гіпотеза: кут між вектором руху і вектором пози не несе інформації про
те, чи є подія перетину істинною. Реалізація нульової гіпотези — перестановка
значень кута між кандидатами на перетин (кількість подій і розподіл кутів
зберігаються, руйнується лише звʼязок «кут ↔ конкретна подія»).

Статистика — найкраще F1 підрахунку по розгортці порога. Для кожної перестановки
поріг оптимізується так само, як для реальних даних, тому тест порівнює
однаково налаштовані варіанти.

Вхід — `sweep_real.json` від `ab_threshold_sweep.py` (містить кути та події).

Запуск з backend/:
    uv run python -m eval.ab_permutation_test --sweep sweep_real.json --n 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.ab_counting_eval import gt_events, load_zone, match  # noqa: E402

THRESHOLDS = [15, 30, 45, 60, 90, 120, 150, 180]


def best_f1(cands: list[dict], gt: list[dict], angles: np.ndarray, window: int) -> float:
    best = 0.0
    for t in THRESHOLDS:
        kept = [c for c, a in zip(cands, angles)
                if (a != a) or a <= t]          # NaN → подія лишається (немає пози)
        f1 = match(gt, kept, window)["f1"]
        best = max(best, f1)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="sweep_real.json")
    ap.add_argument("--n", type=int, default=1000, help="кількість перестановок")
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--out", default="ab_permutation_test.json")
    args = ap.parse_args()

    data = json.loads(Path(args.sweep).read_text())
    rng = np.random.default_rng(42)
    report = {}

    for pair, rec in data.items():
        angles = np.array([np.nan if a is None else a for a in rec["angles_ema"]])
        # кандидати відновлюємо з розгортки: порядок збережено у angles_ema
        zone = load_zone(pair)
        gt = gt_events(pair, zone, 10 ** 9)
        # реконструюємо мінімальні поля кандидата з sweep-даних
        n = len(angles)
        cands = [{"frame": f, "direction": d}
                 for f, d in zip(rec["frames"], rec["directions"])] \
            if "frames" in rec else None
        if cands is None:
            print(f"{pair}: у sweep немає полів frames/directions — пропущено")
            continue

        observed = best_f1(cands, gt, angles, args.window)
        null = np.array([best_f1(cands, gt, rng.permutation(angles), args.window)
                         for _ in range(args.n)])
        # поправка Монте-Карло: p = (k+1)/(n+1), уникає нульових оцінок
        k = int((null >= observed).sum())
        p = (k + 1) / (args.n + 1)
        report[pair] = {"observed": observed, "null_mean": float(null.mean()),
                        "null_std": float(null.std(ddof=1)),
                        "null_p95": float(np.percentile(null, 95)), "p_value": p,
                        "n_permutations": args.n}
        print(f"{pair:<16} спостережене F1={observed*100:5.1f}%  "
              f"нуль={null.mean()*100:5.1f}% ± {null.std(ddof=1)*100:.1f}  "
              f"95-й перцентиль={np.percentile(null,95)*100:5.1f}%  p={p:.3f}")

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nЗбережено: {args.out}")


if __name__ == "__main__":
    main()
