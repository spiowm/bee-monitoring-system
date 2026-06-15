"""
Діагностика: скільки GT foraging-треків фізично перетинають лічильну лінію?

Запуск з backend/:
    uv run python -m eval.diag_foraging_gt_crossings --pair 20230711b-fan
    uv run python -m eval.diag_foraging_gt_crossings --pair 20230711a-fan
    uv run python -m eval.diag_foraging_gt_crossings --pair yt5

Відповідь:
    < 50% → limitation у Methods/Results (GT-foraging бджоли не перетинають лінію)
    > 50% → потенційний bug у системі
"""
import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# ── Константи ──────────────────────────────────────────────────────────────
DEBOUNCE_FRAMES = 30  # з settings.counting_debounce_frames (типове значення)
SHIFT_DOWN = 8        # pixels, як у counter.py та counting_eval.py


GT_COLUMNS = [
    "frame", "track_id",
    "cx", "cy", "w", "h",
    "arrival", "defensive", "fanning", "washboarding",
]

GT_ROOT = Path(__file__).resolve().parent.parent.parent / \
          "research" / "datasets" / "raw" / "tracking_and_behavior"


# ── Helpers ────────────────────────────────────────────────────────────────

def load_entrance_zone(path: Path) -> np.ndarray:
    text = path.read_text()
    nums = re.findall(r"-?\d+\.?\d*", text)
    coords = np.array([float(n) for n in nums[:8]], dtype=np.float32).reshape(4, 2)
    return coords


def get_line_y_for_cx(zone: np.ndarray, cx: float) -> float:
    """Верхній край entrance_zone, інтерпольований по X."""
    x1, y1 = float(zone[0][0]), float(zone[0][1])
    x2, y2 = float(zone[1][0]), float(zone[1][1])
    if abs(x2 - x1) > 1e-3:
        return y1 + (y2 - y1) / (x2 - x1) * (cx - x1) + SHIFT_DOWN
    return y1 + SHIFT_DOWN


def load_gt(pair: str):
    root = GT_ROOT / pair
    tracks_path = root / "tracks_and_behavior.txt"
    zone_path   = root / "entrance_zone.txt"
    if not tracks_path.exists():
        sys.exit(f"[error] tracks_and_behavior.txt не знайдено: {tracks_path}")
    if not zone_path.exists():
        sys.exit(f"[error] entrance_zone.txt не знайдено: {zone_path}")

    df = pd.read_csv(tracks_path, header=None, names=GT_COLUMNS)
    df["frame"]    = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)

    zone = load_entrance_zone(zone_path)
    return df, zone


def get_video_dims(pair: str):
    """Повертає (width, height) відео через ffprobe (або жорсткий fallback)."""
    try:
        import subprocess
        vpath = GT_ROOT / pair / "video.mp4"
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", str(vpath)],
            text=True
        ).strip()
        w, h = out.split(",")
        return int(w), int(h)
    except Exception:
        # Fallback: якщо ffprobe недоступний
        return 1920, 1080


# ── Ядро діагностики ───────────────────────────────────────────────────────

def diagnose(pair: str, verbose: bool = False):
    print(f"\n{'='*65}")
    print(f"  Foraging GT line-crossing diagnostic  ▶  {pair}")
    print(f"{'='*65}")

    df, zone = load_gt(pair)
    W, H = get_video_dims(pair)

    print(f"  Відео: {W}×{H}")
    print(f"  Entrance zone (пікселі): {zone.tolist()}")

    # Денормалізація (cx, cy в GT насправді top-left, не центр — як у gt_loader.py)
    df = df.copy()
    df["x1_px"]  = df["cx"] * W
    df["y1_px"]  = df["cy"] * H
    df["w_px"]   = df["w"]  * W
    df["h_px"]   = df["h"]  * H
    df["cx_px"]  = df["x1_px"] + df["w_px"] * 0.5
    df["cy_px"]  = df["y1_px"] + df["h_px"] * 0.5

    # Усі унікальні GT foraging-треки
    foraging_df = df[df["arrival"] == 1]
    all_foraging_tracks = set(foraging_df["track_id"].unique())
    n_foraging_tracks = len(all_foraging_tracks)

    # Усі унікальні треки взагалі
    n_total_tracks = df["track_id"].nunique()
    n_total_foraging_rows = len(foraging_df)

    print(f"\n  Загальна статистика датасету:")
    print(f"    Рядків всього:            {len(df):>8,}")
    print(f"    Унікальних треків:         {n_total_tracks:>8,}")
    print(f"    Рядків з arrival=1:        {n_total_foraging_rows:>8,}")
    print(f"    Унікальних foraging-треків: {n_foraging_tracks:>7,}")

    # Для кожного foraging-треку перевіряємо перетин лінії
    crossed_tracks     = []
    not_crossed_tracks = []

    track_stats = []

    for track_id, track_df in df.groupby("track_id"):
        track_df = track_df.sort_values("frame")
        is_foraging_track = track_id in all_foraging_tracks

        ys = track_df["cy_px"].to_numpy()
        xs = track_df["cx_px"].to_numpy()
        frames = track_df["frame"].to_numpy()

        if len(ys) < 2:
            if is_foraging_track:
                not_crossed_tracks.append(track_id)
            continue

        # Перевірка перетину лінії
        crossed = False
        crossing_frames = []
        last_event_frame = -9999

        for i in range(1, len(ys)):
            prev_y, curr_y = ys[i - 1], ys[i]
            curr_x = xs[i]
            line_y = get_line_y_for_cx(zone, curr_x)

            crossed_down = prev_y < line_y <= curr_y
            crossed_up   = prev_y > line_y >= curr_y

            if crossed_down or crossed_up:
                frame = int(frames[i])
                if frame - last_event_frame >= DEBOUNCE_FRAMES:
                    crossed = True
                    direction = "OUT" if crossed_down else "IN"
                    crossing_frames.append((frame, direction, round(float(curr_y), 1), round(float(line_y), 1)))
                    last_event_frame = frame

        # Діапазон Y цього треку
        y_min = float(ys.min())
        y_max = float(ys.max())
        line_y_mid = get_line_y_for_cx(zone, float(xs.mean()))

        stat = {
            "track_id": track_id,
            "is_foraging": is_foraging_track,
            "crossed": crossed,
            "n_frames": len(ys),
            "y_min": round(y_min, 1),
            "y_max": round(y_max, 1),
            "line_y": round(line_y_mid, 1),
            "crossings": crossing_frames,
        }
        track_stats.append(stat)

        if is_foraging_track:
            if crossed:
                crossed_tracks.append(track_id)
            else:
                not_crossed_tracks.append(track_id)

    # ── Результати ─────────────────────────────────────────────────────────
    n_crossed     = len(crossed_tracks)
    n_not_crossed = len(not_crossed_tracks)
    pct_crossed   = n_crossed / n_foraging_tracks * 100 if n_foraging_tracks > 0 else 0.0

    print(f"\n  {'─'*55}")
    print(f"  РЕЗУЛЬТАТ: GT foraging-треки що ПЕРЕТИНАЮТЬ лічильну лінію")
    print(f"  {'─'*55}")
    print(f"    Перетинають лінію:    {n_crossed:>5} / {n_foraging_tracks}  ({pct_crossed:.1f}%)")
    print(f"    НЕ перетинають:       {n_not_crossed:>5} / {n_foraging_tracks}  ({100-pct_crossed:.1f}%)")

    if pct_crossed < 50:
        verdict = "LIMITATION"
        color   = "\033[33m"  # yellow
        detail  = "Більшість GT-foraging бджіл НЕ перетинають лічильну лінію.\n" \
                  "    → Це структурна проблема датасету/методу: foraging класифікується\n" \
                  "      за поведінкою (is_arrival=1), але система рахує crossing events.\n" \
                  "    → У статті: описати як limitation у Methods і Results."
    elif pct_crossed > 50:
        verdict = "POTENTIAL BUG"
        color   = "\033[31m"  # red
        detail  = "Більшість GT-foraging бджіл перетинають лінію, але система їх не рахує.\n" \
                  "    → Потенційний bug у detection pipeline або matching window."
    else:
        verdict = "BORDERLINE (50%)"
        color   = "\033[36m"
        detail  = "Рівно половина перетинає. Потрібен додатковий аналіз."

    print(f"\n  \033[1m{color}ВЕРДИКТ: {verdict}\033[0m")
    print(f"    {detail}")

    # ── Аналіз чому не перетинають ────────────────────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  Аналіз НЕ-перетинаючих GT-foraging треків:")
    print(f"  {'─'*55}")

    above_line  = 0  # трек повністю ВИЩЕ лінії (невеликий рух у зоні льотка)
    below_line  = 0  # трек повністю НИЖЧЕ лінії (вже усередині)
    straddle    = 0  # Y-діапазон охоплює лінію але без перетину (jitter)
    short_track = 0  # < 5 кадрів

    not_crossed_stats = [s for s in track_stats if s["is_foraging"] and not s["crossed"]]

    for s in not_crossed_stats:
        if s["n_frames"] < 5:
            short_track += 1
            continue
        line_y = s["line_y"]
        if s["y_max"] < line_y:
            above_line += 1
        elif s["y_min"] > line_y:
            below_line += 1
        else:
            straddle += 1

    print(f"    Коротких треків (< 5 кадрів):    {short_track}")
    print(f"    Повністю ВИЩЕ лінії:              {above_line}")
    print(f"    Повністю НИЖЧЕ лінії:             {below_line}")
    print(f"    Охоплюють лінію але не перетин.:  {straddle}  ← jitter / stationary on line")

    # ── Детальний вивід (verbose) ──────────────────────────────────────────
    if verbose:
        print(f"\n  {'─'*55}")
        print("  [VERBOSE] Не-перетинаючі foraging-треки (перші 20):")
        for s in not_crossed_stats[:20]:
            print(f"    track={s['track_id']:>6}  frames={s['n_frames']:>4}  "
                  f"y=[{s['y_min']:.0f}..{s['y_max']:.0f}]  line_y={s['line_y']:.0f}")
        if len(not_crossed_stats) > 20:
            print(f"    ... ще {len(not_crossed_stats) - 20} треків")

    # ── Кількість GT crossing-подій (для довідки) ──────────────────────────
    total_gt_events = sum(
        len(s["crossings"]) for s in track_stats if s["is_foraging"] and s["crossed"]
    )
    all_track_events = sum(len(s["crossings"]) for s in track_stats)
    print(f"\n  {'─'*55}")
    print(f"  GT crossing-подій від foraging-треків: {total_gt_events}")
    print(f"  GT crossing-подій від ВСІХ треків:     {all_track_events}")
    print(f"  (система рахує ВСІ перетини, незалежно від мітки поведінки)")
    print(f"{'='*65}\n")

    return {
        "pair": pair,
        "n_foraging_tracks": n_foraging_tracks,
        "n_crossed": n_crossed,
        "n_not_crossed": n_not_crossed,
        "pct_crossed": round(pct_crossed, 1),
        "verdict": verdict,
        "above_line": above_line,
        "below_line": below_line,
        "straddle": straddle,
        "short_track": short_track,
        "gt_foraging_crossing_events": total_gt_events,
        "gt_all_crossing_events": all_track_events,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Діагностика: скільки GT-foraging треків перетинають лічильну лінію?"
    )
    parser.add_argument("--pair", nargs="+",
                        default=["20230711b-fan"],
                        help="Basename датасету (можна декілька через пробіл)")
    parser.add_argument("--verbose", action="store_true",
                        help="Вивести список непересічних треків")
    args = parser.parse_args()

    results = []
    for pair in args.pair:
        r = diagnose(pair, verbose=args.verbose)
        results.append(r)

    if len(results) > 1:
        print("\n  ЗВЕДЕНА ТАБЛИЦЯ:")
        print(f"  {'Відео':<20} {'ForagingTracks':>14} {'Crossed':>8} {'%Crossed':>9} {'Verdict'}")
        print("  " + "-" * 70)
        for r in results:
            print(f"  {r['pair']:<20} {r['n_foraging_tracks']:>14} "
                  f"{r['n_crossed']:>8} {r['pct_crossed']:>8.1f}%  {r['verdict']}")


if __name__ == "__main__":
    main()
