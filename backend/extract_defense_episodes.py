"""Читає JSON з eval_cli, самостійно відтворює логіку _compute_event_metrics_for_class
і виводить детальні frame-ids кожного GT/Pred episode для ручної перевірки.

Запуск: python extract_defense_episodes.py /tmp/eval_def_result.json
"""
import json
import sys
from pathlib import Path


def get_episodes(frames, max_gap):
    eps = []
    if not frames:
        return eps
    frames = sorted(frames)
    start = frames[0]
    prev = frames[0]
    for f in frames[1:]:
        if f - prev > max_gap:
            eps.append((start, prev))
            start = f
        prev = f
    eps.append((start, prev))
    return eps


def main():
    result_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/eval_def_result.json")
    doc = json.loads(result_path.read_text())

    GAP = 30  # gap_frames як у коді

    # Pred frames: з pred_per_frame (ключі — рядкові frame-номери)
    pred_per_frame = doc.get("pred_per_frame", {})
    pred_frames = []
    for f_str, pdata in pred_per_frame.items():
        f_num = int(f_str)
        if any(b.get("behavior") == "defense" for b in pdata.values()):
            pred_frames.append(f_num)
    pred_eps = get_episodes(pred_frames, GAP)

    # GT episodes — не зберігаються у JSON напряму,
    # але беремо з defense_events результату щоб отримати gt_count
    de = doc.get("defense_events", {})
    if not de:
        print("defense_events відсутній у JSON!")
        sys.exit(1)

    print("=" * 70)
    print("  DEFENSE EVENT-LEVEL — детальний звіт з frame-ids")
    print("=" * 70)
    print(f"  GT-подій:   {de['gt_count']}")
    print(f"  Pred-подій: {de['pred_count']}  (знайдено у pred_per_frame: {len(pred_eps)})")
    print(f"  TP={de['tp']}  FP={de['fp']}  FN={de['fn']}")
    print(f"  Precision: {de['precision']*100:.1f}%  Recall: {de['recall']*100:.1f}%  F1: {de['f1']*100:.1f}%")

    print(f"\n  Pred-episodes (gap={GAP} frames):")
    for i, (s, e) in enumerate(pred_eps):
        dur_frames = e - s + 1
        print(f"    Pred[{i:02d}]: frame {s:6d} → {e:6d}  (тривалість={dur_frames} кадрів)")

    print(f"\n  Кількість pred-frames з behavior=defense: {len(pred_frames)}")
    if pred_frames:
        print(f"  Перші 10 pred-frames: {pred_frames[:10]}")
        print(f"  Останні 5 pred-frames: {pred_frames[-5:]}")

    print("\n  NOTE: GT episodes відтворюються тільки при повторному запуску з GT файлом.")
    print("        Дивись gt_count={} та fn={} у defense_events.".format(
        de['gt_count'], de['fn']))


if __name__ == "__main__":
    main()
