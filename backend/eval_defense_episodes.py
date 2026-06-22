"""Скрипт для читання JSON-результату eval_cli і виведення детальних
episode frame-ids для defense event-level метрики.

Запуск після eval_cli:
    python eval_defense_episodes.py /tmp/eval_def_result.json
"""
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/eval_def_result.json")
doc = json.loads(result_path.read_text())

de = doc.get("defense_events")
if not de:
    print("defense_events відсутній або порожній у JSON")
    sys.exit(1)

print("=" * 60)
print(f"  Defense (подієва метрика) — детальний episode-звіт")
print("=" * 60)
print(f"  GT-подій:   {de['gt_count']}")
print(f"  Pred-подій: {de['pred_count']}")
print(f"  TP={de['tp']}  FP={de['fp']}  FN={de['fn']}")
print(f"  Precision: {de['precision']*100:.1f}%  Recall: {de['recall']*100:.1f}%  F1: {de['f1']*100:.1f}%")

if "gt_episodes" in de:
    print("\n  GT episodes:")
    for i, ep in enumerate(de["gt_episodes"]):
        print(f"    GT[{i}]: frame {ep[0]} → {ep[1]}  (len={ep[1]-ep[0]+1})")
else:
    print("\n  [GT episodes не збережені в JSON — запусти eval з патчем episodes]")

if "pred_episodes" in de:
    print("\n  Pred episodes:")
    for i, ep in enumerate(de["pred_episodes"]):
        status = ep.get("status", "?")
        print(f"    Pred[{i}]: frame {ep['start']} → {ep['end']}  status={status}")
else:
    print("  [Pred episodes не збережені в JSON — запусти eval з патчем episodes]")
