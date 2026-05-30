"""Витягує ключові метрики з summary-JSON eval_fast (для свіпів)."""
import json
import sys


def grab(path):
    with open(path) as f:
        d = json.load(f)
    # eval_fast: {"results": {pair: doc}}; eval_cli: сам doc (плоский)
    if "results" in d:
        res = d["results"]
    else:
        pair = path.split("/")[-1].replace(".json", "")
        res = {pair: d}
    out = {}
    for pair, doc in res.items():
        pc = doc.get("per_class", {})
        fan = pc.get("fanning", {})
        deff = doc.get("defense_events") or {}
        forr = doc.get("foraging_events") or {}
        cm = doc.get("confusion_matrix", {})
        leak = cm.get("fanning", {}).get("defense", 0)
        out[pair] = {
            "fanF1": fan.get("f1", 0), "fanP": fan.get("precision", 0),
            "fanR": fan.get("recall", 0), "fan_gt": fan.get("gt_count", 0),
            "fan2def": leak,
            "defF1": deff.get("f1", 0), "defR": deff.get("recall", 0),
            "defP": deff.get("precision", 0),
            "forF1": forr.get("f1", 0),
        }
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            g = grab(p)
            for pair, m in g.items():
                print(f"{p.split('/')[-1]:<28} {pair:<16} "
                      f"fanF1={m['fanF1']*100:5.1f} fanP={m['fanP']*100:5.1f} "
                      f"fanR={m['fanR']*100:5.1f} fan→def={m['fan2def']:>6} | "
                      f"defF1={m['defF1']*100:5.1f} defR={m['defR']*100:5.1f} "
                      f"forF1={m['forF1']*100:5.1f}")
        except FileNotFoundError:
            print(f"{p}: відсутній")
