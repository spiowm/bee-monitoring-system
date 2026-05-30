import subprocess
import json
import yaml
import time
import random
from pathlib import Path

videos = ["20230609b-def", "20230711a-fan", "20230711b-fan"]
config_file = "config/eval_config.yaml"

def run_eval(video):
    out = f"/tmp/{video}.json"
    cmd = ["uv", "run", "python", "eval_cli.py", "--pair", video, "--json", out]
    # To run sequentially faster, but we actually have to wait.
    subprocess.run(cmd, check=True, capture_output=True)
    with open(out, "r") as f:
        return json.load(f)

def load_config():
    if Path(config_file).exists():
        with open(config_file, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

def save_config(cfg):
    Path(config_file).parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.safe_dump(cfg, f)

def get_fanning_recall(res):
    return res.get("per_class", {}).get("fanning", {}).get("recall", 0.0)

def get_defense_events_f1(res):
    return res.get("defense_events", {}).get("f1", 0.0)

best_score = -1.0
best_cfg = {}

history = []

for i in range(10):
    print(f"\n==================================================")
    print(f" Iteration {i+1}/10")
    print(f"==================================================")
    cfg = load_config()
    
    # Mutate config
    if i == 0:
        # Baseline iteration with our new stitching defaults
        cfg["stitch_max_dist"] = 30.0
        cfg["stitch_max_frames"] = 30
        cfg["behavior_fanning_zcr_min"] = 15.0
        cfg["behavior_fanning_motion_min"] = 0.05
    else:
        cfg["stitch_max_dist"] = random.choice([20.0, 30.0, 40.0, 50.0])
        cfg["stitch_max_frames"] = random.choice([15, 30, 45, 60])
        cfg["behavior_fanning_zcr_min"] = random.choice([12.0, 14.0, 15.0, 16.0])
        cfg["behavior_fanning_motion_min"] = random.choice([0.03, 0.05, 0.08])
        
    save_config(cfg)
    print(f"Config: {cfg}")
    
    scores = {}
    total_fanning_r = 0.0
    total_defense_f1 = 0.0
    
    for v in videos:
        print(f"  Evaluating {v}...", end="", flush=True)
        t0 = time.time()
        try:
            res = run_eval(v)
            fan_r = get_fanning_recall(res)
            def_f1 = get_defense_events_f1(res)
            print(f" done in {time.time()-t0:.1f}s. Fan_R: {fan_r:.3f}, Def_F1: {def_f1:.3f}")
            scores[v] = {"fan_r": fan_r, "def_f1": def_f1}
            total_fanning_r += fan_r
            total_defense_f1 += def_f1
        except Exception as e:
            print(f" ERROR: {e}")
            
    combined_score = (total_fanning_r / 3) + (total_defense_f1 / 3)
    print(f"--> Combined Score: {combined_score:.4f}")
    
    history.append((combined_score, cfg.copy(), scores))
    
    if combined_score > best_score:
        best_score = combined_score
        best_cfg = cfg.copy()
        print(f"*** NEW BEST SCORE! ***")

print("\n\n=== OPTIMIZATION DONE ===")
print(f"Best score: {best_score:.4f}")
print(f"Best config: {best_cfg}")
save_config(best_cfg)
print("Saved best config to eval_config.yaml.")
