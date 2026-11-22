# `eval/`: offline evaluation and diagnostics

Offline tools for measuring and tuning the pipeline against ground truth. They are not part of the API. Run them from `backend/` as modules so that `config`, `services` and `eval` resolve:

```bash
cd backend
uv run python -m eval.eval_fast                         # fast behavior eval on all 3 videos
uv run python -m eval.eval_cli --pair 20230609b-def     # same, with the real uncached pipeline
uv run python -m eval.ab_counting_eval                  # counting: Approach A vs B
```

## Behavior classification

| Module | Purpose |
|---|---|
| `eval_fast.py` | Behavior evaluation from cached detections (about 150 fps instead of 15). Options: `--pairs`, `--set key=value`, `--json`, `--rebuild`. Metrics are identical to the full pipeline. |
| `eval_cli.py` | The same evaluation through the real pipeline (uncached YOLO plus annotation), to confirm cached results hold. |
| `run_validate.sh` | `eval_cli` on all 3 videos. |
| `detection_cache.py` | Caches raw YOLO detections in `backend/data/eval_cache/`. Detection is the expensive part and does not depend on behavior, defense or stitching parameters, so it is cached once. |
| `diag_features.py` | Feature distributions (zero-crossing rate, motion, displacement, speed) for TP, FP and FN groups, to see which threshold actually separates classes. |
| `diag_frame_presence.py` | Re-scores predictions with the per-frame presence metric of Sledevič et al. |
| `balanced_fp_eval.py` | Their balanced protocol: 2000 frames per class. |
| `track_level_fanning.py` | Track-level fanning recall: was fanning detected at any point during a ground-truth fanning track. |
| `diag_foraging_gt_crossings.py` | How many ground-truth foraging tracks physically cross the counting line, which caps achievable foraging recall. |
| `parse_eval.py` | Extracts key metrics from summary JSON files for parameter sweeps. |
| `baseline_logs/` | Historical baseline logs. |

## Counting: Approach A vs B

| Module | Purpose |
|---|---|
| `ab_counting_eval.py` | Paired comparison of A (trajectory) and B (pose-validated). The dataset has no crossing labels, so ground-truth events are derived from the annotated tracks: stable crossings of the entrance line with debouncing. |
| `ab_threshold_sweep.py` | Sweeps B's rejection angle offline and compares against keypoints shuffled between bees. |
| `ab_permutation_test.py` | Permutation test: does the pose filter beat random rejection of the same number of events? |

## Defense episodes

| Module | Purpose |
|---|---|
| `run_defense_episode_detail.py` | Runs the evaluation on `20230609b-def` and prints frame IDs of every ground-truth and predicted defense episode. |
| `extract_defense_episodes.py`, `extract_defense_episodes_full.py`, `eval_defense_episodes.py` | Read an `eval_cli` JSON result and print the episodes behind the event-level defense metric, for manual checking. |

## Two behavior metrics

- **Strict, per bee (IoU ≥ 0.3).** Every bee must be matched and classified correctly. This is the main metric here.
- **Per-frame presence (Sledevič et al., 2025).** Frame-level and multi-label, with a background class. More lenient and comparable to the numbers in their paper. Computed automatically in `build_behavior_evaluation` as the `frame_presence` field.

## Tuned configuration

`backend/config/eval_config.yaml` is read by `eval_fast` and `eval_cli` and merged over the code defaults. The same values are the defaults of `schemas.ProcessConfig`, so the UI uses them too. The key finding from tuning: the zero-crossing rate is saturated (about 38 Hz for every bee) and useless, while staying in place (`behavior_fanning_max_disp`) is what separates fanning bees.

## Cache

The cache is built on demand; force a rebuild with `--rebuild`. It is invalidated when confidence, IoU, image size or the model change.
