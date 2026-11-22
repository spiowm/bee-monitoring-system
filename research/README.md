# Research: model training and benchmarks

Training of the two YOLO11-pose models used by the backend: the bee model (box plus head and stinger keypoints) and the ramp model (4 corner keypoints of the landing board). Experiments are configured with Hydra and logged to MLflow on DagsHub.

## Setup

```bash
cd research
uv sync
```

`research/.env`:

```env
DAGSHUB_USER=<user>
DAGSHUB_REPO=<repo>
DAGSHUB_TOKEN=...
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```

## Pipeline

```bash
# 1. Download the datasets from Kaggle
uv run src/download.py pose    # -> datasets/raw/pose/  (~200 MB)
uv run src/download.py ramp    # -> datasets/raw/ramp/  (~51 MB)

# 2. Split into train/val and write data.yaml
uv run src/prepare.py experiment=bee_pose
uv run src/prepare.py experiment=ramp_detection

# 3. Train and log to MLflow
uv run src/run_experiment.py experiment=bee_pose
uv run src/run_experiment.py experiment=ramp_detection
```

The Kaggle copies of the datasets are private. To reproduce without access, take the original data published with Sledevič et al., *Agriculture* 2024 (see the root README) and place it in `datasets/raw/pose/` and `datasets/raw/ramp/` as `images/` and `labels/` in YOLO pose format.

The experiment configs default to a quick smoke run (nano model, 320 px, 1 epoch). The final bee model from the thesis was trained with:

```bash
uv run src/run_experiment.py experiment=bee_pose model.name=yolo11s-pose.pt training.imgsz=1920 training.epochs=50
```

## Experiments

**bee_pose:** 2 keypoints per bee (head, stinger), about 400 images from 8 hives. The split is by hive: hives `20230711b` and `20230609e` go entirely to validation, so the model is always tested on hives it has never seen. A random split leaks near-identical frames of the same hive into both sets and inflates the metrics.

**ramp_detection:** 4 corner keypoints of the landing board, about 156 images from 15 hives, random 80/20 split with seed 42. Horizontal flip is disabled because it would swap the order of the corner keypoints.

## Configuration (Hydra)

```
config/config.yaml              # infrastructure defaults: seed, device, workers, HSV augmentation
config/experiment/
    bee_pose.yaml               # full experiment description, overrides config.yaml
    ramp_detection.yaml
```

Any value can be overridden from the command line:

```bash
uv run src/run_experiment.py experiment=bee_pose training.lr0=0.0005 training.batch=8
```

## What is logged to MLflow

- **Parameters:** model, epochs, batch, image size, learning rate, optimizer, augmentations, split strategy.
- **Metrics:** `best_*` and `last_*` (mAP, losses), `speed_fps`, `model_size_mb`. For bee_pose also `pose_nme_mean`, `angular_error_mean_deg` and the share of bees within 15° and 30°.
- **Artifacts:** `best.pt`, `last.pt`, `results.csv`, plots, the resolved Hydra config.

Orientation error (the angle between the predicted and annotated head-to-stinger vectors) turned out to be a more sensitive measure than mAP: it exposed a hive the model does not generalize to, while mAP barely moved.

## Other scripts

| Script | Purpose |
|---|---|
| `src/angular_error_eval.py` | orientation error of a trained model on the validation labels |
| `src/benchmark_fps.py` | inference speed of a model on one image |
| `src/benchmark_speed_comparison.py` | one-stage pose model versus a detect-then-crop two-stage design, as the number of bees grows |
| `notebooks/eda_pose.ipynb`, `notebooks/eda_ramp.ipynb` | dataset exploration |
| `notebooks/count_bees_per_hive.py` | bee density per hive |
| `notebooks/analyze_tracking.py` | analysis of tracking output |

## Notes

- Kaggle datasets must be published (private is fine); a draft returns 403.
- For hive-based splitting, the first 9 characters of an image file name are the hive ID.

## Ground-truth dataset for the backend

`datasets/raw/tracking_and_behavior/` (about 625 MB, git-ignored) is not used for training. The backend's evaluation reads it to compare predictions with annotated tracks and behaviors. Source: Sledevič & Matuzevičius, [Mendeley Data, DOI 10.17632/8gb9r2yhfc.6](https://data.mendeley.com/datasets/8gb9r2yhfc/6). Each video has its own folder:

```
tracking_and_behavior/<video>/
├── video.mp4                  # 1920×1080, 50 FPS
├── tracks_and_behavior.txt    # frame, track_id, cx, cy, w, h (normalized), 4 behavior flags
└── entrance_zone.txt          # 4-point entrance polygon
```

Ground-truth videos used for evaluation: `20230711a-fan`, `20230711b-fan`, `20230609b-def` (defense) and `yt8`. The parser is in [backend/services/evaluation/gt_loader.py](../backend/services/evaluation/gt_loader.py).
