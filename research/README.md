# Research — YOLO Training + MLflow

Навчання YOLO-моделей для pose estimation бджіл та детекції рампи. Hydra-конфіг, трекінг в DagsHub/MLflow.

## Запуск

```bash
cd research
uv sync
```

Файл `research/.env`:

```env
DAGSHUB_USER=spiowm
DAGSHUB_REPO=bee-monitoring-system
DAGSHUB_TOKEN=...
KAGGLE_KEY=...
```

## Pipeline (три кроки)

```bash
# 1. Завантажити датасет з Kaggle
uv run src/download.py pose    # → datasets/raw/pose/  (~200 MB)
uv run src/download.py ramp    # → datasets/raw/ramp/  (~51 MB)

# 2. Розбити на train/val + згенерувати data.yaml
uv run src/prepare.py experiment=bee_pose
uv run src/prepare.py experiment=ramp_detection

# 3. Тренування + логування в MLflow
uv run src/run_experiment.py experiment=bee_pose
uv run src/run_experiment.py experiment=ramp_detection
```

Швидкий локальний тест (без Kaggle, маленькі зображення):

```bash
uv run src/run_experiment.py experiment=bee_pose training.epochs=1 training.imgsz=320 training.batch=2 training.workers=0
```

## Два експерименти

**bee_pose** — pose estimation бджіл:
- Модель: `yolo11s-pose.pt`, 2 keypoints (голова + жало)
- Датасет: ~400 зображень, 8 вуликів
- Split: по вуликах — `val_hives: ["20230711b", "20230609e"]` повністю в val

**ramp_detection** — детекція рампи:
- Модель: `yolo11m-pose.pt`, 4 кутових keypoints
- Датасет: ~156 зображень, 15 вуликів
- Split: random 80/20, seed=42

## Конфіг (Hydra)

```
config/config.yaml           # інфра-дефолти: seed, device, workers, HSV-аугментації
config/experiment/
    bee_pose.yaml            # повний опис експерименту (перевизначає config.yaml)
    ramp_detection.yaml
```

Будь-який параметр можна override через CLI:

```bash
uv run src/run_experiment.py experiment=bee_pose training.lr0=0.0005 training.batch=8
```

## Що логується в MLflow

**Параметри:** модель, epochs, batch, imgsz, lr0, optimizer, аугментації, split strategy.

**Метрики:** `best_*` / `last_*` (mAP, losses), `speed_fps`, `model_size_mb`. Для bee_pose додатково: `pose_nme_mean`, `angular_error_mean_deg`, `angular_accuracy_within_15/30deg`.

**Артефакти:** `best.pt`, `last.pt`, `results.csv`, plots, `hydra_config.yaml`.

## Нюанси

- Датасети Kaggle мають бути **опубліковані** (навіть як private). Draft → 403.
- Hive-based split потрібен для bee_pose: перші 9 символів назви файлу = ID вулика.
- `fliplr=0.0` для ramp_detection — порядок кутових keypoints не зберігається при flip.
