# Research & MLOps Environment

ML-середовище для навчання YOLO-моделей системи BuzzTrack. Автоматичне завантаження даних з Kaggle, підготовка датасету та трекінг експериментів у DagsHub/MLflow.

## Структура

```
research/
├── src/
│   ├── run_experiment.py        # Навчання + логування в MLflow
│   ├── prepare.py               # Split на train/val + генерація data.yaml
│   └── download.py              # Завантаження датасетів з Kaggle
├── config/
│   ├── config.yaml              # Глобальні гіперпараметри (дефолти)
│   └── experiment/
│       ├── bee_pose.yaml        # Bee pose estimation (2 keypoints, hive-based split)
│       └── ramp_detection.yaml  # Ramp detection (4 corner keypoints, random split)
├── notebooks/                   # EDA-ноутбуки
└── pyproject.toml               # Залежності (uv)
```

---

## Встановлення

```bash
cd research
uv sync
```

Потрібен Python ≥ 3.12, [uv](https://docs.astral.sh/uv/).

---

## Налаштування `.env`

Створіть файл `research/.env`:

```env
DAGSHUB_USER=spiowm
DAGSHUB_REPO=bee-monitoring-system
DAGSHUB_TOKEN=ваш_dagshub_токен

KAGGLE_KEY=KGAT_ваш_токен    # kaggle.com → Settings → API
```

> Альтернатива для Kaggle: покладіть `kaggle.json` у `~/.kaggle/kaggle.json`.

---

## Датасети

| Ключ   | Kaggle ID                               | Розмір   | Зображень         | Анотації                          |
| ------ | --------------------------------------- | -------- | ----------------- | --------------------------------- |
| `pose` | `spiowm/bee-monitoring-pose`            | ~200 MB  | 400, 8 вуликів    | bbox + 2 keypoints (голова, жало) |
| `ramp` | `spiowm/bee-monitoring-ramp-detection`  | ~51 MB   | 156, 15 вуликів   | bbox + 4 corner keypoints         |

> Датасети мають бути **опубліковані** на Kaggle (навіть як private). Draft-статус повертає 403.

**Про імена файлів:** перші 9 символів — ID вулика (`20230609a`), решта — Roboflow-хеш. На цьому базується hive-based split.

---

## Pipeline

Три незалежних кроки:

### Крок 1 — завантаження

```bash
uv run src/download.py         # обидва датасети
uv run src/download.py pose    # тільки pose  (~200 MB) → datasets/raw/pose/
uv run src/download.py ramp    # тільки ramp  (~51 MB)  → datasets/raw/ramp/
```

Повторний виклик пропускається якщо дані вже є.

### Крок 2 — підготовка (split + data.yaml)

```bash
uv run src/prepare.py experiment=bee_pose        # hive-based split
uv run src/prepare.py experiment=ramp_detection  # random split 80/20
```

Перезапуск завжди перезаписує `data.yaml` і папки split.

```bash
# Змінити val-вулики для pose
uv run src/prepare.py experiment=bee_pose "data.val_hives=[20230711b,20230609c]"

# Змінити частку val для ramp
uv run src/prepare.py experiment=ramp_detection data.val_ratio=0.15
```

### Крок 3 — навчання

```bash
# Production запуск (параметри з experiment/*.yaml)
uv run src/run_experiment.py experiment=bee_pose
uv run src/run_experiment.py experiment=ramp_detection

# Локальний тест (override epochs/imgsz для швидкої перевірки)
uv run src/run_experiment.py experiment=bee_pose training.epochs=1 training.imgsz=320
```

Якщо `data.yaml` відсутній — падає з підказкою запустити `prepare.py`.

```bash
# Перевизначення гіперпараметрів
uv run src/run_experiment.py experiment=bee_pose training.lr0=0.0005 training.batch=8
```

---

## Два варіанти split

### Hive-based split (`bee_pose`)

Розподіл **по вуликах**, а не по зображеннях:

| Split | Вулики                    | Зображень |
| ----- | ------------------------- | --------- |
| train | 6 вуликів                 | ~300      |
| val   | `20230711b`, `20230609e`  | ~100      |

**Чому важливо:** при random split модель бачить кадри тих самих вуликів у train і val → метрики завищені. Hive-based split оцінює справжню здатність до узагальнення на **нові** вулики.

```yaml
# bee_pose.yaml
data:
  split_strategy: "hive"
  val_hives: ["20230711b", "20230609e"]
```

### Random split (`ramp_detection`)

Стандартний split 80/20 з фіксованим seed=42.

```yaml
# ramp_detection.yaml
data:
  split_strategy: "random"
  val_ratio: 0.2
```

---

## Конфігурація (Hydra)

```text
config/config.yaml          ← інфраструктурні дефолти (seed, device, workers)
config/experiment/*.yaml    ← повний опис експерименту (дані + навчання + модель)
```

Кожен experiment-конфіг — **самодостатній**: містить усі гіперпараметри для production-запуску. `config.yaml` задає лише інфраструктурні дефолти, які рідко змінюються. Будь-який параметр можна перевизначити через CLI.

### Параметри experiment/*.yaml

| Параметр              | bee_pose | ramp_detection | Опис                             |
| --------------------- | -------- | -------------- | -------------------------------- |
| `training.epochs`     | `50`     | `100`          | Кількість епох                   |
| `training.imgsz`      | `1920`   | `640`          | Розмір входу (кратне 32)         |
| `training.batch`      | `4`      | `8`            | Розмір батчу                     |
| `training.lr0`        | `0.01`   | `0.01`         | Початковий learning rate         |
| `training.optimizer`  | `AdamW`  | `AdamW`        | Оптимізатор                      |
| `training.patience`   | `20`     | `20`           | Early stopping patience          |
| `data.split_strategy` | `hive`   | `random`       | `random` \| `hive`               |

### Аугментації

| Параметр    | bee_pose | ramp_detection | Опис                              |
| ----------- | -------- | -------------- | --------------------------------- |
| `mosaic`    | 0.48     | 1.0            | Ймовірність mosaic                |
| `degrees`   | 7.6      | 0.0            | Ротація (градуси)                 |
| `fliplr`    | 0.36     | 0.0 ¹          | Горизонтальний flip               |
| `translate` | 0.07     | 0.1            | Зсув                              |
| `scale`     | 0.42     | 0.5            | Масштаб                           |

¹ `fliplr=0.0` для ramp — порядок кутових keypoints потребує верифікації перед увімкненням flip.

### Інфраструктурні дефолти (config.yaml)

| Параметр              | Значення | Опис                              |
| --------------------- | -------- | --------------------------------- |
| `training.seed`       | `42`     | Random seed                       |
| `training.device`     | `""`     | Auto (GPU якщо є, інакше CPU)     |
| `training.workers`    | `4`      | Dataloader workers                |
| `training.project`    | `runs`   | Директорія для збереження         |

### Додати новий експеримент

Створіть `config/experiment/my_experiment.yaml`:

```yaml
# @package _global_
project:
  experiment_name: "My_Experiment"
  note: ""

data:
  kaggle_id: "spiowm/bee-monitoring-pose"
  raw_dir: "datasets/raw/pose"
  prepared_dir: "datasets/my_exp"
  dataset_path: "datasets/my_exp/data.yaml"
  nc: 1
  names: ["bee"]
  kpt_shape: [2, 2]
  split_strategy: "hive"
  val_hives: ["20230711b", "20230609e"]

training:
  epochs: 50
  batch: 4
  imgsz: 1920
  optimizer: "AdamW"
  lr0: 0.01
  patience: 20
  mosaic: 0.48
  degrees: 7.6
  fliplr: 0.36
  translate: 0.07
  scale: 0.42

model:
  name: "yolo11m-pose.pt"
```

---

## DagsHub / MLflow — що логується

**Параметри:**

- **Модель:** `model`
- **Навчання:** `epochs`, `batch`, `imgsz`, `lr0`, `optimizer`, `patience`, `seed`, `mosaic`, `degrees`, `fliplr`, `translate`, `scale`
- **Дані:** `nc`, `classes`, `split_strategy`, `val_ratio`/`val_hives`, `train_images`, `val_images`, `total_images`
- **Швидкість:** `speed_device`

**Метрики:**

- `last_*` — метрики останньої епохи (train losses + val)
- `best_*` — метрики `best.pt` після завершення навчання
- **Швидкість inference:**
  - `speed_preprocess_ms`, `speed_inference_ms`, `speed_postprocess_ms` — час кожного етапу
  - `speed_total_ms`, `speed_fps` — загальний час та FPS
- **Pose-метрики** (лише для bee_pose, kpt_shape=[2,2]):
  - `pose_nme_mean`, `pose_nme_median` — Normalized Mean Error (менше = краще)
  - `angular_error_mean_deg`, `angular_error_median_deg` — помилка кута голова→жало
  - `angular_accuracy_within_15deg`, `angular_accuracy_within_30deg` — % передбачень в межах порогу
  - `pose_n_matched` — кількість зіставлених пар GT↔prediction
- `model_size_mb` — розмір best.pt

**Артефакти:**

| Файл | Опис |
|------|------|
| `hydra_config.yaml` | Повна конфігурація запуску |
| `yolo_run/weights/best.pt` | Найкраща модель |
| `yolo_run/weights/last.pt` | Остання модель |
| `yolo_run/results.csv` | Метрики по епохах |
| `yolo_run/*.png` | Loss curves, confusion matrix, PR curve |

---

## Google Colab

```python
# ── Клітинка 1: Встановлення ──────────────────────────────────────────────
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] += ":/root/.cargo/bin"

!git clone https://github.com/spiowm/bee-monitoring-system.git
os.chdir("bee-monitoring-system/research")
!uv sync

# ── Клітинка 2: Ключі доступу (або Colab Secrets) ─────────────────────────
from google.colab import userdata
os.environ["DAGSHUB_USER"]  = "spiowm"
os.environ["DAGSHUB_REPO"]  = "bee-monitoring-system"
os.environ["DAGSHUB_TOKEN"] = userdata.get("DAGSHUB_TOKEN")
os.environ["KAGGLE_KEY"]    = userdata.get("KAGGLE_KEY")

# ── Клітинка 3: Pipeline ──────────────────────────────────────────────────
!uv run src/download.py pose
!uv run src/prepare.py experiment=bee_pose
!uv run src/run_experiment.py experiment=bee_pose  # production params з experiment config

# Або ramp detection
# !uv run src/download.py ramp
# !uv run src/prepare.py experiment=ramp_detection
# !uv run src/run_experiment.py experiment=ramp_detection training.epochs=100
```
