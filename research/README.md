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
uv run src/run_experiment.py experiment=bee_pose training.epochs=50
uv run src/run_experiment.py experiment=ramp_detection training.epochs=100
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
config/config.yaml          ← глобальні дефолти
config/experiment/*.yaml    ← перевизначення для конкретного завдання
```

Experiment-конфіг завжди перекриває дефолти з `config.yaml`. Будь-який параметр можна перевизначити через CLI без зміни файлів.

### Ключові параметри

| Параметр              | Дефолт   | Опис                              |
| --------------------- | -------- | --------------------------------- |
| `training.epochs`     | `1`      | Кількість епох (1 для тесту)      |
| `training.imgsz`      | `320`    | Розмір входу (кратне 32)          |
| `training.batch`      | `4`      | Розмір батчу                      |
| `training.lr0`        | `0.001`  | Початковий learning rate          |
| `training.optimizer`  | `AdamW`  | Оптимізатор                       |
| `training.patience`   | `20`     | Early stopping patience           |
| `data.split_strategy` | `random` | `random` \| `hive`                |
| `data.val_ratio`      | `0.2`    | Частка val при random split       |
| `data.val_hives`      | `[]`     | Hive IDs для val при hive split   |

### Аугментації (`bee_pose` — налаштовані за результатами попередніх експериментів)

| Параметр    | bee_pose | ramp (дефолт YOLO) |
| ----------- | -------- | ------------------ |
| `mosaic`    | 0.48     | 1.0                |
| `degrees`   | 7.6      | 0.0                |
| `fliplr`    | 0.36     | 0.0 ¹              |
| `translate` | 0.07     | 0.1                |
| `scale`     | 0.42     | 0.5                |

¹ `fliplr=0.0` для ramp — порядок кутових keypoints потребує верифікації перед увімкненням flip.

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

model:
  name: "yolo11m-pose.pt"
```

---

## DagsHub / MLflow — що логується

**Параметри:**

- **Модель:** `model`
- **Навчання:** `epochs`, `batch`, `imgsz`, `lr0`, `optimizer`, `patience`, `seed`, `mosaic`, `degrees`, `fliplr`, `translate`, `scale`
- **Дані:** `nc`, `classes`, `split_strategy`, `val_ratio`/`val_hives`, `train_images`, `val_images`, `total_images`

**Метрики:**

- `last_*` — метрики останньої епохи (train losses + val)
- `best_*` — метрики `best.pt` після завершення навчання

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
!uv run src/run_experiment.py experiment=bee_pose training.epochs=50 training.batch=8

# Або ramp detection
# !uv run src/download.py ramp
# !uv run src/prepare.py experiment=ramp_detection
# !uv run src/run_experiment.py experiment=ramp_detection training.epochs=100
```
