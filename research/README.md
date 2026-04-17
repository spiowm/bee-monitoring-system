# Research & MLOps Environment

ML-середовище для навчання YOLO-моделей системи BuzzTrack. Підтримує три режими запуску, автоматичне завантаження даних з Kaggle та трекінг експериментів у DagsHub/MLflow.

## Структура

```
research/
├── src/
│   ├── run_experiment.py   # Головна точка входу
│   └── data_manager.py     # Завантаження Kaggle + підготовка split
├── config/
│   ├── config.yaml         # Глобальні гіперпараметри і режими
│   └── experiment/
│       ├── pose_baseline.yaml    # Bee pose estimation (2 keypoints)
│       └── ramp_detection.yaml   # Ramp detection (4 corner keypoints)
├── notebooks/
│   └── 01_eda.ipynb        # Розвідувальний аналіз датасету
└── pyproject.toml          # Залежності (uv)
```

## Встановлення

```bash
cd research
uv sync          # створює .venv і встановлює всі пакети
```

## Налаштування `.env`

Створіть файл `research/.env` (не потрапляє в git):

```env
# DagsHub (MLflow tracking)
DAGSHUB_USER=spiowm
DAGSHUB_REPO=bee-monitoring-system
DAGSHUB_TOKEN=ваш_токен

# Kaggle (завантаження датасету)
KAGGLE_KEY=KGAT_ваш_токен
```

> Альтернатива: покладіть `kaggle.json` у `~/.kaggle/kaggle.json`.

---

## Запуск

### Локальний тест (1 epoch, швидко)

```bash
uv run src/run_experiment.py                            # pose, 1 epoch
uv run src/run_experiment.py experiment=ramp_detection  # ramp, 1 epoch
```

### Повне навчання

```bash
uv run src/run_experiment.py training.epochs=50
uv run src/run_experiment.py experiment=ramp_detection training.epochs=100
```

### Перевизначення будь-якого параметра (Hydra CLI)

```bash
uv run src/run_experiment.py training.imgsz=1280 training.batch=8 project.note="test lr"
```

### Grid sweep — кілька варіантів одночасно (Hydra multirun)

```bash
# Запускає 4 комбінації: 2 lr × 2 imgsz
uv run src/run_experiment.py --multirun \
  training.lr0=0.001,0.0005 \
  training.imgsz=1280,1920
```

Кожна комбінація — окремий MLflow run. Результати у `multirun/`.

### HPO — автоматичний підбір гіперпараметрів (Optuna)

```bash
uv run src/run_experiment.py mode=hpo hpo.n_trials=20
uv run src/run_experiment.py experiment=ramp_detection mode=hpo hpo.n_trials=10
```

Простір пошуку задається в `config/config.yaml` у секції `hpo.search_space`. Кожен trial — вкладений MLflow run. Найкращі параметри логуються в батьківський run.

---

## Датасет

Два окремих Kaggle-датасети (~50-200 МБ кожен):

| Датасет | Kaggle ID | Зображень | Анотації |
|---------|-----------|-----------|---------|
| Pose estimation | `spiowm/bee-monitoring-pose` | 400, 8 вуликів | bbox + 2 keypoints |
| Ramp detection | `spiowm/bee-monitoring-ramp-detection` | 156, 15 вуликів | 4 corner keypoints |

Завантаження відбувається автоматично через `kagglehub` при першому запуску експерименту. Потрібен `KAGGLE_KEY` у `.env`.

> Датасети мають бути **опубліковані** на Kaggle (навіть як private). Draft-статус повертає 403.

Якщо дані вже є локально — пропустіть завантаження:
```bash
uv run src/run_experiment.py data.source_type=local ...
# дані мають бути в datasets/raw/pose/ або datasets/raw/ramp/
```

Після завантаження `data_manager` автоматично:
1. Знаходить `images/` і `labels/` у завантаженому датасеті
2. Розбиває на train/val
3. Генерує `data.yaml` для YOLO

Локальні дані зберігаються в `datasets/` (gitignored).

---

## Hive-based split

Для pose estimation використовується **розподіл по вуликах** замість випадкового:

| Split | Вулики | Зображень |
|-------|--------|-----------|
| train | 6 вуликів | 300 |
| val   | `20230711b`, `20230609e` | 100 |

**Чому це важливо:** при випадковому split модель бачить кадри тих самих вуликів в train і val → метрики завищені. Hive-based split оцінює реальну здатність до узагальнення на **нові** вулики (cross-hive generalization).

Для `ramp_detection` задано `val_hives: []` → використовується random split (val_ratio=0.2).

Щоб змінити val hives через CLI:
```bash
uv run src/run_experiment.py "data.val_hives=[20230711b,20230609c]"
```

---

## Конфігурація

### Структура конфігів (Hydra)

```
config/config.yaml          ← глобальні дефолти (epochs=1, batch=4, ...)
config/experiment/*.yaml    ← перевизначають поля для конкретного завдання
```

Поточні експерименти:
- `pose_baseline` — bee pose, yolo11s-pose.pt, hive-based split
- `ramp_detection` — ramp 4kpt, yolo11s-pose.pt, random split

### Ключові параметри `config.yaml`

| Параметр | Дефолт | Опис |
|----------|--------|------|
| `mode` | `single` | `single` \| `hpo` |
| `training.epochs` | `1` | Epoch count (1 = локальний тест) |
| `training.imgsz` | `1920` | Input resolution |
| `training.batch` | `4` | Batch size |
| `training.lr0` | `0.001` | Initial learning rate |
| `hpo.n_trials` | `10` | Кількість Optuna trials |
| `hpo.search_space` | — | Простір пошуку для HPO |

### Додати новий експеримент

Створіть `config/experiment/my_experiment.yaml`:
```yaml
# @package _global_
project:
  experiment_name: "My_Experiment"
  note: ""

data:
  kaggle_id: "spiowm/monitoring-bees-at-the-hive-entrance"
  kaggle_subfolder: "pose"
  raw_dir: "datasets/raw"
  prepared_dir: "datasets/my_exp"
  dataset_path: "datasets/my_exp/data.yaml"
  nc: 1
  names: ["bee"]
  kpt_shape: [2, 2]
  val_hives: ["20230711b", "20230609e"]

model:
  name: "yolo11m-pose.pt"
  task: "pose"
```

Запуск:
```bash
uv run src/run_experiment.py experiment=my_experiment training.epochs=50
```

---

## DagsHub / MLflow — що логується

Кожен run зберігає:

| Артефакт | Де |
|----------|-----|
| `hydra_config.yaml` | Повна конфігурація запуску |
| `yolo_run/weights/best.pt` | Найкраща модель |
| `yolo_run/weights/last.pt` | Остання модель |
| `yolo_run/results.csv` | Метрики по epochs |
| `yolo_run/*.png` | Графіки (loss curves, confusion matrix, PR curve) |
| Параметри + метрики | MLflow params/metrics |

Для HPO додатково: вкладені runs для кожного trial + `best_*` параметри в батьківському run.

---

## Google Colab

```python
# 1. Встановлення uv і клонування
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] += ":/root/.cargo/bin"
!git clone https://github.com/spiowm/bee-monitoring-system.git
os.chdir("bee-monitoring-system/research")
!uv sync

# 2. Ключі доступу (або через Colab Secrets)
os.environ["DAGSHUB_USER"]  = "spiowm"
os.environ["DAGSHUB_REPO"]  = "bee-monitoring-system"
os.environ["DAGSHUB_TOKEN"] = "ваш_токен"
os.environ["KAGGLE_KEY"]    = "ваш_kaggle_key"   # KGAT_... токен

# 3. Повне навчання (датасет ~200 МБ завантажується автоматично)
!uv run src/run_experiment.py training.epochs=50 training.batch=8

# Або ramp detection (~50 МБ)
!uv run src/run_experiment.py experiment=ramp_detection training.epochs=100

# Або HPO
!uv run src/run_experiment.py mode=hpo hpo.n_trials=20 training.epochs=30
```

> Всі артефакти (best.pt, метрики, графіки) автоматично зберігаються в DagsHub.
