# Research & MLOps Environment

ML-середовище для навчання YOLO-моделей системи BuzzTrack. Підтримує три режими запуску, автоматичне завантаження даних з Kaggle та трекінг експериментів у DagsHub/MLflow.

## Структура

```
research/
├── src/
│   ├── run_experiment.py        # Головна точка входу
│   └── data_manager.py          # Завантаження Kaggle + підготовка split
├── config/
│   ├── config.yaml              # Глобальні гіперпараметри і режими
│   └── experiment/
│       ├── pose_baseline.yaml   # Bee pose estimation (2 keypoints, hive-based split)
│       └── ramp_detection.yaml  # Ramp detection (4 corner keypoints, random split)
├── notebooks/
│   └── 01_eda.ipynb             # Розвідувальний аналіз датасету
└── pyproject.toml               # Залежності (uv)
```

---

## Встановлення (локально)

```bash
cd research
uv sync          # створює .venv і встановлює всі пакети
```

Потрібен Python ≥ 3.12, [uv](https://docs.astral.sh/uv/).

---

## Налаштування `.env`

Створіть файл `research/.env` (не потрапляє в git):

```env
# DagsHub (MLflow tracking)
DAGSHUB_USER=spiowm
DAGSHUB_REPO=bee-monitoring-system
DAGSHUB_TOKEN=ваш_dagshub_токен

# Kaggle (завантаження датасетів)
KAGGLE_KEY=KGAT_ваш_токен    # знайти: kaggle.com → Settings → API
```

> Альтернатива для Kaggle: покладіть `kaggle.json` у `~/.kaggle/kaggle.json`.

---

## Датасети

Два окремих Kaggle-датасети (завантажуються незалежно):

| Ключ | Kaggle ID | Розмір | Зображень | Анотації |
|------|-----------|--------|-----------|---------|
| `pose` | `spiowm/bee-monitoring-pose` | ~200 MB | 400, 8 вуликів | bbox + 2 keypoints (голова, жало) |
| `ramp` | `spiowm/bee-monitoring-ramp-detection` | ~51 MB | 156, 15 вуликів | 4 corner keypoints |

> Датасети мають бути **опубліковані** на Kaggle (навіть як private). Draft-статус повертає 403.

**Про імена файлів:** файли мають суфікс `.rf.<hash>` — це Roboflow додає автоматично при експорті. Перші 9 символів імені — ID вулика (`20230609a`), тому hive-based split працює коректно.

### Явне завантаження у `datasets/raw/`

```bash
# Завантажити обидва датасети (рекомендовано перед першим запуском)
uv run src/data_manager.py

# Завантажити тільки один
uv run src/data_manager.py pose
uv run src/data_manager.py ramp
```

Дані зберігаються в `datasets/raw/pose/` і `datasets/raw/ramp/` і **залишаються** там після обробки. При повторному запуску завантаження пропускається (кеш).

### Підготовка (split + data.yaml)

Split запускається автоматично як частина `run_experiment.py`. Але можна запустити окремо — через `ensure_dataset` безпосередньо або просто запустивши `run_experiment.py` з `training.epochs=0` (не підтримується YOLO). Найпростіше — просто запустити навчання на 1 епоху.

Дані після обробки зберігаються в `datasets/pose/` або `datasets/ramp/`.

---

## Два варіанти split

### 1. Hive-based split (pose_baseline)

Розподіл **по вуликах** — не по зображеннях:

| Split | Вулики | Зображень |
|-------|--------|-----------|
| train | 6 вуликів | ~300 |
| val   | `20230711b`, `20230609e` | ~100 |

**Чому важливо:** при random split модель бачить кадри тих самих вуликів в train і val → метрики завищені. Hive-based split оцінює справжню здатність до узагальнення на **нові** вулики (cross-hive generalization).

Задається в `config/experiment/pose_baseline.yaml`:
```yaml
data:
  val_hives: ["20230711b", "20230609e"]
```

### 2. Random split (ramp_detection)

Звичайний рандомний split 80/20 з фіксованим seed=42.

Задається відсутністю `val_hives`:
```yaml
data:
  val_hives: []   # → random split, val_ratio=0.2
```

### Зміна split через CLI

```bash
# Інші val hives для pose
uv run src/run_experiment.py "data.val_hives=[20230711b,20230609c]"

# Перейти на random split для pose (ігнорувати hive-based)
uv run src/run_experiment.py "data.val_hives=[]" training.epochs=50

# Інша частка val для ramp
uv run src/run_experiment.py experiment=ramp_detection data.val_ratio=0.15
```

> При зміні split старий `datasets/pose/data.yaml` треба видалити вручну,
> щоб примусити повторну підготовку:
> ```bash
> rm datasets/pose/data.yaml
> ```

---

## Запуск

### Локальний тест (1 epoch, без реального датасету)

```bash
# pose (завантажує ~200 MB якщо ще нема)
uv run src/run_experiment.py training.epochs=1

# ramp (~51 MB)
uv run src/run_experiment.py experiment=ramp_detection training.epochs=1

# Якщо дані вже є в datasets/raw/ — можна з source_type=local (не завантажує з Kaggle)
uv run src/run_experiment.py data.source_type=local training.epochs=1 training.workers=0 training.imgsz=64 training.batch=2
```

### Повне навчання

```bash
uv run src/run_experiment.py training.epochs=50
uv run src/run_experiment.py experiment=ramp_detection training.epochs=100
```

### Перевизначення гіперпараметрів (Hydra CLI)

```bash
uv run src/run_experiment.py training.imgsz=1280 training.batch=8 project.note="test lr"
```

### Grid sweep (Hydra multirun)

```bash
# Запускає 4 комбінації: 2 lr × 2 imgsz
uv run src/run_experiment.py --multirun \
  training.lr0=0.001,0.0005 \
  training.imgsz=1280,1920
```

Кожна комбінація — окремий MLflow run. Результати у `multirun/`.

### HPO — Optuna

```bash
uv run src/run_experiment.py mode=hpo hpo.n_trials=20
uv run src/run_experiment.py experiment=ramp_detection mode=hpo hpo.n_trials=10
```

Простір пошуку задається в `config/config.yaml` у секції `hpo.search_space`. Кожен trial — вкладений MLflow run. Найкращі параметри логуються в батьківський run.

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

# ── Клітинка 2: Ключі доступу (або Colab Secrets → без коду) ─────────────
os.environ["DAGSHUB_USER"]  = "spiowm"
os.environ["DAGSHUB_REPO"]  = "bee-monitoring-system"
os.environ["DAGSHUB_TOKEN"] = "your_dagshub_token"
os.environ["KAGGLE_KEY"]    = "KGAT_your_kaggle_token"

# ── Клітинка 3: Тренування ────────────────────────────────────────────────
# Датасет завантажується автоматично (~200 MB) при першому запуску
!uv run src/run_experiment.py training.epochs=50 training.batch=8

# Або ramp detection (~51 MB)
# !uv run src/run_experiment.py experiment=ramp_detection training.epochs=100

# Або HPO
# !uv run src/run_experiment.py mode=hpo hpo.n_trials=20 training.epochs=30
```

> Артефакти (best.pt, метрики, графіки) автоматично зберігаються в DagsHub/MLflow.

**Colab Secrets замість хардкоду:**
В Colab → 🔑 Secrets → додати `KAGGLE_KEY`, `DAGSHUB_TOKEN` тощо. Тоді замість `os.environ[...]`:
```python
from google.colab import userdata
os.environ["KAGGLE_KEY"]    = userdata.get("KAGGLE_KEY")
os.environ["DAGSHUB_TOKEN"] = userdata.get("DAGSHUB_TOKEN")
```

---

## Kaggle Notebooks

На Kaggle датасети підключаються через UI — не завантажуються програмно.

1. Відкрити ноутбук → **Add data** → знайти `spiowm/bee-monitoring-pose` або `spiowm/bee-monitoring-ramp-detection` → Add
2. Дані будуть в `/kaggle/input/bee-monitoring-pose/` або `/kaggle/input/bee-monitoring-ramp-detection/`
3. Запускати з `source_type=local`:

```python
!git clone https://github.com/spiowm/bee-monitoring-system.git
import os
os.chdir("bee-monitoring-system/research")
!uv sync

os.environ["DAGSHUB_USER"]  = "spiowm"
os.environ["DAGSHUB_REPO"]  = "bee-monitoring-system"
os.environ["DAGSHUB_TOKEN"] = "your_token"

# Датасет вже є — вказуємо шлях, не завантажуємо
!uv run src/run_experiment.py \
  data.source_type=local \
  data.raw_dir=/kaggle/input/bee-monitoring-pose \
  training.epochs=50 training.batch=8
```

---

## DagsHub / MLflow — що логується

| Артефакт | Де |
|----------|-----|
| `hydra_config.yaml` | Повна конфігурація запуску |
| `yolo_run/weights/best.pt` | Найкраща модель |
| `yolo_run/weights/last.pt` | Остання модель |
| `yolo_run/results.csv` | Метрики по epochs |
| `yolo_run/*.png` | Loss curves, confusion matrix, PR curve |
| Параметри + метрики | MLflow params/metrics |

Для HPO: вкладені runs для кожного trial + `best_*` параметри в батьківському run.

---

## Конфігурація

### Структура (Hydra)

```
config/config.yaml          ← глобальні дефолти
config/experiment/*.yaml    ← перевизначення для конкретного завдання
```

### Ключові параметри

| Параметр | Дефолт | Опис |
|----------|--------|------|
| `mode` | `single` | `single` \| `hpo` |
| `training.epochs` | `1` | Epoch count |
| `training.imgsz` | `1920` | Input resolution |
| `training.batch` | `4` | Batch size |
| `training.lr0` | `0.001` | Initial learning rate |
| `data.source_type` | `kaggle` | `kaggle` — auto-download; `local` — взяти з `raw_dir` |
| `data.val_hives` | `[]` | Hive IDs для val (якщо `[]` — random split) |
| `data.val_ratio` | `0.2` | Частка val при random split |
| `hpo.n_trials` | `10` | Кількість Optuna trials |

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
  val_hives: ["20230711b", "20230609e"]

model:
  name: "yolo11m-pose.pt"
  task: "pose"
```

Запуск:
```bash
uv run src/run_experiment.py experiment=my_experiment training.epochs=50
```
