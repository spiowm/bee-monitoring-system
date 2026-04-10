# Research & MLOps Environment

Тут містяться всі скрипти, ноутбуки та конфігурації для проведення ML експериментів (навчання YOLO моделей) із трекінгом у DagsHub (MLflow) та керуванням датасетами через Kaggle.

## Структура

- `src/` — вихідні коди для навчання та керування даними.
  - `run_experiment.py` — головна точка входу для запуску навчання.
  - `data_manager.py` — логіка завантаження датасетів з Kaggle.
- `config/` — конфігураційні файли експериментів.
  - `config.yaml` — єдине місце для зміни всіх гіперпараметрів, моделі та назви експерименту.
- `notebooks/` — Jupyter notebooks для EDA (розвідувального аналізу), кастомних перевірок та швидких прототипів.
- `pyproject.toml` — залежності проєкту (kerowani `uv`).
- `.env` — ваші приватні ключі (не пушиться в git).

## Встановлення та Запуск (за допомогою `uv`)

Ми використовуємо **`uv`** як основний менеджер середовищ та пакетів (це в десятки разів швидше за звичайний `pip`).

1. У папці `research` створіть віртуальне середовище та встановіть все необхідне:
   ```bash
   uv sync
   ```
   *Це створить папку `.venv` і встановить всі пакети з `pyproject.toml`.*

2. Активуйте середовище:
   ```bash
   source .venv/bin/activate
   ```

3. Запустіть експеримент:
   ```bash
   uv run src/run_experiment.py
   ```

## Робота з Kaggle (Датасети > 7ГБ)

Оскільки датасет важить кілька гігабайтів, найкращою практикою є завантажити його з Kaggle безпосередньо під час запуску в середовище (особливо зручно для Colab).

### Як підготувати дані на Kaggle
Вам **не обов'язково** самому писати скрипти для архівації по папках. Зробіть так:
1. Майте локально папку з форматом YOLO (де є `data.yaml`, папки `train/`, `val/`, `test/`).
2. Заархівуйте це все у звичайний `.zip`.
3. Створіть на Kaggle новий **Dataset** і просто завантажте цей `.zip` (Kaggle сам його розархівує на своїх серверах у правильну структуру).
4. Отримайте його ідентифікатор виду `ваш_юзер/назва-датасету`.
5. Вставте цей ID у наш `config/config.yaml` у поле `kaggle_id`.

### Авторизація Kaggle API
Щоб скрипт зміг стягнути дані, налаштуйте Kaggle токен. Є два варіанти:
1. Завантажте `kaggle.json` (з налаштувань профілю Kaggle) і помістіть його у `~/.kaggle/kaggle.json`.
2. АБО в Colab / у ваш робочий `.env` файл додайте:
   ```env
   KAGGLE_USERNAME=ваш_юзер
   KAGGLE_KEY=ваш_ключ
   ```

Наш скрипт `src/data_manager.py` робить `kaggle datasets download -d <ID> --unzip`. Він стягне архів і автоматично покладе розпаковані файли (з вашим `data.yaml`) у вказану папку `datasets/kaggle_dataset`.

## Google Colab
В Google Colab запуск повноцінного навчання виглядатиме так:

```python
# 1. Встановлення середовища
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] += ":/root/.cargo/bin"

# 2. Клонування і запуск
!git clone https://github.com/spiowm/bee-monitoring-system.git
%cd bee-monitoring-system/research
!uv sync

# 3. Ключі доступу (або використовуйте Colab Secrets)
os.environ["DAGSHUB_TOKEN"] = "твій_дагсхаб_токен"
os.environ["DAGSHUB_USER"] = "oleksiitatar"
os.environ["DAGSHUB_REPO"] = "bee-monitoring-system"
os.environ["KAGGLE_USERNAME"] = "твій_кагл_юзер"
os.environ["KAGGLE_KEY"] = "твій_кагл_ключ"

# 4. Запуск!
!uv run src/run_experiment.py
```
Всі моделі (`best.pt`), метрики, деталі параметрів автоматично будуть "летіти" у ваш DagsHub репозиторій на вкладку Experiments.
