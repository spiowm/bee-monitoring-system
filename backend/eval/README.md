# `eval/` — інструменти оцінки та діагностики behavior-класифікації

Це **офлайн-інструменти** (не частина рантайму API). Запускати з теки `backend/`
через `-m`, щоб пакети `config`/`services`/`eval` коректно резолвились:

```bash
cd backend
uv run python -m eval.eval_fast                 # швидкий евал усіх 3 відео
uv run python -m eval.eval_cli --pair 20230609b-def   # реальний YOLO (для валідації)
uv run python -m eval.diag_features --pair 20230711a-fan
```

## Модулі
| Файл | Призначення |
|------|-------------|
| `eval_fast.py` | Швидкий евал через **кеш детекцій** (~150 fps замість 15). `--pairs`, `--set k=v`, `--json`, `--rebuild`, зведена таблиця по 3 відео. Метрики **ідентичні** повному пайплайну. |
| `eval_cli.py` | Евал одним РЕАЛЬНИМ пайплайном (некешований YOLO + анотація). Для фінальної валідації, що кеш-результати тримаються в продакшені. |
| `detection_cache.py` | Кешує сирі YOLO-детекції у `backend/data/eval_cache/*.pkl`. YOLO — найдорожча частина; кешується раз. Детекції НЕ залежать від behavior/defense/stitch параметрів. |
| `diag_features.py` | Розподіли ознак (zcr/motion/max_disp/speed) по групах GT×Pred (TP/FP/FN) — щоб бачити, який поріг реально розділяє класи. |
| `diag_frame_presence.py` | Переоцінка наших передбачень **метрикою статті** (frame-presence, multi-label). |
| `parse_eval.py` | Витяг ключових метрик зі summary-JSON (для свіпів). |
| `run_validate.sh` | Реальний пайплайн на всіх 3 відео (через `-m eval.eval_cli`). |
| `baseline_logs/` | Історичні логи бейзлайну (довідково). |

## Дві метрики (важливо для роботи)
- **Сувора (per-bee IoU)** — наш стандарт: кожну бджолу треба правильно класифікувати. Чесна, але строга.
- **Frame-presence (як Sledevič et al. 2025)** — на рівні кадру, multi-label, з класом `background`. Поблажлива; порівнянна з їхніми 87%. Рахується автоматично (`build_behavior_evaluation` → поле `frame_presence`).

## Оптимальний конфіг
`backend/config/eval_config.yaml` читається `eval_fast`/`eval_cli` і мерджиться поверх
дефолтів коду. Ті самі значення зашиті в `schemas.ProcessConfig` (продакшен-UI).
Ключова знахідка: **ZCR насичений (~38 Гц у всіх) і марний; справжній дискримінатор
fanning — стаціонарність (`behavior_fanning_max_disp`).**

## Кеш
Будується автоматично за потреби; форс — `--rebuild`. Інвалідовується при зміні
conf/iou/imgsz/моделі. Зберігається в `backend/data/eval_cache/`.
