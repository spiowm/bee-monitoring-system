# Backend — FastAPI + YOLO

Відео-пайплайн з 7 стадій: YOLO-детекція → ByteTrack → класифікація поведінки (PLOS ONE 2025) → детекція захисних кластерів → підрахунок трафіку (з прозорим pose-фільтром) → анотоване H.264 відео → MongoDB. Плюс — evaluation проти ground-truth і rule-based рекомендації пасічнику.

## Запуск

> **Увага:** Для фінальної конвертації відео у формат H.264 (для показу в браузері) на сервері має бути встановлений `ffmpeg`.
> - Ubuntu/Debian: `sudo apt install ffmpeg`
> - MacOS: `brew install ffmpeg`

```bash
cd backend
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Потрібен файл `backend/.env`:

```env
MONGO_URI=mongodb://localhost:27017
```

API docs: `http://localhost:8000/docs`

## Стек

- **FastAPI** + Motor (async MongoDB)
- **Ultralytics YOLO** (детекція + 2-keypoint pose estimation)
- **Supervision** (ByteTrack трекінг)
- **OpenCV** + FFmpeg (відео процесинг)
- **Pydantic v2** + pydantic-settings

## Структура

```
backend/
├── main.py                  # FastAPI app + lifespan (MongoDB + YOLO warm-up)
├── config.py                # Settings з .env
├── schemas/schemas.py       # ProcessConfig, VizConfig, JobCreate/Live/Evaluation схеми
├── db/mongodb.py            # Motor клієнт
├── routers/
│   ├── jobs.py              # POST /jobs, /jobs/test, /jobs/evaluate; GET /jobs, /jobs/{id}, /live; DELETE
│   └── analytics.py         # GET /analytics/summary, /analytics/compare-approaches
└── services/
    ├── video_processor.py   # Точка входу: get_bee_model() singleton, process_video()
    ├── pipeline.py          # VideoPipeline: оркестрація 7 стадій
    ├── pipeline_stages.py   # FrameContext + 7 стадій (Detection→Tracking→…→Defense→Counting→Annotation)
    ├── ramp_detector.py     # Singleton ramp bbox (оновлюється кожні N кадрів)
    ├── tracker_factory.py   # ByteTrack (єдиний підтримуваний)
    ├── track_history.py     # TrackEntry + max_displacement, zero_cross_rate (для behavior)
    ├── counter.py           # TrafficCounter: повертає (events, rejected_events)
    ├── orientation.py       # head→stinger, vector_to_entrance, aligned, get_angular_error
    ├── behavior.py          # PLOS ONE класифікація: Foraging / Fanning / Washboarding
    ├── annotator.py         # FrameAnnotator: бокси, ключові точки, glow IN/OUT/REJECTED, defense circle
    ├── ffmpeg_service.py    # raw mp4 → H.264 з faststart
    ├── recommendations.py   # Rule-based двигун порад пасічнику (~12 правил)
    └── evaluation/          # GT evaluation submodule
        ├── gt_loader.py     # парсер tracks_and_behavior_classes_*.txt
        ├── counting_eval.py # GT events + greedy matching (TP/FP/FN, F1, MAE)
        ├── gt_annotator.py  # рендер GT-відео (жовті бокси)
        └── evaluator.py     # оркестратор run_evaluation
```

## Два підходи підрахунку

**Approach A — траєкторія:**
Перетин горизонтальної лінії на рампі. Усі перетини зараховані з `method="trajectory_only"`.

**Approach B — pose-validated:**
Той самий перетин + перевірка вектора голова→жало проти напрямку руху (поріг `angle_threshold_deg`, default ±60°).
- `pose_confirmed` — ключові точки є й узгоджені → зараховано
- `trajectory_fallback` — ключових точок немає → зараховано без перевірки
- `rejected` — ключові точки є, але кут не пройшов → **НЕ зараховано**, але збережено в `rejected_events` для аудиту

`counter.update()` повертає `(events, rejected_events)`. Rejected стрім видно скрізь:
- `live_stats.pose_rejected` — лічильник у live UI
- `result.pose_rejected_events` — фінальне число
- На відео — сірий glow «✗ pose» 1.5 с навколо відсіяної бджоли
- `PoseFilterCard` у JobDetailModal — breakdown «А зарахував би N, B залишив M, відсіяв K»

## Класифікація поведінки (PLOS ONE 2025)

Per-track у `behavior.py:HeuristicBehaviorStrategy`. Кожен трек класифікується (порядок пріоритету):

1. **Foraging (Фуражування)** — `avg_speed > 100 px/s` І рух у напрямку льотка ±60°.
2. **Fanning (Вентиляція)** — `max_displacement < 10 px` І `duration > 1 с` І тіло спрямоване до льотка ±90°.
3. **Washboarding (Полірування)** — `avg_speed < 60 px/s` І `duration > 2 с` І `zero_cross_rate > 2 Hz` (періодичні зміни знаку прискорення).

**Defense (Захист)** — окрема стадія `DefenseStage` (multi-bee):
- Для кожної бджоли A знаходить ≥2 сусідок у радіусі `2 × bee_length`, чий вектор тіла спрямований на A в межах ±45°
- Якщо кластер тримається ≥1 с — всі учасники отримують `behavior="defense"`
- Annotator малює червоне коло «DEFENSE»

## Evaluation проти ground-truth

`POST /jobs/evaluate` приймає `{filename, gt_basename, config, viz_config}` і запускає `evaluator.run_evaluation`:

1. Звичайний пайплайн → predicted events.
2. Завантажує GT з `research/datasets/raw/tracking_and_behavior/`:
   - `tracks_and_behavior_classes_<basename>.txt`
   - `entrance_zone_<basename>.txt`
3. Реплеїть GT-треки через лінію → `gt_events`.
4. Greedy matching pred ↔ GT із вікном ±15 кадрів → TP/FP/FN/F1/MAE окремо для IN та OUT.
5. Рендерить другий mp4 з жовтими GT-боксами (для side-by-side).
6. Записує `result.evaluation` у Mongo.

`GET /jobs/evaluate/test-pairs` повертає список доступних пар:
- `20230711a-fan` (6000 кадрів @ 50 FPS, 127k GT детекцій, фанінг)
- `20230711b-fan`
- `20230609b-def` (defense / robbery — для тесту DefenseStage)
- `yt8`

Frontend `/evaluation` запускає **A і B паралельно** на одному відео і показує 3-колонне відео + Winner Bars.

## Рекомендації пасічнику

`recommendations.py:generate_recommendations(result)` повертає список `Recommendation(severity, icon, title, description, action)`. ~12 детермінованих правил:

- 🚨 **critical** — defense events, перегрів вулика (fanning > 50%), масовий вилет
- ⚠️ **warning** — інтенсивна вентиляція, дисбаланс трафіку, льоток не виявлено
- ℹ️ **info** — активність фуражування (висока/низька), низький трафік, pose-filter ефективність

Серіалізується в `result["recommendations"]` і рендериться компонентом `RecommendationsSection.tsx`.

## MongoDB

База: `buzz_buzz_buzz`, колекція: `jobs`. Документ:

```
{
  job_id, filename, status, progress, created_at,
  config: { ProcessConfig },
  viz_config: { VizConfig },
  live_stats: { current_frame, total_in, total_out, pose_confirmed,
                pose_rejected, behavior_counts, recent_events: [...10] },
  result: {
    total_in, total_out, fps_processed, approach_used,
    pose_confirmed_events, fallback_events, pose_rejected_events,
    ramp_detected,
    behavior_summary: { foraging_, fanning_, washboarding_, defense_detections },
    events: [...], rejected_events: [...], defense_events: [...],
    annotated_video_url, recommendations: [...],
  },
  evaluation: { ... } | null,
  gt_basename, error,
}
```

## Ключові поля ProcessConfig

| Поле | Default | Що робить |
|------|---------|-----------|
| `approach` | `"A"` | `"A"` (траєкторія) / `"B"` (pose-validated) |
| `tracker_name` | `"bytetrack"` | єдиний підтримуваний (kwarg для back-compat) |
| `line_position` | `0.5` | Лінія підрахунку як частка висоти рампи |
| `conf_threshold` | `0.20` | YOLO confidence cutoff |
| `kp_conf_threshold` | `0.5` | Поріг впевненості ключових точок |
| `angle_threshold_deg` | `60.0` | Approach B: max кут між рухом і позою |
| `behavior_foraging_speed_min` | `100` | Vfor (px/s) |
| `behavior_foraging_angle_deg` | `60` | Afor (рух → льоток) |
| `behavior_fanning_max_displacement_px` | `10` | Dfan |
| `behavior_fanning_duration_min` | `1.0` | Tfan (с) |
| `behavior_fanning_angle_deg` | `90` | Afan (тіло → льоток) |
| `behavior_washboarding_speed_max` | `60` | Vwash |
| `behavior_washboarding_duration_min` | `2.0` | Twash (с) |
| `behavior_washboarding_zcr_min` | `2.0` | Zero-cross rate (Hz) |
| `defense_radius_factor` | `2.0` | Rdef = factor × довжина бджоли |
| `defense_angle_deg` | `45.0` | Adef |
| `defense_min_defenders` | `2` | Ndef |
| `defense_duration_sec` | `1.0` | Tdef |

## Посилання

- Академічна специфікація: PLOS ONE 2025 — *Visual recognition of honeybee behavior patterns at the hive entrance*
- Ground-truth датасет: `research/datasets/raw/tracking_and_behavior/` (gitignored)
