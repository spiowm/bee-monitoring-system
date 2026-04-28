# Backend — FastAPI + YOLO

Відео-пайплайн: YOLO-детекція → ByteTrack → підрахунок трафіку → анотоване H.264 відео → MongoDB.

## Запуск

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
- **Ultralytics YOLO** (детекція + pose estimation)
- **Supervision** (ByteTrack / OC-SORT трекінг)
- **OpenCV** + FFmpeg (відео процесинг)
- **Pydantic v2** + pydantic-settings

## Структура

```
backend/
├── main.py              # FastAPI app + lifespan (MongoDB + YOLO warm-up)
├── config.py            # Settings з .env
├── schemas/schemas.py   # ProcessConfig, VizConfig, JobCreateResponse
├── db/mongodb.py        # Motor клієнт
├── routers/
│   ├── jobs.py          # POST /jobs, GET /jobs/{id}, GET /jobs/{id}/live, DELETE
│   └── analytics.py    # GET /analytics/summary, GET /analytics/compare-approaches
└── services/
    ├── video_processor.py   # Точка входу: get_bee_model() singleton, process_video()
    ├── pipeline.py          # VideoPipeline: оркестрація стадій
    ├── pipeline_stages.py   # 6 стадій: Detection→Tracking→Update→Behavior→Counting→Annotation
    ├── ramp_detector.py     # Singleton ramp bbox (оновлюється кожні N кадрів)
    ├── counter.py           # TrafficCounter: Approach A і B
    ├── orientation.py       # Вектор голова→жало з keypoints
    ├── behavior.py          # Heuristic класифікація (foraging/fanning/guarding/washboarding)
    └── ffmpeg_service.py    # raw mp4 → H.264 з faststart
```

## Два підходи підрахунку

**Approach A** — тільки траєкторія: перетин горизонтальної лінії на рампі.

**Approach B** — pose-валідований: також перевіряє що вектор голова→жало збігається з напрямком руху (в межах `angle_threshold_deg`). При відсутності keypoints — fallback на A. Події теговані `pose_confirmed` / `trajectory_fallback`.

## Поведінковий аналіз

4 класи, налаштовуються через `ProcessConfig`:
- **foraging** — висока швидкість
- **fanning** — дуже повільно, довго
- **guarding** — середня швидкість, горизонтальний рух
- **washboarding** — все інше

Пороги: `behavior_foraging_speed_min`, `behavior_fanning_speed_max`, тощо.
