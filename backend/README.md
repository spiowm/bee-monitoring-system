# Backend: FastAPI + YOLO video pipeline

REST API that takes a hive entrance video, runs it through a 7-stage computer vision pipeline in a background job, stores statistics in MongoDB and serves an annotated H.264 video back to the dashboard. It also scores itself against a ground-truth dataset and turns statistics into beekeeper recommendations.

## Running

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), FFmpeg (to convert the output video to browser-friendly H.264), MongoDB.

```bash
cd backend
cp .env.example .env
uv sync
uv run uvicorn main:app --port 8000
```

Interactive API docs: http://localhost:8000/docs

`.env` (see `.env.example` for all options):

```env
MONGO_URI=mongodb://localhost:27017
DB_NAME=buzz_buzz_buzz
CORS_ORIGINS=http://localhost:5173
MAX_VIDEO_SIZE_MB=500
```

Tests:

```bash
uv run pytest
```

After changing the API, regenerate the OpenAPI schema that the frontend client is built from:

```bash
uv run python export_openapi.py
```

## Stack

- FastAPI + Motor (async MongoDB driver), Pydantic v2, pydantic-settings
- Ultralytics YOLO11-pose: detection plus 2 keypoints (head, stinger) per bee; a second pose model finds the 4 corners of the landing ramp
- Supervision for ByteTrack tracking
- OpenCV and FFmpeg for video I/O

PyTorch is installed from the CUDA 12.1 wheel index (see `pyproject.toml`); the pipeline falls back to CPU without a GPU.

## Pipeline

`services/pipeline.py` runs these stages per frame (`services/pipeline_stages.py`):

| Stage | What it does |
|---|---|
| Detection | YOLO11-pose boxes and keypoints; ramp detection every N frames (the ramp does not move) |
| Tracking | ByteTrack |
| TrackUpdate | track history, short-gap stitching of broken tracks (temporal re-ID) |
| Behavior | per-track rule-based classification: foraging, fanning, washboarding |
| Defense | multi-bee guard clusters around an intruder |
| Counting | line crossings along the entrance edge of the ramp, Approach A or B |
| Annotation | boxes, arrows, ramp, counters, event glows; skipped in fast mode |

### Two counting approaches

- **A, trajectory only.** Every crossing of the line is counted.
- **B, pose-validated (default).** The same crossing is checked against the head-to-stinger vector: if keypoints exist and the body direction disagrees with the motion direction by more than `angle_threshold_deg` (60° by default), the event is rejected. Crossings without keypoints are counted as a fallback.

Rejected events are kept for audit, not dropped silently: they appear in live stats (`pose_rejected`), in the final result, on the video as a grey "✗ pose" glow, and in the job details card ("A would count N, B kept M, rejected K").

### Behavior rules

Adapted from the definitions in Sledevič et al., PLOS ONE 2025. Thresholds were tuned on full ground-truth videos with the offline tools in [`eval/`](eval/README.md).

1. **Fanning (ventilation):** the bee stays in place (`max_disp < 60 px`) for longer than 0.6 s while moving its wings. It takes priority over foraging for stationary bees, because detection jitter inflates their speed. Body orientation is only a soft bonus, since about half of the bees have no confident keypoints.
2. **Foraging:** speed above 100 px/s towards the entrance (±60°).
3. **Washboarding:** slow (5 to 60 px/s), long (over 2 s), rocking motion; rare in the data.
4. **Defense** (separate stage): at least 2 neighbours within `2 × bee length` whose bodies point at the same bee (±45°), confirmed if the cluster reappears within a 60-frame window.

A finding from tuning: the zero-crossing rate of motion is saturated (about 38 Hz for every bee) and useless as a fanning feature; staying in place (`max_disp`) is what actually separates fanning bees.

### Fast mode

`config.skip_video = true` computes statistics without rendering the annotated video, which is 2 to 3 times faster.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/jobs` | upload a video with `config` and `viz_config` (multipart) and start processing |
| `POST` | `/jobs/test` | process a demo video from the ground-truth dataset |
| `GET` | `/jobs/test/videos` | list demo videos |
| `GET` | `/jobs` | list jobs |
| `GET` | `/jobs/{id}` | full job document |
| `GET` | `/jobs/{id}/live` | live statistics while processing |
| `DELETE` | `/jobs/{id}` | delete a job and its videos |
| `GET` | `/jobs/evaluate/test-pairs` | videos that have ground truth |
| `POST` | `/jobs/evaluate` | process a video and score behavior against ground truth |
| `GET` | `/analytics/summary` | totals across finished jobs |
| `GET` | `/analytics/compare-approaches` | average counts of Approach A versus B |
| `GET` | `/analytics/export/md` | Markdown report for one or two jobs |
| `GET` | `/models` | available pose models with their architecture and training image size |

Processed videos are served from `/static/output/`.

## Evaluation against ground truth

`POST /jobs/evaluate` (`services/evaluation/evaluator.py`):

1. Runs the normal pipeline and keeps per-frame behavior predictions.
2. Loads ground truth from `research/datasets/raw/tracking_and_behavior/<video>/`: `tracks_and_behavior.txt` (per-frame boxes with 4 behavior flags) and `entrance_zone.txt` (entrance polygon).
3. Scores behavior two ways: strict per-bee matching (IoU ≥ 0.3) and per-frame presence, the metric used by Sledevič et al.; plus event-level foraging and defense metrics.
4. Renders a second video with the ground-truth labels for side-by-side viewing.
5. Saves the result to `job.evaluation`.

Counting accuracy (A versus B), threshold sweeps and the permutation test live in the offline tools in [`eval/`](eval/README.md).

## Beekeeper recommendations

`services/recommendations.py` holds 12 deterministic rules that turn a finished job into cards with a severity, title, explanation and suggested action, for example:

- **critical:** frequent defense reactions, possible overheating (more than half of the behavior is fanning), mass departure without return
- **warning:** intense ventilation, traffic imbalance, entrance not detected, low pose validation quality
- **info:** high or low foraging activity, low overall traffic, active washboarding

## Layout

```
backend/
├── main.py                  # FastAPI app; startup connects MongoDB and warms up YOLO
├── config.py                # settings from .env
├── export_openapi.py        # writes openapi.json for the frontend client
├── schemas/schemas.py       # ProcessConfig, VizConfig and API models
├── db/mongodb.py            # Motor client
├── routers/
│   ├── jobs.py              # jobs, demo videos, evaluation
│   └── analytics.py         # summary, A vs B, Markdown export
├── services/
│   ├── video_processor.py   # entry point: model singleton, process_video()
│   ├── pipeline.py          # VideoPipeline: runs the stages
│   ├── pipeline_stages.py   # the 7 stages
│   ├── ramp_detector.py     # ramp keypoints, refreshed every N frames
│   ├── tracker_factory.py   # ByteTrack
│   ├── track_history.py     # per-track features (displacement, speed, zero-crossing rate)
│   ├── temporal_motion.py   # wing motion signal
│   ├── behavior.py          # behavior rules
│   ├── counter.py           # line crossings, Approach A and B
│   ├── orientation.py       # head-to-stinger vectors and angles
│   ├── annotator.py         # drawing on frames
│   ├── ffmpeg_service.py    # raw mp4 to H.264 with faststart
│   ├── recommendations.py   # beekeeper rules
│   └── evaluation/          # ground-truth loading, matching, metrics, GT video
├── eval/                    # offline evaluation and diagnostics, not part of the API
├── config/eval_config.yaml  # tuned behavior thresholds used by eval/
├── tests/                   # unit tests for counting and orientation
└── data/models/             # trained weights: bee_pose/best.pt, ramp_detector/best.pt
```

## Main `ProcessConfig` fields

| Field | Default | Meaning |
|---|---|---|
| `approach` | `"B"` | `"A"` trajectory only, `"B"` pose-validated |
| `conf_threshold` | `0.20` | YOLO confidence cutoff |
| `kp_conf_threshold` | `0.5` | keypoint confidence cutoff |
| `angle_threshold_deg` | `60.0` | Approach B: max angle between motion and body direction |
| `imgsz` | model's training size | inference image size |
| `skip_video` | `false` | fast mode, no annotated video |
| `behavior_foraging_speed_min` | `100` | foraging speed, px/s |
| `behavior_fanning_max_disp` | `60` | max displacement of a fanning bee, px |
| `behavior_fanning_duration_min` | `0.6` | min fanning duration, s |
| `behavior_washboarding_speed_max` | `60` | max washboarding speed, px/s |
| `defense_radius_factor` | `2.0` | defense radius in bee lengths |
| `defense_min_defenders` | `2` | min guards around one bee |
| `defense_angle_deg` | `45.0` | max angle between a guard's body and the direction to the intruder |
| `stitch_max_dist` / `stitch_max_frames` | `30` / `45` | track stitching limits |

The full list with comments is in `schemas/schemas.py`.
