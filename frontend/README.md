# Frontend: React dashboard

Web UI for uploading hive videos, watching processing live, browsing past runs and checking accuracy against ground truth. The interface language is Ukrainian.

## Running

```bash
cd frontend
cp .env.example .env    # VITE_API_URL=http://localhost:8000
npm install
npm run dev             # http://localhost:5173
```

Other scripts:

```bash
npm run build          # type check + production build
npm run lint           # ESLint
npm run generate-api   # regenerate the API client from ../backend/openapi.json
```

## Stack

- React 19, TypeScript, Vite
- TanStack Query for server state and polling
- Recharts for charts
- Tailwind utilities plus CSS variables as design tokens
- API client generated from the backend's OpenAPI schema with `@hey-api/openapi-ts`
- lucide-react icons

## Pages

**`/` Upload.** Pick a demo video or drop your own file, optionally open advanced settings (Approach A or B, thresholds, overlays, fast mode without video) and start. While the job runs, the page polls `GET /jobs/{id}/live` every 2 seconds and shows counts, bees on the ramp, behavior bars and a ticker of the latest crossing events. When it finishes, the annotated video plays inline with beekeeper recommendations below it.

**`/analytics` Analytics.** Totals across runs, behavior distribution, traffic per run and a history table. Each run opens a details modal (statistics, pose filter breakdown, recommendations, events). Selecting 2 to 4 runs opens a comparison with bar and radar charts, for example Approach A versus B on the same video.

**`/evaluation` Accuracy.** Choose a video that has ground truth and run a behavior evaluation. The result shows overall accuracy, per-class precision, recall and F1, event-level foraging metrics, a confusion matrix, and the ground-truth video next to BuzzTrack's output, played in sync. The report can be downloaded as Markdown.

## Structure

```
src/
├── api/generated/                 # generated client, do not edit by hand
├── hooks/useLocalStorageState.ts  # useState persisted to localStorage
├── pages/
│   ├── Upload.tsx
│   ├── Analytics.tsx
│   └── Evaluation.tsx
└── components/
    ├── JobConfigPanel.tsx         # video choice and processing settings
    ├── LiveStatsPanel.tsx         # live counts, behavior bars, event ticker
    ├── EventTicker.tsx            # last 10 crossing events
    ├── PoseFilterCard.tsx         # Approach B: what the pose filter rejected
    ├── RecommendationsSection.tsx # beekeeper recommendation cards
    ├── BehaviorLegend.tsx         # behavior class descriptions on hover
    ├── JobDetailModal.tsx         # details of one run
    ├── RunComparisonModal.tsx     # comparison of 2 to 4 runs
    ├── BehaviorEvalResults.tsx    # accuracy metrics and confusion matrix
    ├── VideoPlayer.tsx            # player with speed control and shortcuts
    └── SideBySideVideoPlayer.tsx  # several videos played in sync
```

Use the generated client in `src/api/generated` for API calls. Links to videos and the Markdown report are built from `VITE_API_URL`.

## Video player shortcuts

When the player has focus:

| Key | Action |
|---|---|
| Space / K | play or pause |
| ← / → | seek ±5 s |
| J / L | seek ±10 s |
| , / . | previous or next frame |
| 0 to 9 | jump to 0%, 10%, …, 90% |
| M | mute |
| + / − | speed up or down (0.25× to 4×) |

`SideBySideVideoPlayer` supports the same keys; the first video is the master and the others follow its time.

## Design tokens

CSS variables live in `src/index.css`. Use the semantic colours (`--color-in`, `--color-out`, `--behavior-*`) rather than raw Tailwind colours:

| Behavior | Variable | Colour |
|---|---|---|
| Foraging | `--behavior-foraging` | green |
| Fanning | `--behavior-fanning` | blue |
| Washboarding | `--behavior-washboarding` | orange |
| Defense | `--behavior-defense` | red |
