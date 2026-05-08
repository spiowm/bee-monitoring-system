# Frontend — React + TypeScript

Дашборд для завантаження відео, live-моніторингу процесингу, дослідницької аналітики та оцінки точності системи проти ground-truth датасету.

## Запуск

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

Файл `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

## Стек

- React 19 + TypeScript + Vite
- TailwindCSS (утиліти) + CSS variables (дизайн-токени)
- TanStack React Query v5 (серверний стан)
- Recharts (графіки)
- `@hey-api/openapi-ts` — генерований TypeScript клієнт з OpenAPI бекенда
- lucide-react (іконки)

## Структура

```
src/
├── api/generated/                 # Авто-генерований клієнт — не редагувати вручну
├── hooks/
│   └── useLocalStorageState.ts    # SSR-safe useState з мапінгом у localStorage
├── pages/
│   ├── Upload.tsx                 # Завантаження + конфіг + live + результат
│   ├── Analytics.tsx              # Hero A vs B + історія + порівняння runs
│   └── Evaluation.tsx             # Auto A vs B vs GT з 3-колонним відео
└── components/
    ├── JobConfigPanel.tsx         # Quick Start + advanced toggle
    ├── LiveStatsPanel.tsx         # IN/OUT, FPS, pose breakdown, behavior bars, EventTicker
    ├── JobDetailModal.tsx         # Деталі job: header chips, BehaviorLegend,
    │                              #   PoseFilterCard, RecommendationsSection, графіки, події
    ├── RunComparisonModal.tsx     # Порівняння 2-4 jobs (BarChart + RadarChart)
    ├── VideoPlayer.tsx            # 0.25×–4× швидкість + клавіатура (Space, ← →, J/K/L, …)
    ├── SideBySideVideoPlayer.tsx  # Синхронізовані N відео для Evaluation
    ├── EventTicker.tsx            # Стрічка останніх 10 подій
    ├── BehaviorLegend.tsx         # Hover-popover опис поведінки
    ├── PoseFilterCard.tsx         # Approach B: «А зарахував би N, B залишив M, відсіяв K»
    ├── RecommendationsSection.tsx # Картки рекомендацій з severity badges
    ├── EvaluationKPICards.tsx     # 2×2 KPI блоки на Evaluation
    └── ApproachWinnerBars.tsx     # Метрики A vs B з 🏆 переможцем
```

## API клієнт

Генерується з OpenAPI схеми бекенда (бекенд має бути запущений):

```bash
npm run generate-api
```

Імпортувати тільки з `./api/generated` — ніяких прямих fetch/axios.

## Сторінки

**`/` Завантаження** — основний workflow:
1. Hero empty state з демо-кнопками тестових відео
2. Sidebar `JobConfigPanel` у Quick Start mode (drag-drop + Approach + Start); решта — за toggle «Розширені налаштування»
3. Submission → polling `GET /jobs/{id}/live` кожні 2с
4. `LiveStatsPanel`: IN/OUT/FPS, pose-confirmed/fallback/rejected (Approach B), поведінкові бари, BehaviorLegend, EventTicker (останні 10 подій з кольорами IN/OUT/REJECTED)
5. Після завершення — вбудований `VideoPlayer` зі швидкостями і клавішами

**`/analytics` Аналітика** — дослідницький дашборд:
- Empty state з CTA, коли немає jobs
- Hero «Метод А проти Б» вгорі (золотий бордюр, Trophy badge, висновок)
- KPI-картки (всього сесій, IN/OUT, баланс)
- Загальний розподіл поведінки (PieChart) і трафік по сесіях (BarChart)
- Таблиця історії з multi-select (2–4 → `RunComparisonModal`)
- Кнопка деталей → `JobDetailModal` (включає `PoseFilterCard` для Approach B і `RecommendationsSection`)

**`/evaluation` Точність** — auto A vs B vs GT:
- Селектор пари відео+GT (зберігається у localStorage)
- Один клік запускає **дві паралельні задачі** (approach=A і approach=B)
- Два live-прогресу
- Після завершення:
  - `ApproachWinnerBars` — 4 метрики (Accuracy, F1 IN, F1 OUT, MAE) з 🏆 біля переможця
  - `SideBySideVideoPlayer` 3-колонне (Pred-A | Pred-B | GT, синхронізовані)
  - Per-метод `EvaluationKPICards` (2×2)
  - Кумулятивний area chart (GT vs Pred-A vs Pred-B)

## VideoPlayer — клавіатурні шорткати

Коли контейнер плеєра має фокус:

| Клавіша | Дія |
|---------|-----|
| Space / K | play / pause |
| ← / → | seek ±5 с |
| J / L | seek ±10 с |
| , / . | покадрово назад/вперед |
| 0–9 | стрибок до 0%, 10%, …, 90% |
| M | mute toggle |
| + / − | швидкість на крок вгору/вниз |

`SideBySideVideoPlayer` має той самий набір (master = перше відео, інші sync через `currentTime`).

## LocalStorage

| Ключ | Призначення |
|------|-------------|
| `viz_config` | VizConfig toggles (Upload) |
| `eval_basename` | Остання вибрана GT-пара (Evaluation) |

## Дизайн-система

CSS-змінні визначено в `index.css`. Семантичні кольори (`--color-in`, `--color-out`, `--behavior-*`) — використовувати їх, не Tailwind класи `text-yellow-400`.

Поведінкові кольори:

| Клас | CSS-змінна | Колір |
|------|------------|-------|
| Foraging (Фуражування) | `--behavior-foraging` | зелений |
| Fanning (Вентиляція) | `--behavior-fanning` | блакитний |
| Washboarding (Полірування) | `--behavior-washboarding` | помаранчевий |
| Defense (Захист) | `--behavior-defense` | червоний |
