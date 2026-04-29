# Frontend — React + TypeScript

Дашборд для завантаження відео, live-моніторингу процесингу та перегляду аналітики.

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

- React 19, TypeScript, Vite
- TailwindCSS (утиліти), CSS variables (дизайн-токени)
- TanStack React Query v5 (серверний стан)
- `@hey-api/openapi-ts` — генерований TypeScript клієнт з OpenAPI схеми бекенда

## Структура

```
src/
├── api/generated/      # Авто-генерований клієнт — не редагувати вручну
├── pages/
│   ├── Upload.tsx      # Завантаження + конфіг + live stats
│   └── Analytics.tsx  # Порівняння підходів + таблиця jobs + відеоплеєр
└── components/
    ├── JobConfigPanel.tsx   # Форма ProcessConfig + VizConfig
    └── LiveStatsPanel.tsx   # Поточний кадр, IN/OUT, активні треки
```

## API клієнт

Генерується з OpenAPI схеми бекенда. Після зміни ендпоінтів/схем:

```bash
# бекенд має бути запущений
npm run generate-api
```

Імпортувати тільки з `./api/generated` — не використовувати fetch/axios напряму.

## Сторінки

**Upload** — основний workflow:
1. Вибір файлу або тестового відео
2. Налаштування `ProcessConfig` (підхід, трекер, пороги) та `VizConfig`
3. Submission → polling `GET /jobs/{id}/live` кожні 2с через React Query (`refetchInterval`)
4. Після завершення — вбудований відеоплеєр

**Analytics** — дослідний вид:
- React Query `useQuery` для summary та compare-approaches
- Таблиця jobs з видаленням (`useMutation` + `invalidateQueries`)
- Модальний відеоплеєр для будь-якого завершеного job
