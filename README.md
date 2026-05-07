# Bee Monitoring System (BuzzTrack)

Комплексна AI-система для комп'ютерного зору та аналізу поведінки бджіл на прилітній дошці вулика. Проєкт створено для автоматизованого моніторингу активності бджолиних сімей, підрахунку трафіку (вліт/виліт) та розпізнавання патернів поведінки.

![Bee Monitoring Preview](https://img.shields.io/badge/Status-Active_Development-success)
![Python 3.12](https://img.shields.io/badge/Python-3.12+-blue.svg)
![React](https://img.shields.io/badge/Frontend-React_Vite-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)

---

## Архітектура проєкту

Система побудована як монорепозиторій і складається з трьох повністю незалежних компонентів. Кожен модуль має власну детальну інструкцію в своєму `README.md`.

```mermaid
graph TD
    User["Користувач (браузер)"]

    subgraph Research ["research/ — ML-навчання"]
        R1["Датасет (Kaggle)"]
        R2["YOLO training + MLflow"]
        R3["best.pt моделі"]
        R1 --> R2 --> R3
    end

    subgraph Backend ["backend/ — FastAPI"]
        B1["POST /jobs (відео)"]
        B2["YOLO детекція + ByteTrack"]
        B3["Підрахунок трафіку (A/B)"]
        B4["Анотоване відео (H.264)"]
        B5[("MongoDB")]
        B1 --> B2 --> B3 --> B4
        B3 -->|"статистика"| B5
    end

    subgraph Frontend ["frontend/ — React + Vite"]
        F1["Завантаження відео"]
        F2["Налаштування пайплайну"]
        F3["Дашборд аналітики"]
    end

    R3 -.->|"моделі"| B2
    User --> F1 & F2 & F3
    F1 -->|"multipart upload"| B1
    F3 <-->|"REST API"| B5
    B4 -->|"/static/output/"| F3

    classDef comp fill:#3b82f6,stroke:#1d4ed8,color:#fff
    classDef io fill:#10b981,stroke:#047857,color:#fff
    classDef db fill:#f59e0b,stroke:#b45309,color:#fff
    class B1,B2,B3,B4,F1,F2,F3,R1,R2,R3 comp
    class User io
    class B5 db
```

1. **`research/` — ML & MLOps середовище**
   - Навчання моделей YOLO-pose для визначення пози бджіл та детектора дошки.
   - Детальна інструкція: [research/README.md](research/README.md).

2. **`backend/` — REST API сервер (FastAPI)**
   - Процесує відео, здійснює трекінг та аналіз поведінки за допомогою YOLO та ByteTrack.
   - Детальна інструкція: [backend/README.md](backend/README.md).

3. **`frontend/` — Веб-інтерфейс (React + TypeScript + Vite)**
   - Дашборд для завантаження відео, налаштування пайплайну та перегляду аналітики та результатів.
   - Детальна інструкція: [frontend/README.md](frontend/README.md).

---

## Швидкий старт (Загальний огляд)

Для роботи системи потрібна запущена **MongoDB**. Система запускається модульно:

1. Запустіть бекенд через UV (`uv run uvicorn...`).
2. Запустіть фронтенд через npm (`npm run dev`).

Більш детальну інструкцію по кожному модулю шукайте в їхніх відповідних `README.md`.

---
*Зроблено для дослідження та автоматизації пасік.*
