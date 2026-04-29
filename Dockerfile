# ── Stage 1: build React frontend ─────────────────────────────────────────
FROM node:20-slim AS frontend

WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci --quiet
COPY frontend/ ./

# Empty VITE_API_URL → falls back to window.location.origin at runtime.
# Works because FastAPI serves both the API and the frontend on the same port.
ARG VITE_API_URL=""
RUN npm run build


# ── Stage 2: GPU runtime (CUDA 12.1 + Python 3.12) ────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PATH="/root/.local/bin:$PATH"

# Python 3.12 (via deadsnakes), ffmpeg, curl
RUN apt-get update -q && \
    apt-get install -y --no-install-recommends software-properties-common curl ffmpeg && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update -q && \
    apt-get install -y --no-install-recommends python3.12 python3.12-venv && \
    # uv — fast Python package manager
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer — only re-runs when pyproject.toml changes)
COPY backend/pyproject.toml backend/uv.lock* ./
RUN uv sync --no-dev

# Application code
COPY backend/ ./

# Built frontend (from stage 1)
COPY --from=frontend /build/dist ./frontend/dist

# Writable directories for uploads / processed videos
RUN mkdir -p data/videos/uploads data/videos/processed data/videos/test

EXPOSE 8000

# Models are expected at data/models/ — mount via volume (see docker-compose.yml)
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
