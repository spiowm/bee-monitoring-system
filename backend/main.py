import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config import settings, BASE_DIR
from db.mongodb import connect_to_mongo, close_mongo_connection
from routers import jobs, analytics
from services.video_processor import get_bee_model, list_available_models
from services.ramp_detector import RampDetector

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    await connect_to_mongo()
    
    # Check GPU
    import torch
    if torch.cuda.is_available():
        logger.info(f"🟩 YOLO GPU Support Activated! Detected device: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("🟨 YOLO GPU Support False. Falling back to CPU. Ensure CUDA is installed.")
        
    # Initialize Models singletons
    logger.info("Initializing models...")
    _ = await get_bee_model()
    _ = RampDetector() # Instantiates singleton
    
    yield
    
    # Shutdown logic
    await close_mongo_connection()

app = FastAPI(title="Bee Monitor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
app.mount("/static/output", StaticFiles(directory=settings.OUTPUT_DIR), name="output")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

app.include_router(jobs.router)
app.include_router(analytics.router)

@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}

from schemas.schemas import ModelInfo

@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    return list_available_models()

# ── Serve built React frontend (single-origin Docker / Lightning AI) ────────
# Only active when frontend/dist/ exists (i.e. after `npm run build`).
# Falls back gracefully: if dist is missing, only the API is served.
# Must be registered AFTER all API routes so API paths take priority.
from fastapi.responses import FileResponse as _FileResponse

# Docker: COPY backend/ → /app/, frontend/dist → /app/frontend/dist (BASE_DIR/frontend/dist)
# Local dev: frontend is at project_root/frontend/dist (BASE_DIR.parent/frontend/dist)
_frontend_dist = BASE_DIR / "frontend" / "dist"
if not _frontend_dist.exists():
    _frontend_dist = BASE_DIR.parent / "frontend" / "dist"

if (_frontend_dist / "assets").exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(_frontend_dist / "assets")),
        name="fe-assets",
    )

@app.get("/{_path:path}", include_in_schema=False)
async def _spa(_path: str):
    index = _frontend_dist / "index.html"
    if index.exists():
        return _FileResponse(str(index))
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Frontend not built — run: npm run build")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
