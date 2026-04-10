import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config import settings, BASE_DIR
from db.mongodb import connect_to_mongo, close_mongo_connection
from routers import jobs, analytics
from services.video_processor import get_bee_model
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
