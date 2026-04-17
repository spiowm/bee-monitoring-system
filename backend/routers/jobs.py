from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pathlib import Path
import uuid
import os
import json
from datetime import datetime
from schemas.schemas import ProcessConfig, VizConfig, JobCreateResponse
from services.video_processor import process_video
from db.mongodb import get_db

router = APIRouter(prefix="/jobs", tags=["Jobs"])

@router.post("", response_model=JobCreateResponse)
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    config: str = Form(...),
    viz_config: str = Form(...)
):
    try:
        cfg = json.loads(config)
        v_cfg = json.loads(viz_config)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in config or viz_config")

    from config import BASE_DIR
    job_id = str(uuid.uuid4())
    upload_dir = str(BASE_DIR / "data" / "videos" / "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = f"{upload_dir}/{job_id}_{video.filename}"
    with open(file_path, "wb") as f:
        f.write(await video.read())

    db = get_db()
    
    job_doc = {
        "job_id": job_id,
        "filename": video.filename,
        "created_at": datetime.utcnow(),
        "status": "pending",
        "progress": 0.0,
        "config": cfg,
        "viz_config": v_cfg,
        "live_stats": {},
        "result": None,
        "error": None
    }
    await db["jobs"].insert_one(job_doc)

    background_tasks.add_task(process_video, job_id, file_path, cfg, v_cfg)
    
    return {"job_id": job_id, "status": "pending"}

@router.get("/{job_id}")
async def get_job(job_id: str):
    db = get_db()
    job = await db["jobs"].find_one({"job_id": job_id}, {"_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

from pydantic import BaseModel
class TestJobRequest(BaseModel):
    filename: str
    config: ProcessConfig
    viz_config: VizConfig

@router.post("/test", response_model=JobCreateResponse)
async def create_test_job(
    request: TestJobRequest,
    background_tasks: BackgroundTasks
):
    from config import BASE_DIR
    import shutil
    
    source_path = BASE_DIR / "data" / "videos" / "test" / request.filename
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Test video not found")
        
    job_id = str(uuid.uuid4())
    upload_dir = str(BASE_DIR / "data" / "videos" / "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = f"{upload_dir}/{job_id}_{request.filename}"
    shutil.copy2(source_path, file_path)
    
    db = get_db()
    
    job_doc = {
        "job_id": job_id,
        "filename": request.filename,
        "created_at": datetime.utcnow(),
        "status": "pending",
        "progress": 0.0,
        "config": request.config.model_dump(),
        "viz_config": request.viz_config.model_dump(),
        "live_stats": {},
        "result": None,
        "error": None
    }
    await db["jobs"].insert_one(job_doc)
    
    background_tasks.add_task(process_video, job_id, file_path, request.config.model_dump(), request.viz_config.model_dump())
    
    return {"job_id": job_id, "status": "pending"}

@router.get("/test/videos")
async def list_test_videos():
    from config import BASE_DIR
    import os
    test_dir = BASE_DIR / "data" / "videos" / "test"
    if not test_dir.exists():
        return []
    videos = [f for f in os.listdir(test_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    return videos

@router.delete("/{job_id}")
async def delete_job(job_id: str):
    db = get_db()
    job = await db["jobs"].find_one({"job_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    await db["jobs"].delete_one({"job_id": job_id})
    
    import os
    from config import settings
    output_dir = Path(settings.OUTPUT_DIR)
    
    raw_file = output_dir / f"{job_id}_raw.mp4"
    processed_file = output_dir / f"{job_id}.mp4"
    
    if raw_file.exists():
        try: os.remove(raw_file)
        except Exception: pass
    if processed_file.exists():
        try: os.remove(processed_file)
        except Exception: pass
        
    return {"status": "deleted"}

@router.get("/{job_id}/live")
async def get_job_live_stats(job_id: str):
    db = get_db()
    job = await db["jobs"].find_one({"job_id": job_id}, {"live_stats": 1, "status": 1, "progress": 1, "_id": 0})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("")
async def list_jobs():
    db = get_db()
    jobs_cursor = db["jobs"].find({}, {"_id": 0, "events": 0}).sort("created_at", -1).limit(20)
    jobs = await jobs_cursor.to_list(length=20)
    return jobs
