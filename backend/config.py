from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    MONGO_URI: str              # обов'язково, з .env
    DB_NAME: str = "buzz_buzz_buzz"
    MODEL_PATH: str = str(BASE_DIR / "data" / "models" / "bee_pose" / "best.pt")
    RAMP_MODEL_PATH: str = str(BASE_DIR / "data" / "models" / "ramp_detector" / "best.pt")
    CORS_ORIGINS: str = "http://localhost:5173"
    MAX_VIDEO_SIZE_MB: int = 500
    OUTPUT_DIR: str = str(BASE_DIR / "data" / "videos" / "processed")
    RAMP_DETECT_INTERVAL: int = 60  # кадрів між запусками ramp детектора

    class Config:
        env_file = str(BASE_DIR / ".env")
        extra = "ignore"

settings = Settings()
