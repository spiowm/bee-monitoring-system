import logging
import pathlib
import torch
from ultralytics import YOLO
from config import settings

_DEVICE = 0 if torch.cuda.is_available() else "cpu"
_HALF = isinstance(_DEVICE, int)

logger = logging.getLogger(__name__)

class RampDetector:
    """
    Завантажує models/ramp/best.pt один раз при старті.
    detect(frame) → bbox (x1,y1,x2,y2) або None якщо не знайдено.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RampDetector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        logger.info(f"Loading Ramp model from {settings.RAMP_MODEL_PATH}...")
        try:
            if pathlib.Path(settings.RAMP_MODEL_PATH).exists():
                self.model = YOLO(settings.RAMP_MODEL_PATH)
                logger.info("Ramp model loaded.")
            else:
                logger.warning(f"Ramp model not found at {settings.RAMP_MODEL_PATH}")
                self.model = None
        except Exception as e:
            logger.error(f"Failed to load Ramp model: {e}")
            self.model = None
            
        self.cached_bbox = None
        self.frames_since_detect = settings.RAMP_DETECT_INTERVAL
        self._initialized = True

    def detect(self, frame):
        if self.model is None:
            return None
            
        self.frames_since_detect += 1
        
        if self.frames_since_detect >= settings.RAMP_DETECT_INTERVAL:
            results = self.model(frame, verbose=False, device=_DEVICE, half=_HALF)
            if results and len(results[0].boxes) > 0:
                box = results[0].boxes[0].xyxy[0].cpu().numpy()
                self.cached_bbox = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            self.frames_since_detect = 0
            
        return self.cached_bbox
        
    @staticmethod
    def is_in_ramp(bbox_bee, ramp_bbox, padding=10.0):
        if ramp_bbox is None:
            return True
            
        bx1, by1, bx2, by2 = bbox_bee
        rx1, ry1, rx2, ry2 = ramp_bbox
        
        center_x = (bx1 + bx2) / 2.0
        center_y = (by1 + by2) / 2.0
        
        return (rx1 - padding <= center_x <= rx2 + padding and 
                ry1 - padding <= center_y <= ry2 + padding)
