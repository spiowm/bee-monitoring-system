import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def convert_to_h264(input_path: str, output_path: str):
    """
    Конвертація відео у формат H264 з faststart (для web стрімінгу).
    """
    logger.info(f"Converting video {input_path} to H264...")
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-movflags", "+faststart",
        output_path
    ]
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        _, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"FFmpeg conversion failed: {stderr}")
            raise Exception("FFmpeg conversion failed.")
            
        logger.info(f"Video successfully converted to {output_path}")
        # Optionally remove the original raw OpenCV video file
        try:
            Path(input_path).unlink()
        except:
            pass
            
    except Exception as e:
        logger.error(f"Error during video conversion: {e}")
        raise
