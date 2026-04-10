import logging
import supervision as sv

logger = logging.getLogger(__name__)

class TrackerFactory:
    """
    Фабрика трекерів через supervision.
    Підтримувані: "bytetrack", "ocsort".
    Всі трекери supervision мають інтерфейс: tracker.update_with_detections(detections)
    """
    @staticmethod
    def create(name: str, **kwargs):
        name = name.lower()
        if name == "bytetrack":
            return sv.ByteTrack(**kwargs)
        elif name == "ocsort":
            if hasattr(sv, "OCSORT"):
                return sv.OCSORT(**kwargs)
            else:
                logger.warning("OCSORT is not available in supervision>=0.23. Falling back to ByteTrack.")
                return sv.ByteTrack(**kwargs)
        else:
            return sv.ByteTrack(**kwargs)
