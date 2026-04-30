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
            defaults = dict(
                track_activation_threshold=0.2,
                lost_track_buffer=30,
                minimum_matching_threshold=0.5,
                minimum_consecutive_frames=1,
            )
            defaults.update(kwargs)
            return sv.ByteTrack(**defaults)
        elif name == "ocsort":
            if hasattr(sv, "OCSORT"):
                return sv.OCSORT(**kwargs)
            else:
                logger.warning("OCSORT is not available in supervision>=0.23. Falling back to ByteTrack.")
                return sv.ByteTrack(**kwargs)
        else:
            return sv.ByteTrack(**kwargs)
