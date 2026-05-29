import supervision as sv


class TrackerFactory:
    """Фабрика трекерів. Зараз підтримується ByteTrack."""

    @staticmethod
    def create(name: str = "bytetrack", config: dict | None = None, **kwargs):
        config = config or {}
        # Параметри підібрані емпірично на датасеті (50 fps, дрібні швидкі об'єкти).
        # Вищий minimum_matching_threshold різко зменшує фрагментацію треків
        # (median lifespan 1 → 16 кадрів), від чого залежить уся класифікація поведінки.
        defaults = dict(
            track_activation_threshold=float(config.get("track_activation_threshold", 0.1)),
            lost_track_buffer=int(config.get("lost_track_buffer", 100)),
            minimum_matching_threshold=float(config.get("minimum_matching_threshold", 0.9)),
            minimum_consecutive_frames=int(config.get("minimum_consecutive_frames", 1)),
            frame_rate=int(config.get("tracker_frame_rate", 50)),
        )
        defaults.update(kwargs)
        return sv.ByteTrack(**defaults)
