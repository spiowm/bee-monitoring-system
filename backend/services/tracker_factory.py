import supervision as sv


class TrackerFactory:
    """Фабрика трекерів. Зараз підтримується ByteTrack."""

    @staticmethod
    def create(name: str = "bytetrack", **kwargs):
        defaults = dict(
            track_activation_threshold=0.2,
            lost_track_buffer=60,
            minimum_matching_threshold=0.5,
            minimum_consecutive_frames=1,
        )
        defaults.update(kwargs)
        return sv.ByteTrack(**defaults)
