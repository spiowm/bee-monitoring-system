import os
import sys

os.environ["YOLO_VERBOSE"] = "False"

try:
    from services.pipeline_stages import *
    from services.pipeline import VideoPipeline
    from services.track_history import TrackHistory
    from services.behavior import BehaviorAnalyzer, HeuristicBehaviorStrategy
    print("All syntax and imports OK!")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
