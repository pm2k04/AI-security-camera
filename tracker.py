"""
AI Security Camera - Tracker Module
Multi-object tracking using Ultralytics YOLO ByteTrack / BoT-SORT persistent tracking.
Provides object class, tracking ID, confidence, and bounding box coordinates.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    MODEL_NAME,
    CONFIDENCE_THRESHOLD,
    DETECT_CLASSES,
)
from utils.helpers import get_box_center


class ObjectTracker:
    """Wrapper around YOLO model.track() for persistent multi-object tracking."""

    def __init__(
        self,
        model_source: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        detect_classes: Optional[List[str]] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.detect_classes = detect_classes if detect_classes is not None else DETECT_CLASSES
        self.model_path = model_source or (str(MODEL_PATH) if MODEL_PATH.exists() else MODEL_NAME)
        self.model: Optional[YOLO] = None
        self.load_model()

    def load_model(self) -> bool:
        """Initialize YOLO model instance."""
        try:
            self.model = YOLO(self.model_path)
            return True
        except Exception as e:
            print(f"[Tracker] Error loading YOLO model: {e}")
            self.model = None
            return False

    def track(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
        persist: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run tracking on a single frame.
        Returns a list of tracked objects:
        [
            {
                "track_id": Optional[int],
                "class_name": str,
                "confidence": float,
                "box": (x1, y1, x2, y2),
                "center": (cx, cy)
            },
            ...
        ]
        """
        if self.model is None and not self.load_model():
            return []

        conf = confidence if confidence is not None else self.confidence_threshold

        try:
            # model.track with persist=True preserves IDs across frames
            results = self.model.track(
                frame,
                conf=conf,
                persist=persist,
                verbose=False
            )
        except Exception as e:
            # Fallback to standard inference if tracking backend encounters an issue
            print(f"[Tracker] Tracking exception, falling back to detect: {e}")
            results = self.model(frame, conf=conf, verbose=False)

        tracked_objects: List[Dict[str, Any]] = []

        if not results:
            return tracked_objects

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]
                score = float(box.conf[0].item())

                # Filter target classes
                if self.detect_classes and class_name.lower() not in [c.lower() for c in self.detect_classes]:
                    continue

                # Retrieve persistent tracking ID if available
                track_id: Optional[int] = None
                if box.id is not None:
                    try:
                        track_id = int(box.id[0].item())
                    except Exception:
                        track_id = None

                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in coords]
                cx, cy = get_box_center(x1, y1, x2, y2)

                tracked_objects.append({
                    "track_id": track_id,
                    "class_name": class_name,
                    "confidence": round(score, 3),
                    "box": (x1, y1, x2, y2),
                    "center": (cx, cy)
                })

        return tracked_objects
