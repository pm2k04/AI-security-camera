"""
AI Security Camera - Detector Module
YOLO object detection engine using Ultralytics YOLOv8.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO

from config import (
    MODEL_NAME,
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    DETECT_CLASSES,
)
from utils.helpers import get_box_center


class ObjectDetector:
    """Wrapper around Ultralytics YOLO for targeted object detection."""

    def __init__(
        self,
        model_source: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        detect_classes: Optional[List[str]] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.detect_classes = detect_classes if detect_classes is not None else DETECT_CLASSES

        # Determine model path
        if model_source:
            self.model_path = model_source
        elif Path(MODEL_PATH).exists():
            self.model_path = str(MODEL_PATH)
        else:
            self.model_path = MODEL_NAME

        self.model: Optional[YOLO] = None
        self.load_model()

    def load_model(self) -> bool:
        """Load YOLO model, allowing automatic download if not present locally."""
        try:
            print(f"[Detector] Loading YOLO model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            print("[Detector] YOLO model loaded successfully.")
            return True
        except Exception as e:
            print(f"[Detector] Failed to load YOLO model: {e}")
            self.model = None
            return False

    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self.model is not None

    def detect(self, frame: np.ndarray, confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Run inference on frame.
        Returns a list of detected objects matching configured DETECT_CLASSES:
        [
            {
                "class_name": str,
                "confidence": float,
                "box": (x1, y1, x2, y2),
                "center": (cx, cy)
            },
            ...
        ]
        """
        if self.model is None:
            if not self.load_model():
                return []

        conf = confidence if confidence is not None else self.confidence_threshold
        results = self.model(frame, conf=conf, verbose=False)
        detections: List[Dict[str, Any]] = []

        if not results:
            return detections

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]
                score = float(box.conf[0].item())

                # Filter by targeted classes
                if self.detect_classes and class_name.lower() not in [c.lower() for c in self.detect_classes]:
                    continue

                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in coords]
                cx, cy = get_box_center(x1, y1, x2, y2)

                detections.append({
                    "class_name": class_name,
                    "confidence": round(score, 3),
                    "box": (x1, y1, x2, y2),
                    "center": (cx, cy)
                })

        return detections
