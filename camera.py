"""
AI Security Camera - Camera Module
Manages video capture devices, RTSP streams, error recovery, and demo video generation.
"""

import cv2
import time
import numpy as np
from typing import Union, Tuple, Optional
from config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT


class VideoCamera:
    """Manages OpenCV VideoCapture with robust error handling and fallbacks."""

    def __init__(self, source: Union[int, str] = CAMERA_ID, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        self.source = source
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self.demo_mode: bool = False
        self._demo_tick: int = 0

    def start(self) -> Tuple[bool, str]:
        """
        Open the camera device. Returns (success, message).
        Falls back to demo simulation if physical hardware camera cannot be opened.
        """
        self.release()

        # Parse source if given as numeric string
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        try:
            # On Windows, cv2.CAP_DSHOW can provide faster startup for USB webcams
            if isinstance(src, int):
                self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(src)
            else:
                self.cap = cv2.VideoCapture(src)

            if self.cap is not None and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.is_running = True
                self.demo_mode = False
                return True, f"Camera opened successfully (Source: {src})."
            else:
                # Hardware camera unavailable, enable demo mode simulation
                self.demo_mode = True
                self.is_running = True
                return True, f"Hardware camera {src} unavailable. Activated Demo Security Feed simulation."

        except Exception as e:
            self.demo_mode = True
            self.is_running = True
            return True, f"Camera error ({e}). Activated Demo Security Feed simulation."

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
        """
        Read a single frame from the camera stream.
        Returns: (success: bool, frame: Optional[np.ndarray], error_msg: Optional[str])
        """
        if not self.is_running:
            return False, None, "Camera is stopped."

        if self.demo_mode:
            frame = self._generate_demo_frame()
            return True, frame, None

        if self.cap is None or not self.cap.isOpened():
            return False, None, "Camera device is not connected."

        ret, frame = self.cap.read()
        if not ret or frame is None:
            # Fallback to demo mode if camera disconnects mid-stream
            self.demo_mode = True
            frame = self._generate_demo_frame()
            return True, frame, "Hardware stream disconnected, switched to simulated security feed."

        return True, frame, None

    def _generate_demo_frame(self) -> np.ndarray:
        """
        Generates a synthetic, high-tech security test pattern with moving test targets
        so detection, tracking, zone intrusion, and evidence capture can be verified.
        """
        self._demo_tick += 1
        h, w = self.height, self.width
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw tech grid background
        grid_color = (25, 30, 35)
        for x in range(0, w, 40):
            cv2.line(frame, (x, 0), (x, h), grid_color, 1)
        for y in range(0, h, 40):
            cv2.line(frame, (0, y), (w, y), grid_color, 1)

        # Timestamp and watermark
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame,
            f"SIMULATED SECURITY FEED | {ts}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 120, 120),
            1,
            cv2.LINE_AA
        )

        # Moving demo targets
        # Target 1: Simulated person moving back and forth across restricted zone
        cycle = (self._demo_tick * 4) % (w + 100) - 50
        px = int(cycle)
        py = int(220 + 30 * np.sin(self._demo_tick * 0.05))

        if -40 < px < w + 40:
            # Draw synthetic humanoid figure
            # Head
            cv2.circle(frame, (px, py - 35), 14, (180, 180, 200), -1)
            # Torso
            cv2.rectangle(frame, (px - 20, py - 20), (px + 20, py + 40), (140, 140, 160), -1)
            # Legs
            cv2.line(frame, (px - 10, py + 40), (px - 10, py + 80), (100, 100, 120), 4)
            cv2.line(frame, (px + 10, py + 40), (px + 10, py + 80), (100, 100, 120), 4)

        return frame

    def release(self) -> None:
        """Properly release camera hardware resource."""
        self.is_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                print(f"[Camera] Error releasing camera: {e}")
            finally:
                self.cap = None

    def is_opened(self) -> bool:
        """Check if camera is currently active."""
        return self.is_running
