"""
AI Security Camera - Alert & Evidence Module
Handles intrusion alerts, cooldown logic, evidence image capturing, and database persistence.
"""

import cv2
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from config import EVIDENCE_FOLDER, ALERT_COOLDOWN
from database import save_detection


class AlertManager:
    """Manages alert triggering, evidence capture, cooldown, and dispatching."""

    def __init__(self, cooldown_seconds: int = ALERT_COOLDOWN):
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time: float = 0.0
        self.evidence_folder = Path(EVIDENCE_FOLDER)
        self.evidence_folder.mkdir(parents=True, exist_ok=True)

    def is_cooldown_active(self) -> bool:
        """Return True if cooldown is still in effect."""
        return (time.time() - self.last_alert_time) < self.cooldown_seconds

    def get_remaining_cooldown(self) -> float:
        """Return remaining seconds of cooldown."""
        remaining = self.cooldown_seconds - (time.time() - self.last_alert_time)
        return max(0.0, round(remaining, 1))

    def trigger_alert(
        self,
        frame,
        object_name: str,
        confidence: float,
        alert_type: str = "intrusion",
        send_email: bool = False
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check cooldown; if expired, save evidence image and log detection to database.
        Returns: (triggered: bool, image_path: Optional[str], timestamp: Optional[str])
        """
        current_time = time.time()
        if (current_time - self.last_alert_time) < self.cooldown_seconds:
            # Cooldown active, suppress duplicate alert
            return False, None, None

        # Update last alert timestamp
        self.last_alert_time = current_time

        # Generate timestamp and filename: alert_YYYY-MM-DD_HH-MM-SS.jpg
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"alert_{file_timestamp}.jpg"
        
        self.evidence_folder.mkdir(parents=True, exist_ok=True)
        image_path = str(self.evidence_folder / filename)

        # Save evidence snapshot
        try:
            cv2.imwrite(image_path, frame)
        except Exception as e:
            print(f"[AlertManager] Error saving evidence image: {e}")
            image_path = None

        # Record in SQLite database
        try:
            save_detection(
                object_name=object_name,
                confidence=confidence,
                timestamp=timestamp_str,
                image_path=image_path,
                alert_type=alert_type
            )
        except Exception as e:
            print(f"[AlertManager] Error writing detection to database: {e}")

        # Optional email notification
        if send_email and image_path:
            try:
                from email_alert import send_intrusion_email
                send_intrusion_email(
                    object_name=object_name,
                    confidence=confidence,
                    timestamp=timestamp_str,
                    image_path=image_path
                )
            except Exception as e:
                print(f"[AlertManager] Error dispatching email alert: {e}")

        return True, image_path, timestamp_str


# Global default instance
default_alert_manager = AlertManager()


def save_evidence(frame) -> Tuple[str, str]:
    """
    Direct evidence saving helper function matching original signature.
    Saves frame to EVIDENCE_FOLDER as alert_YYYY-MM-DD_HH-MM-SS.jpg
    """
    folder = Path(EVIDENCE_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)
    timestamp_file = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamp_db = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f"alert_{timestamp_file}.jpg"
    image_path = str(folder / filename)
    cv2.imwrite(image_path, frame)
    return image_path, timestamp_db
