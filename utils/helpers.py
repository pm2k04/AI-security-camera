"""
AI Security Camera - Helpers Utility Module
Geometry checks, overlay rendering, restricted zone processing, and formatting utilities.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


def is_point_in_zone(point: Tuple[int, int], zone: Tuple[int, int, int, int]) -> bool:
    """
    Check if a 2D point (x, y) is inside the rectangular zone (x1, y1, x2, y2).
    """
    x, y = point
    zx1, zy1, zx2, zy2 = zone
    min_x, max_x = min(zx1, zx2), max(zx1, zx2)
    min_y, max_y = min(zy1, zy2), max(zy1, zy2)
    return min_x <= x <= max_x and min_y <= y <= max_y


def get_box_center(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    """Calculate the center (x, y) coordinates of a bounding box."""
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def draw_restricted_zone(
    frame: np.ndarray,
    zone: Tuple[int, int, int, int],
    is_intrusion: bool = False,
    label: str = "RESTRICTED ZONE"
) -> np.ndarray:
    """
    Draw a stylish semi-transparent overlay for the restricted zone.
    Red if intrusion detected, yellow/cyan when armed and clear.
    """
    zx1, zy1, zx2, zy2 = zone
    x1, x2 = min(zx1, zx2), max(zx1, zx2)
    y1, y2 = min(zy1, zy2), max(zy1, zy2)

    overlay = frame.copy()
    color = (0, 0, 220) if is_intrusion else (0, 215, 255)  # Red vs Gold/Yellow in BGR
    alpha = 0.25 if is_intrusion else 0.12

    # Draw filled overlay
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw solid border
    border_thickness = 3 if is_intrusion else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, border_thickness)

    # Zone banner
    banner_text = f"⚠️ {label} [INTRUSION]" if is_intrusion else f"🛡️ {label} [ARMED]"
    text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
    
    # Label background
    cv2.rectangle(
        frame,
        (x1, max(0, y1 - 25)),
        (x1 + text_size[0] + 12, max(25, y1)),
        color,
        -1
    )
    cv2.putText(
        frame,
        banner_text,
        (x1 + 6, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame


def draw_detection_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    class_name: str,
    confidence: float,
    track_id: Optional[int] = None,
    is_intruding: bool = False
) -> np.ndarray:
    """
    Draw a clean, high-contrast bounding box and badge for detected objects.
    """
    x1, y1, x2, y2 = box

    # Determine color: Red for intruding, Neon Green for person, Blue for vehicles
    if is_intruding:
        color = (0, 0, 255)
    elif class_name.lower() == "person":
        color = (0, 255, 127)  # Spring green
    else:
        color = (255, 165, 0)  # Bright Cyan/Sky blue in BGR

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Construct label text
    if track_id is not None:
        label = f"{class_name.capitalize()} #{track_id} {confidence:.2f}"
    else:
        label = f"{class_name.capitalize()} {confidence:.2f}"

    if is_intruding:
        label += " [INTRUDER]"

    # Draw center point
    cx, cy = get_box_center(x1, y1, x2, y2)
    cv2.circle(frame, (cx, cy), 4, color, -1)

    # Label badge with background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    badge_y1 = max(0, y1 - text_h - 8)
    badge_y2 = y1
    badge_x1 = x1
    badge_x2 = x1 + text_w + 10

    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
    cv2.putText(
        frame,
        label,
        (badge_x1 + 5, badge_y2 - 5),
        font,
        font_scale,
        (0, 0, 0) if not is_intruding else (255, 255, 255),
        font_thickness,
        cv2.LINE_AA
    )

    return frame


def draw_hud(
    frame: np.ndarray,
    status_text: str = "SYSTEM ONLINE",
    is_alert: bool = False,
    fps: Optional[float] = None,
    people_count: int = 0,
    vehicle_count: int = 0
) -> np.ndarray:
    """
    Draw HUD (Heads-Up Display) status header on the video frame.
    """
    h, w = frame.shape[:2]

    # Top status strip background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Status indicator dot and text
    dot_color = (0, 0, 255) if is_alert else (0, 255, 0)
    cv2.circle(frame, (16, 18), 6, dot_color, -1)
    
    cv2.putText(
        frame,
        status_text,
        (30, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        dot_color,
        2,
        cv2.LINE_AA
    )

    # Metrics on right side
    right_text = f"People: {people_count} | Vehicles: {vehicle_count}"
    if fps is not None and fps > 0:
        right_text += f" | {fps:.1f} FPS"

    (rw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(
        frame,
        right_text,
        (max(10, w - rw - 15), 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    # If intrusion is active, draw large flashing alert banner at bottom
    if is_alert:
        alert_msg = "⚠️ INTRUSION DETECTED ⚠️"
        (aw, ah), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.rectangle(
            frame,
            (int((w - aw) / 2) - 15, h - 50),
            (int((w + aw) / 2) + 15, h - 10),
            (0, 0, 200),
            -1
        )
        cv2.putText(
            frame,
            alert_msg,
            (int((w - aw) / 2), h - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return frame
