"""
AI Security Camera - Configuration Module
Centralized settings for detection, camera, database, alerts, and authentication.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env if present
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------
# Camera & Video Configuration
# ---------------------------------------------------------
# Default camera index (0 for primary webcam), or RTSP/file path
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# ---------------------------------------------------------
# YOLO AI Model & Detection Configuration
# ---------------------------------------------------------
# Ultralytics YOLO model file
MODEL_NAME = "yolov8n.pt"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = BASE_DIR / MODEL_NAME if (BASE_DIR / MODEL_NAME).exists() else (MODELS_DIR / MODEL_NAME)

# Default confidence threshold
CONFIDENCE_THRESHOLD = 0.50

# Target classes to detect and monitor
DETECT_CLASSES = [
    "person",
    "car",
    "motorcycle",
    "bus",
    "truck"
]

# ---------------------------------------------------------
# Restricted Zone Coordinates (Default rectangle)
# ---------------------------------------------------------
ZONE_X1 = 160
ZONE_Y1 = 120
ZONE_X2 = 480
ZONE_Y2 = 380

# ---------------------------------------------------------
# Alert & Evidence Configuration
# ---------------------------------------------------------
# Cooldown between duplicate alerts in seconds
ALERT_COOLDOWN = 10

# Evidence image storage directory
EVIDENCE_FOLDER = BASE_DIR / "evidence" / "images"

# ---------------------------------------------------------
# Database Configuration
# ---------------------------------------------------------
DATABASE_PATH = BASE_DIR / "database" / "security.db"

# ---------------------------------------------------------
# Demo Authentication Credentials
# ---------------------------------------------------------
# NOTE: Demo credentials for local/testing use. Change before production!
DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin123"

# ---------------------------------------------------------
# Email Alert Configuration (SMTP)
# ---------------------------------------------------------
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
