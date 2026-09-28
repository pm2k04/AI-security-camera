"""
AI Security Camera - Database Module
SQLite persistence for detections, alerts, timestamps, and evidence images.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from config import DATABASE_PATH


def get_connection() -> sqlite3.Connection:
    """Ensure database directory exists and return a connection."""
    db_file = Path(DATABASE_PATH)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def create_database() -> None:
    """Initialize the SQLite database and create detections table if not present."""
    conn = get_connection()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    image_path TEXT,
                    alert_type TEXT DEFAULT 'normal'
                )
            """)
            # Check for column schema upgrade if alert_type is missing from previous version
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(detections)")
            columns = [col[1] for col in cursor.fetchall()]
            if "alert_type" not in columns:
                cursor.execute("ALTER TABLE detections ADD COLUMN alert_type TEXT DEFAULT 'normal'")
    finally:
        conn.close()


def save_detection(
    object_name: str,
    confidence: float,
    timestamp: str,
    image_path: Optional[str] = None,
    alert_type: str = "normal"
) -> int:
    """
    Save a detection event to the database using parameterized queries.
    Returns the new row ID.
    """
    create_database()
    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO detections (
                    object_name,
                    confidence,
                    timestamp,
                    image_path,
                    alert_type
                ) VALUES (?, ?, ?, ?, ?)
            """, (object_name, float(confidence), timestamp, image_path, alert_type))
            return cursor.lastrowid
    finally:
        conn.close()


def get_detections(
    limit: Optional[int] = None,
    object_filter: Optional[str] = None,
    alert_filter: Optional[str] = None,
    date_filter: Optional[str] = None,
    min_confidence: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve detection records with optional filters.
    """
    create_database()
    conn = get_connection()
    try:
        query = "SELECT id, object_name, confidence, timestamp, image_path, alert_type FROM detections WHERE 1=1"
        params: List[Any] = []

        if object_filter and object_filter.lower() != "all":
            query += " AND LOWER(object_name) = LOWER(?)"
            params.append(object_filter)

        if alert_filter and alert_filter.lower() != "all":
            query += " AND LOWER(alert_type) = LOWER(?)"
            params.append(alert_filter)

        if date_filter:
            # Matches YYYY-MM-DD prefix in timestamp
            query += " AND timestamp LIKE ?"
            params.append(f"{date_filter}%")

        if min_confidence is not None:
            query += " AND confidence >= ?"
            params.append(float(min_confidence))

        query += " ORDER BY id DESC"

        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(int(limit))

        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_today_detections() -> List[Dict[str, Any]]:
    """Retrieve detections recorded today."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    return get_detections(date_filter=today_str)


def get_detection_statistics() -> Dict[str, Any]:
    """
    Calculate summary statistics from the detections table.
    """
    create_database()
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Total detections
        cursor.execute("SELECT COUNT(*) FROM detections")
        total_detections = cursor.fetchone()[0]

        # People detected
        cursor.execute("SELECT COUNT(*) FROM detections WHERE LOWER(object_name) = 'person'")
        people_detected = cursor.fetchone()[0]

        # Vehicles detected (car, motorcycle, bus, truck)
        cursor.execute("""
            SELECT COUNT(*) FROM detections 
            WHERE LOWER(object_name) IN ('car', 'motorcycle', 'bus', 'truck')
        """)
        vehicles_detected = cursor.fetchone()[0]

        # Intrusion alerts
        cursor.execute("SELECT COUNT(*) FROM detections WHERE LOWER(alert_type) = 'intrusion'")
        intrusion_alerts = cursor.fetchone()[0]

        # Today's detections
        today_str = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT COUNT(*) FROM detections WHERE timestamp LIKE ?", (f"{today_str}%",))
        today_detections = cursor.fetchone()[0]

        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM detections")
        avg_conf_row = cursor.fetchone()[0]
        avg_confidence = round(float(avg_conf_row), 3) if avg_conf_row is not None else 0.0

        # Most frequent object
        cursor.execute("""
            SELECT object_name, COUNT(*) as cnt 
            FROM detections 
            GROUP BY object_name 
            ORDER BY cnt DESC 
            LIMIT 1
        """)
        top_obj_row = cursor.fetchone()
        most_frequent_object = top_obj_row[0] if top_obj_row else "None"

        # Evidence count (valid image_path non-null and not empty)
        cursor.execute("SELECT COUNT(*) FROM detections WHERE image_path IS NOT NULL AND image_path != ''")
        evidence_count = cursor.fetchone()[0]

        return {
            "total_detections": total_detections,
            "people_detected": people_detected,
            "vehicles_detected": vehicles_detected,
            "intrusion_alerts": intrusion_alerts,
            "today_detections": today_detections,
            "avg_confidence": avg_confidence,
            "most_frequent_object": most_frequent_object,
            "evidence_count": evidence_count,
        }
    finally:
        conn.close()


def delete_detection(detection_id: int) -> bool:
    """Delete a single detection record by ID."""
    create_database()
    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM detections WHERE id = ?", (int(detection_id),))
            return cursor.rowcount > 0
    finally:
        conn.close()


def clear_all_detections() -> None:
    """Clear all records from detections table (maintenance utility)."""
    create_database()
    conn = get_connection()
    try:
        with conn:
            conn.execute("DELETE FROM detections")
    finally:
        conn.close()
