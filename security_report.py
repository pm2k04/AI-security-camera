"""
AI Security Camera - Security Report Module
Compiles analytical summaries, audit metrics, and downloadable security reports (TXT/CSV).
"""

import io
import csv
from datetime import datetime
from typing import Dict, Any, List
from database import get_detection_statistics, get_detections


def generate_text_report() -> str:
    """
    Generate a clean, structured security incident and audit summary text report.
    """
    stats = get_detection_statistics()
    recent = get_detections(limit=10)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        "=" * 60,
        "             AI SECURITY CAMERA - INCIDENT & AUDIT REPORT",
        "=" * 60,
        f"Generated At: {now_str}",
        f"System Status: ONLINE & ACTIVE",
        "-" * 60,
        "EXECUTIVE SUMMARY METRICS:",
        f"  Total Detections Logged : {stats['total_detections']}",
        f"  People Detected         : {stats['people_detected']}",
        f"  Vehicles Detected       : {stats['vehicles_detected']}",
        f"  Intrusion Alerts        : {stats['intrusion_alerts']}",
        f"  Today's Detections      : {stats['today_detections']}",
        f"  Average Confidence      : {stats['avg_confidence'] * 100:.1f}%",
        f"  Most Frequent Object    : {stats['most_frequent_object']}",
        f"  Evidence Images Stored  : {stats['evidence_count']}",
        "-" * 60,
        "RECENT INCIDENTS / DETECTIONS (Last 10 Events):",
    ]

    if not recent:
        report_lines.append("  [No detection records logged to date]")
    else:
        report_lines.append(f"  {'ID':<5} | {'Timestamp':<20} | {'Object':<12} | {'Confidence':<10} | {'Alert Type'}")
        report_lines.append("  " + "-" * 56)
        for row in recent:
            report_lines.append(
                f"  {row['id']:<5} | {row['timestamp']:<20} | {row['object_name']:<12} | "
                f"{row['confidence'] * 100:>8.1f}% | {row.get('alert_type', 'normal')}"
            )

    report_lines.extend([
        "=" * 60,
        "END OF SECURITY REPORT",
        "=" * 60
    ])

    return "\n".join(report_lines)


def generate_csv_data() -> str:
    """
    Export all detection records as a CSV string.
    """
    records = get_detections()
    output = io.StringIO()
    fieldnames = ["id", "object_name", "confidence", "timestamp", "image_path", "alert_type"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for row in records:
        writer.writerow({
            "id": row["id"],
            "object_name": row["object_name"],
            "confidence": row["confidence"],
            "timestamp": row["timestamp"],
            "image_path": row.get("image_path", ""),
            "alert_type": row.get("alert_type", "normal")
        })

    return output.getvalue()
