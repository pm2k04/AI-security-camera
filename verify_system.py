"""
AI Security Camera - System Verification Suite
Validates imports, database creation, YOLO model inference, tracking,
restricted zone logic, evidence capture, analytics queries, and security report generation.
"""

import sys
import os
from pathlib import Path
import numpy as np

def run_tests():
    print("=" * 60)
    print("RUNNING AI SECURITY CAMERA VERIFICATION SUITE")
    print("=" * 60)

    # 1. Imports
    print("\n[TEST 1] Testing Imports...")
    try:
        import cv2
        import numpy as np
        import pandas as pd
        import plotly
        import streamlit
        import PIL
        from ultralytics import YOLO
        import config
        import database
        import camera
        import detector
        import tracker
        import alert
        import email_alert
        import security_report
        import login
        import utils.helpers
        print("  -> All core modules and third-party libraries imported successfully.")
    except Exception as e:
        print(f"  -> FAILED on imports: {e}")
        return False

    # 2. Database Creation & CRUD
    print("\n[TEST 2] Testing SQLite Database...")
    try:
        database.create_database()
        test_id = database.save_detection(
            object_name="test_person",
            confidence=0.92,
            timestamp="2026-09-28 12:00:00",
            image_path=None,
            alert_type="intrusion"
        )
        print(f"  -> Detection saved with record ID #{test_id}")
        
        records = database.get_detections(limit=5)
        assert len(records) > 0, "No records returned"
        print(f"  -> get_detections returned {len(records)} records")

        stats = database.get_detection_statistics()
        print(f"  -> Statistics computed: Total={stats['total_detections']}, Intrusions={stats['intrusion_alerts']}")

        # Clean up test row
        deleted = database.delete_detection(test_id)
        assert deleted, "Failed to delete test row"
        print("  -> Test row cleaned up successfully.")
    except Exception as e:
        print(f"  -> FAILED on database test: {e}")
        return False

    # 3. Model Loading & Detection Engine
    print("\n[TEST 3] Testing YOLO Object Detector...")
    try:
        det = detector.ObjectDetector()
        assert det.is_loaded(), "Model is not loaded"
        print(f"  -> YOLO Model loaded successfully from {det.model_path}")

        # Run synthetic frame detection
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(dummy_frame)
        print(f"  -> Synthetic frame inference passed (detections: {len(results)})")
    except Exception as e:
        print(f"  -> FAILED on detector test: {e}")
        return False

    # 4. Object Tracker
    print("\n[TEST 4] Testing YOLO Tracker...")
    try:
        trk = tracker.ObjectTracker()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracked = trk.track(dummy_frame)
        print(f"  -> Tracking pass completed without error (tracked: {len(tracked)})")
    except Exception as e:
        print(f"  -> FAILED on tracker test: {e}")
        return False

    # 5. Camera & Video Capture
    print("\n[TEST 5] Testing Camera Capture & Fallback...")
    try:
        cam = camera.VideoCamera(source=0)
        started, msg = cam.start()
        print(f"  -> Camera start result: {msg}")
        success, frame, err = cam.read_frame()
        assert success and frame is not None, "Failed to read camera frame"
        assert frame.shape == (480, 640, 3), f"Unexpected frame shape: {frame.shape}"
        cam.release()
        print("  -> Camera frame captured and hardware released properly.")
    except Exception as e:
        print(f"  -> FAILED on camera test: {e}")
        return False

    # 6. Restricted Zone Geometry & Overlays
    print("\n[TEST 6] Testing Restricted Zone Geometry...")
    try:
        zone = (100, 100, 300, 300)
        assert utils.helpers.is_point_in_zone((200, 200), zone) is True, "Center point should be in zone"
        assert utils.helpers.is_point_in_zone((50, 50), zone) is False, "Point (50,50) should be outside zone"
        
        # Test drawing functions
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame = utils.helpers.draw_restricted_zone(test_frame, zone, is_intrusion=True)
        test_frame = utils.helpers.draw_detection_box(test_frame, (120, 120, 220, 220), "person", 0.91, 1, True)
        test_frame = utils.helpers.draw_hud(test_frame, "INTRUSION DETECTED", is_alert=True, fps=30.0)
        print("  -> Zone math and OpenCV overlays executed cleanly.")
    except Exception as e:
        print(f"  -> FAILED on zone/geometry test: {e}")
        return False

    # 7. Alert Manager & Evidence Capture
    print("\n[TEST 7] Testing Evidence Capture & Alert Cooldown...")
    try:
        mgr = alert.AlertManager(cooldown_seconds=2)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 1st trigger: should succeed
        trig1, path1, time1 = mgr.trigger_alert(test_frame, "person", 0.94, alert_type="intrusion")
        assert trig1 is True and path1 is not None and os.path.exists(path1), "Evidence was not saved"
        print(f"  -> Evidence image generated at: {path1}")

        # 2nd trigger immediately: should be suppressed by cooldown
        trig2, path2, time2 = mgr.trigger_alert(test_frame, "person", 0.94, alert_type="intrusion")
        assert trig2 is False, "Cooldown failed to suppress duplicate alert"
        print("  -> Duplicate alert properly suppressed by cooldown.")

        # Clean up test evidence file
        if path1 and os.path.exists(path1):
            os.remove(path1)
            print("  -> Test evidence image cleaned up.")
    except Exception as e:
        print(f"  -> FAILED on alert/evidence test: {e}")
        return False

    # 8. Security Report & CSV Generation
    print("\n[TEST 8] Testing Security Report Generator...")
    try:
        rep_txt = security_report.generate_text_report()
        assert "AI SECURITY CAMERA" in rep_txt, "Report header missing"
        rep_csv = security_report.generate_csv_data()
        assert "object_name" in rep_csv, "CSV header missing"
        print("  -> Text report and CSV export generated successfully.")
    except Exception as e:
        print(f"  -> FAILED on security report test: {e}")
        return False

    # 9. Email Alert Module Diagnostics
    print("\n[TEST 9] Testing Email Module Config Check...")
    try:
        configured = email_alert.is_email_configured()
        print(f"  -> Email configured status: {configured} (Checked without revealing secrets)")
    except Exception as e:
        print(f"  -> FAILED on email module test: {e}")
        return False

    # 10. Login & Authentication
    print("\n[TEST 10] Testing Authentication Logic...")
    try:
        assert login.verify_credentials("admin", "admin123") is True, "Valid credentials failed"
        assert login.verify_credentials("admin", "wrongpassword") is False, "Invalid password passed"
        assert login.verify_credentials("hacker", "admin123") is False, "Invalid username passed"
        print("  -> Authentication verification passed.")
    except Exception as e:
        print(f"  -> FAILED on auth test: {e}")
        return False

    print("\n" + "=" * 60)
    print("ALL 10 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
