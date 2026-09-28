import cv2
import time

from ultralytics import YOLO

from config import (
    CAMERA_ID,
    MODEL_NAME,
    CONFIDENCE_THRESHOLD,
    ALERT_COOLDOWN,
    DETECT_CLASSES,
)

from database import (
    create_database,
    save_detection,
)

from alert import save_evidence

print("Starting AI Security Camera...")

# --------------------------------
# CREATE DATABASE
# --------------------------------

create_database()

# --------------------------------
# LOAD AI MODEL
# --------------------------------

print("Loading AI model...")

model = YOLO(MODEL_NAME)

print("AI Model Loaded Successfully")

# --------------------------------
# OPEN CAMERA
# --------------------------------

camera = cv2.VideoCapture(CAMERA_ID)

if not camera.isOpened():
    print("ERROR: Camera could not be opened")
    exit()

print("Camera Started")

# --------------------------------
# ALERT TIMER
# --------------------------------

last_alert_time = 0

# --------------------------------
# MAIN CAMERA LOOP
# --------------------------------

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera")
        break

    # --------------------------------
    # RUN AI DETECTION
    # --------------------------------

    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD
    )

    detection_found = False
    detected_object = None
    detected_confidence = None

    # --------------------------------
    # PROCESS DETECTIONS
    # --------------------------------

    for result in results:
        boxes = result.boxes

        for box in boxes:
            # Get class ID
            class_id = int(box.cls[0])

            # Get class name
            class_name = model.names[class_id]

            # Get confidence
            confidence = float(box.conf[0])

            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # --------------------------------
            # CHECK SECURITY OBJECT
            # --------------------------------

            if class_name in DETECT_CLASSES:
                detection_found = True
                detected_object = class_name
                detected_confidence = confidence

                # Draw detection box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Create label
                label = f"{class_name} {confidence:.2f}"

                # Display label
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # --------------------------------
    # SECURITY ALERT
    # --------------------------------

    if detection_found:
        current_time = time.time()

        # Check cooldown
        if current_time - last_alert_time > ALERT_COOLDOWN:
            print("\n🚨 SECURITY ALERT!")
            print("Object Detected:", detected_object)
            print("Confidence:", detected_confidence)

            # Save evidence image
            image_path, timestamp = save_evidence(frame)

            # Save detection in database
            save_detection(
                detected_object,
                detected_confidence,
                timestamp,
                image_path
            )

            print("Evidence Saved:", image_path)

            # Update alert time
            last_alert_time = current_time

        # Display alert
        cv2.putText(
            frame,
            "SECURITY ALERT!",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # --------------------------------
    # SHOW CAMERA
    # --------------------------------

    cv2.imshow("AI Security Camera", frame)

    # --------------------------------
    # PRESS Q TO EXIT
    # --------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------
# CLOSE PROGRAM
# --------------------------------

camera.release()
cv2.destroyAllWindows()
print("AI Security Camera Closed")
