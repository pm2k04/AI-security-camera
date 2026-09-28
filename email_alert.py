"""
AI Security Camera - Email Alert Module
Dispatches SMTP email notifications with evidence image attachments upon intrusion alerts.
"""

import smtplib
import os
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from typing import Optional, Tuple
from config import (
    EMAIL_SENDER,
    EMAIL_PASSWORD,
    EMAIL_RECEIVER,
    SMTP_SERVER,
    SMTP_PORT
)


def is_email_configured() -> bool:
    """Check if all necessary email credentials are provided in environment."""
    return bool(EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER and SMTP_SERVER and SMTP_PORT)


def send_intrusion_email(
    object_name: str,
    confidence: float,
    timestamp: str,
    image_path: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Send an intrusion alert email with the evidence image attached.
    Does not crash on network or authentication failure.
    """
    if not is_email_configured():
        return False, "Email alerts are not configured. Check .env settings."

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = "AI Security Camera - Intrusion Alert"

        # Email body
        body_text = f"""
SECURITY ALERT: RESTRICTED ZONE INTRUSION DETECTED

Timestamp: {timestamp}
Detected Object: {object_name.capitalize()}
Confidence Score: {confidence:.2%}
Status: CRITICAL

An unauthorized object or individual entered the monitored restricted zone.
Please review the attached evidence snapshot immediately.

---
AI Security Camera Automated Monitoring System
"""
        msg.attach(MIMEText(body_text, "plain"))

        # Attach snapshot if present
        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, "rb") as img_file:
                    img_data = img_file.read()
                    image_attachment = MIMEImage(img_data, name=Path(image_path).name)
                    msg.attach(image_attachment)
            except Exception as img_err:
                print(f"[EmailAlert] Failed to attach image {image_path}: {img_err}")

        # Connect and send via SSL or STARTTLS
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        else:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

        return True, "Alert email sent successfully."

    except Exception as e:
        err_msg = f"Failed to send email alert: {str(e)}"
        print(f"[EmailAlert] {err_msg}")
        return False, err_msg


def test_email_connection() -> Tuple[bool, str]:
    """
    Test SMTP server credentials and connectivity.
    """
    if not is_email_configured():
        return False, "Missing credentials. Please check EMAIL_SENDER, EMAIL_PASSWORD, and EMAIL_RECEIVER in .env."

    try:
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=8) as server:
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        else:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=8) as server:
                server.ehlo()
                server.starttls()
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)

        return True, "SMTP connection and credentials verified successfully!"
    except Exception as e:
        return False, f"SMTP Connection failed: {str(e)}"
