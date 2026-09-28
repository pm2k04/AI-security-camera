"""
AI Security Camera - Main Streamlit Dashboard Application
Intelligent real-time security monitoring with YOLOv8, restricted zone intrusion detection,
evidence archiving, analytics, and security reporting.
"""

import os
import time
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Application Module Imports
import config
from database import (
    create_database,
    save_detection,
    get_detections,
    get_today_detections,
    get_detection_statistics,
    delete_detection,
    clear_all_detections
)
from camera import VideoCamera
from tracker import ObjectTracker
from alert import AlertManager
from email_alert import test_email_connection, is_email_configured
from security_report import generate_text_report, generate_csv_data
from login import render_login_screen, logout_user, init_auth_session
from utils.helpers import (
    is_point_in_zone,
    get_box_center,
    draw_restricted_zone,
    draw_detection_box,
    draw_hud
)

# ---------------------------------------------------------
# Page Configuration & Modern Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Security Camera",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Security Dark Theme CSS
st.markdown("""
<style>
    /* Dark Theme Security Accent Styles */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Top Bar and Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetricLabel"] p {
        color: #9ca3af !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] div {
        color: #10b981 !important;
        font-weight: 700 !important;
    }

    /* System Status Badges */
    .status-badge-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-badge-alert {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Modern Card Container */
    .sec-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d121f;
        border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
create_database()
init_auth_session()

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "conf_threshold" not in st.session_state:
    st.session_state.conf_threshold = config.CONFIDENCE_THRESHOLD
if "alert_cooldown" not in st.session_state:
    st.session_state.alert_cooldown = config.ALERT_COOLDOWN
if "zone_x1" not in st.session_state:
    st.session_state.zone_x1 = config.ZONE_X1
if "zone_y1" not in st.session_state:
    st.session_state.zone_y1 = config.ZONE_Y1
if "zone_x2" not in st.session_state:
    st.session_state.zone_x2 = config.ZONE_X2
if "zone_y2" not in st.session_state:
    st.session_state.zone_y2 = config.ZONE_Y2
if "enable_alerts" not in st.session_state:
    st.session_state.enable_alerts = True
if "enable_zone" not in st.session_state:
    st.session_state.enable_zone = True
if "camera_source" not in st.session_state:
    st.session_state.camera_source = str(config.CAMERA_ID)
if "selected_classes" not in st.session_state:
    st.session_state.selected_classes = list(config.DETECT_CLASSES)
if "enable_email" not in st.session_state:
    st.session_state.enable_email = False


# ---------------------------------------------------------
# Authentication Gate
# ---------------------------------------------------------
if not st.session_state.get("authenticated", False):
    render_login_screen()
    st.stop()


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px 0 15px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.8rem;">🛡️</span>
                <div>
                    <h3 style="margin: 0; color: #f9fafb; font-size: 1.15rem; font-weight: 700;">AI Security Cam</h3>
                    <p style="margin: 0; color: #9ca3af; font-size: 0.75rem;">INTELLIGENT SURVEILLANCE</p>
                </div>
            </div>
            <div style="margin-top: 12px;">
                <span class="status-badge-online">● SYSTEM ONLINE</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    menu_selection = st.radio(
        "Navigation",
        [
            "1. Dashboard",
            "2. Live Camera",
            "3. Detection History",
            "4. Evidence Gallery",
            "5. Analytics",
            "6. Security Report",
            "7. Settings",
            "8. Logout"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"""
        <div style="font-size: 0.8rem; color: #6b7280; padding: 4px;">
            Operator: <b style="color: #9ca3af;">{st.session_state.get('user', 'admin')}</b><br>
            Engine: <b style="color: #9ca3af;">YOLOv8n + ByteTrack</b><br>
            Storage: <b style="color: #9ca3af;">SQLite (Encrypted Local)</b>
        </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Logout Action
# ---------------------------------------------------------
if menu_selection == "8. Logout":
    logout_user()


# =========================================================
# 1. DASHBOARD PAGE
# =========================================================
elif menu_selection == "1. Dashboard":
    st.title("🛡️ Security Operations Dashboard")
    st.caption("Real-time telemetry, threat detection metrics, and monitored zone status.")

    # Fetch real SQLite statistics
    stats = get_detection_statistics()

    # Metric Cards Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Detections", stats["total_detections"])
    with m2:
        st.metric("People Detected", stats["people_detected"])
    with m3:
        st.metric("Vehicles Detected", stats["vehicles_detected"])
    with m4:
        st.metric("Intrusion Alerts", stats["intrusion_alerts"])
    with m5:
        st.metric("Today's Events", stats["today_detections"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Analytics Row
    detections = get_detections(limit=500)

    if not detections:
        st.info("ℹ️ No detection data recorded yet. Start the **Live Camera** to begin real-time surveillance and event logging.")
    else:
        df = pd.DataFrame(detections)
        # Parse timestamp
        df["dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = df["dt"].dt.hour

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("📊 Objects Detected Distribution")
            obj_counts = df["object_name"].value_counts().reset_index()
            obj_counts.columns = ["Object", "Count"]
            fig_obj = px.bar(
                obj_counts,
                x="Object",
                y="Count",
                color="Object",
                text="Count",
                color_discrete_sequence=px.colors.qualitative.Prism,
                template="plotly_dark"
            )
            fig_obj.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_obj, use_container_width=True)

        with c2:
            st.subheader("👥 People vs Vehicles")
            # Group into People, Vehicles, and Others
            def categorize(row_obj):
                name = str(row_obj).lower()
                if name == "person":
                    return "People"
                elif name in ["car", "motorcycle", "bus", "truck"]:
                    return "Vehicles"
                return "Others"

            df["category"] = df["object_name"].apply(categorize)
            cat_counts = df["category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]

            fig_pie = px.pie(
                cat_counts,
                values="Count",
                names="Category",
                hole=0.45,
                color="Category",
                color_discrete_map={"People": "#10b981", "Vehicles": "#3b82f6", "Others": "#f59e0b"},
                template="plotly_dark"
            )
            fig_pie.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            st.subheader("🕒 Detections by Hour of Day")
            hourly = df["hour"].value_counts().sort_index().reset_index()
            hourly.columns = ["Hour", "Detections"]
            fig_hour = px.bar(
                hourly,
                x="Hour",
                y="Detections",
                template="plotly_dark",
                color_discrete_sequence=["#6366f1"]
            )
            fig_hour.update_layout(
                xaxis=dict(tickmode="linear", tick0=0, dtick=2, range=[-0.5, 23.5]),
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_hour, use_container_width=True)

        with c4:
            st.subheader("🚨 Alerts Breakdown Over Time")
            alert_counts = df["alert_type"].value_counts().reset_index()
            alert_counts.columns = ["Alert Type", "Count"]
            fig_alert = px.bar(
                alert_counts,
                x="Alert Type",
                y="Count",
                color="Alert Type",
                color_discrete_map={"intrusion": "#ef4444", "normal": "#10b981"},
                template="plotly_dark"
            )
            fig_alert.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_alert, use_container_width=True)

    # Recent Incidents Table Preview
    st.subheader("📋 Recent Security Events")
    recent_events = get_detections(limit=5)
    if recent_events:
        st.dataframe(pd.DataFrame(recent_events)[["id", "timestamp", "object_name", "confidence", "alert_type"]], use_container_width=True)
    else:
        st.caption("No events logged yet.")


# =========================================================
# 2. LIVE CAMERA PAGE
# =========================================================
elif menu_selection == "2. Live Camera":
    st.title("📹 Live Surveillance & Threat Detection")
    st.caption("Real-time AI inference, persistent tracking, and restricted-zone perimeter defense.")

    # Control Bar
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.5, 2, 2])

    with ctrl_col1:
        start_btn = st.button("▶ Start Camera", use_container_width=True, type="primary")
    with ctrl_col2:
        stop_btn = st.button("⏹ Stop Camera", use_container_width=True)
    with ctrl_col3:
        st.session_state.enable_zone = st.checkbox("Restricted Zone Active", value=st.session_state.enable_zone)
    with ctrl_col4:
        st.session_state.enable_alerts = st.checkbox("Intrusion Alerts & Evidence", value=st.session_state.enable_alerts)

    if start_btn:
        st.session_state.camera_running = True
    if stop_btn:
        st.session_state.camera_running = False

    # Dynamic Parameters
    with st.expander("⚙️ Live Inference Parameters", expanded=False):
        p1, p2, p3 = st.columns(3)
        with p1:
            st.session_state.conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.20,
                max_value=0.95,
                value=float(st.session_state.conf_threshold),
                step=0.05
            )
        with p2:
            st.session_state.alert_cooldown = st.slider(
                "Alert Cooldown (Seconds)",
                min_value=3,
                max_value=60,
                value=int(st.session_state.alert_cooldown),
                step=1
            )
        with p3:
            cam_src_in = st.text_input("Camera Source (0 = Default Webcam, or RTSP URL)", value=str(st.session_state.camera_source))
            if cam_src_in != st.session_state.camera_source:
                st.session_state.camera_source = cam_src_in

    # Stream & Status Containers
    status_box = st.empty()
    video_placeholder = st.empty()

    if not st.session_state.camera_running:
        status_box.markdown("""
            <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                <span style="color: #9ca3af;">Status:</span> <b style="color: #6b7280;">CAMERA STANDBY</b> | Click <b>▶ Start Camera</b> to begin live monitoring.
            </div>
        """, unsafe_allow_html=True)
    else:
        # Initialize Camera, Tracker, and Alert Manager
        camera_source = st.session_state.camera_source
        if camera_source.isdigit():
            camera_source = int(camera_source)

        camera = VideoCamera(source=camera_source)
        success, init_msg = camera.start()

        if not success:
            st.error(f"Failed to start camera: {init_msg}")
            st.session_state.camera_running = False
        else:
            status_box.success(f"{init_msg}")

            tracker = ObjectTracker(
                confidence_threshold=st.session_state.conf_threshold,
                detect_classes=st.session_state.selected_classes
            )
            alert_mgr = AlertManager(cooldown_seconds=st.session_state.alert_cooldown)

            zone = (
                int(st.session_state.zone_x1),
                int(st.session_state.zone_y1),
                int(st.session_state.zone_x2),
                int(st.session_state.zone_y2)
            )

            prev_time = time.time()
            fps = 0.0

            try:
                # Main Video Processing Loop
                while st.session_state.get("camera_running", False):
                    frame_ok, frame, err_msg = camera.read_frame()

                    if not frame_ok or frame is None:
                        st.warning(err_msg or "Failed to read camera frame.")
                        break

                    # Compute FPS
                    curr_time = time.time()
                    fps = 1.0 / max(0.001, (curr_time - prev_time))
                    prev_time = curr_time

                    # Run YOLO Multi-Object Tracking
                    tracked = tracker.track(frame, confidence=st.session_state.conf_threshold, persist=True)

                    intrusion_detected = False
                    people_count = 0
                    vehicle_count = 0

                    # Evaluate detections & check restricted zone
                    for obj in tracked:
                        cname = obj["class_name"].lower()
                        cx, cy = obj["center"]
                        box = obj["box"]
                        track_id = obj["track_id"]
                        score = obj["confidence"]

                        if cname == "person":
                            people_count += 1
                        elif cname in ["car", "motorcycle", "bus", "truck"]:
                            vehicle_count += 1

                        # Restricted Zone Intrusion Check
                        is_intruding = False
                        if st.session_state.enable_zone:
                            if cname == "person" and is_point_in_zone((cx, cy), zone):
                                is_intruding = True
                                intrusion_detected = True

                        # Draw bounding box and label badge
                        draw_detection_box(
                            frame,
                            box=box,
                            class_name=cname,
                            confidence=score,
                            track_id=track_id,
                            is_intruding=is_intruding
                        )

                    # Draw Restricted Zone Overlay
                    if st.session_state.enable_zone:
                        draw_restricted_zone(frame, zone=zone, is_intrusion=intrusion_detected)

                    # Trigger Evidence & Alert on Intrusion
                    if intrusion_detected and st.session_state.enable_alerts:
                        triggered, ev_path, ev_time = alert_mgr.trigger_alert(
                            frame=frame,
                            object_name="person",
                            confidence=0.95,
                            alert_type="intrusion",
                            send_email=st.session_state.enable_email
                        )
                        if triggered:
                            st.toast(f"🚨 INTRUSION RECORDED: {Path(ev_path).name}", icon="⚠️")

                    # Draw HUD
                    hud_status = "INTRUSION DETECTED" if intrusion_detected else "CAMERA CONNECTED - NO ALERTS"
                    draw_hud(
                        frame,
                        status_text=hud_status,
                        is_alert=intrusion_detected,
                        fps=fps,
                        people_count=people_count,
                        vehicle_count=vehicle_count
                    )

                    # Convert BGR to RGB for Streamlit rendering
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    # Responsive sleep
                    time.sleep(0.01)

            except Exception as stream_err:
                st.error(f"Live camera stream error: {stream_err}")
            finally:
                camera.release()
                status_box.markdown("""
                    <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 12px 16px;">
                        <span style="color: #9ca3af;">Status:</span> <b style="color: #eab308;">CAMERA STOPPED & RELEASED</b>
                    </div>
                """, unsafe_allow_html=True)


# =========================================================
# 3. DETECTION HISTORY PAGE
# =========================================================
elif menu_selection == "3. Detection History":
    st.title("🗄️ Detection History")
    st.caption("Search, filter, and inspect persisted security events from SQLite.")

    # Filter Bar
    f1, f2, f3, f4, f5 = st.columns([1.5, 1.5, 1.5, 1.5, 1])

    with f1:
        obj_options = ["All"] + list(config.DETECT_CLASSES)
        selected_obj = st.selectbox("Object Filter", obj_options)
    with f2:
        selected_alert = st.selectbox("Alert Type", ["All", "intrusion", "normal"])
    with f3:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)
    with f4:
        date_query = st.text_input("Date (YYYY-MM-DD)", placeholder="e.g. 2026-09-28")
    with f5:
        st.write("")
        st.write("")
        refresh_btn = st.button("🔄 Refresh", use_container_width=True)

    records = get_detections(
        object_filter=selected_obj,
        alert_filter=selected_alert,
        date_filter=date_query.strip() if date_query else None,
        min_confidence=min_conf
    )

    if not records:
        st.info("ℹ️ No detection records found matching the specified filters.")
    else:
        df_records = pd.DataFrame(records)
        st.write(f"Displaying **{len(df_records)}** event records:")
        
        # Display Table
        st.dataframe(
            df_records[["id", "object_name", "confidence", "timestamp", "alert_type", "image_path"]],
            column_config={
                "id": "ID",
                "object_name": "Object",
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                "timestamp": "Timestamp",
                "alert_type": "Alert Type",
                "image_path": "Evidence Path"
            },
            use_container_width=True,
            hide_index=True
        )

        # Inspection & Action Row
        with st.expander("🔍 Inspect or Delete an Event Record"):
            del_c1, del_c2 = st.columns([2, 1])
            with del_c1:
                selected_id = st.number_input("Enter Record ID to Inspect / Delete", min_value=1, step=1)
            with del_c2:
                st.write("")
                st.write("")
                if st.button("🗑️ Delete Record", use_container_width=True):
                    if delete_detection(selected_id):
                        st.success(f"Record #{selected_id} deleted successfully.")
                        st.rerun()
                    else:
                        st.error(f"Record #{selected_id} not found.")

            # Show image if exists
            target_row = next((r for r in records if r["id"] == selected_id), None)
            if target_row and target_row.get("image_path") and os.path.exists(target_row["image_path"]):
                st.image(target_row["image_path"], caption=f"Evidence for Event #{selected_id} ({target_row['timestamp']})", width=500)


# =========================================================
# 4. EVIDENCE GALLERY PAGE
# =========================================================
elif menu_selection == "4. Evidence Gallery":
    st.title("📸 Evidence Gallery")
    st.caption("Visual audit log of snapshots captured during security alarms and restricted-zone intrusions.")

    evidence_dir = Path(config.EVIDENCE_FOLDER)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(list(evidence_dir.glob("*.jpg")) + list(evidence_dir.glob("*.png")), key=os.path.getmtime, reverse=True)

    if not images:
        st.info("ℹ️ No evidence snapshots collected yet. When an intrusion is triggered, images are automatically saved here.")
    else:
        st.write(f"Total Snapshots: **{len(images)}**")

        # Responsive Grid Layout
        cols_per_row = 3
        for i in range(0, len(images), cols_per_row):
            row_images = images[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            for j, img_path in enumerate(row_images):
                with cols[j]:
                    st.markdown(f"""
                        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 10px; margin-bottom: 15px;">
                    """, unsafe_allow_html=True)
                    try:
                        st.image(str(img_path), use_container_width=True)
                        st.caption(f"📁 `{img_path.name}`")
                        
                        # File timestamp
                        mod_time = datetime.fromtimestamp(os.path.getmtime(img_path)).strftime("%Y-%m-%d %H:%M:%S")
                        st.markdown(f"<span style='font-size: 0.75rem; color: #9ca3af;'>Captured: {mod_time}</span>", unsafe_allow_html=True)

                        with open(img_path, "rb") as file:
                            st.download_button(
                                label="⬇️ Download",
                                data=file,
                                file_name=img_path.name,
                                mime="image/jpeg",
                                key=f"dl_{img_path.name}"
                            )
                    except Exception as img_err:
                        st.error(f"Could not load image: {img_err}")
                    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 5. ANALYTICS PAGE
# =========================================================
elif menu_selection == "5. Analytics":
    st.title("📈 Security Analytics & Threat Intelligence")
    st.caption("In-depth analysis derived from authentic SQLite surveillance history.")

    records = get_detections()

    if not records:
        st.info("No detection data available. Connect the live camera to record events.")
    else:
        df = pd.DataFrame(records)
        df["dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = df["dt"].dt.hour
        df["date"] = df["dt"].dt.date

        # 1. Object Detection Bar Chart
        st.subheader("1. Object Detection Distribution")
        obj_df = df["object_name"].value_counts().reset_index()
        obj_df.columns = ["Object", "Detections"]
        fig1 = px.bar(
            obj_df,
            x="Object",
            y="Detections",
            color="Object",
            text="Detections",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Detection Timeline
        st.subheader("2. Detection Timeline Over Days")
        timeline_df = df.groupby(["date", "object_name"]).size().reset_index(name="Count")
        fig2 = px.line(
            timeline_df,
            x="date",
            y="Count",
            color="object_name",
            markers=True,
            template="plotly_dark"
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Date", yaxis_title="Events")
        st.plotly_chart(fig2, use_container_width=True)

        a_col1, a_col2 = st.columns(2)

        with a_col1:
            st.subheader("3. Hourly Activity Heatmap")
            hourly_df = df["hour"].value_counts().sort_index().reset_index()
            hourly_df.columns = ["Hour", "Frequency"]
            fig3 = px.bar(
                hourly_df,
                x="Hour",
                y="Frequency",
                template="plotly_dark",
                color_discrete_sequence=["#10b981"]
            )
            fig3.update_layout(
                xaxis=dict(tickmode="linear", tick0=0, dtick=2),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig3, use_container_width=True)

        with a_col2:
            st.subheader("4. People vs Vehicle Comparison")
            def categorize(val):
                v = str(val).lower()
                return "People" if v == "person" else ("Vehicles" if v in ["car", "motorcycle", "bus", "truck"] else "Other")
            
            df["group"] = df["object_name"].apply(categorize)
            group_df = df["group"].value_counts().reset_index()
            group_df.columns = ["Category", "Total"]
            fig4 = px.pie(
                group_df,
                names="Category",
                values="Total",
                hole=0.4,
                template="plotly_dark",
                color_discrete_sequence=["#34d399", "#60a5fa", "#f59e0b"]
            )
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)

        # 5. Alert Type Breakdown
        st.subheader("5. Security Event Types Breakdown")
        alert_df = df["alert_type"].value_counts().reset_index()
        alert_df.columns = ["Alert Type", "Count"]
        fig5 = px.bar(
            alert_df,
            x="Alert Type",
            y="Count",
            color="Alert Type",
            color_discrete_map={"intrusion": "#ef4444", "normal": "#10b981"},
            template="plotly_dark"
        )
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig5, use_container_width=True)


# =========================================================
# 6. SECURITY REPORT PAGE
# =========================================================
elif menu_selection == "6. Security Report":
    st.title("📄 Security Incident & Audit Report")
    st.caption("Generate, inspect, and export formal security audit reports for compliance and review.")

    stats = get_detection_statistics()

    # Executive Overview Cards
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Total Events Logged", stats["total_detections"])
    with r2:
        st.metric("Intrusion Alerts", stats["intrusion_alerts"])
    with r3:
        st.metric("Average Confidence", f"{stats['avg_confidence']*100:.1f}%")
    with r4:
        st.metric("Top Detected Target", stats["most_frequent_object"])

    st.markdown("---")

    # Download Actions
    down_col1, down_col2 = st.columns(2)
    report_text = generate_text_report()
    csv_data = generate_csv_data()

    with down_col1:
        st.download_button(
            label="📥 Download Formal Audit Report (.TXT)",
            data=report_text,
            file_name=f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with down_col2:
        st.download_button(
            label="📥 Export Full Records (.CSV)",
            data=csv_data,
            file_name=f"detections_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("### 📝 Formal Report Preview")
    st.code(report_text, language="text")


# =========================================================
# 7. SETTINGS PAGE
# =========================================================
elif menu_selection == "7. Settings":
    st.title("⚙️ System Configuration")
    st.caption("Customize camera inputs, detection boundaries, alert thresholds, and email dispatch.")

    with st.form("settings_form"):
        st.subheader("🎥 Camera & AI Parameters")
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            cam_src = st.text_input("Camera Source / Index", value=str(st.session_state.camera_source), help="0 for default webcam, 1 for secondary, or rtsp:// url")
        with c_col2:
            conf_val = st.slider("Detection Confidence Threshold", 0.1, 0.95, float(st.session_state.conf_threshold), 0.05)

        st.subheader("🛡️ Restricted Zone Coordinates (Pixels)")
        z_col1, z_col2, z_col3, z_col4 = st.columns(4)
        with z_col1:
            zx1 = st.number_input("Zone X1 (Left)", min_value=0, max_value=1280, value=int(st.session_state.zone_x1))
        with z_col2:
            zy1 = st.number_input("Zone Y1 (Top)", min_value=0, max_value=720, value=int(st.session_state.zone_y1))
        with z_col3:
            zx2 = st.number_input("Zone X2 (Right)", min_value=0, max_value=1280, value=int(st.session_state.zone_x2))
        with z_col4:
            zy2 = st.number_input("Zone Y2 (Bottom)", min_value=0, max_value=720, value=int(st.session_state.zone_y2))

        st.subheader("🔔 Alerting & Cooldown")
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            cd_val = st.slider("Alert Cooldown (Seconds)", 3, 60, int(st.session_state.alert_cooldown))
        with a_col2:
            email_toggle = st.checkbox("Enable SMTP Email Alerts for Intrusions", value=st.session_state.enable_email)

        st.subheader("🎯 Monitored Classes")
        available_classes = ["person", "car", "motorcycle", "bus", "truck", "bicycle", "dog", "cat"]
        chosen_classes = st.multiselect("Select Target Classes to Monitor", available_classes, default=st.session_state.selected_classes)

        save_settings = st.form_submit_button("💾 Save Settings", use_container_width=True)

        if save_settings:
            st.session_state.camera_source = cam_src
            st.session_state.conf_threshold = conf_val
            st.session_state.zone_x1 = zx1
            st.session_state.zone_y1 = zy1
            st.session_state.zone_x2 = zx2
            st.session_state.zone_y2 = zy2
            st.session_state.alert_cooldown = cd_val
            st.session_state.enable_email = email_toggle
            st.session_state.selected_classes = chosen_classes
            st.success("✅ Settings updated successfully without restarting application.")

    st.markdown("---")

    # Email Diagnostics
    st.subheader("📧 SMTP Email Diagnostics")
    st.write("Verifies connectivity to the configured SMTP email server.")
    if is_email_configured():
        st.info(f"Configured: Sender `{config.EMAIL_SENDER}` via `{config.SMTP_SERVER}:{config.SMTP_PORT}`")
    else:
        st.warning("⚠️ Email alerts not fully configured. Define EMAIL_SENDER, EMAIL_PASSWORD, and EMAIL_RECEIVER in `.env`.")

    if st.button("🧪 Test SMTP Connection"):
        with st.spinner("Testing SMTP connection..."):
            ok, msg = test_email_connection()
            if ok:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")

    st.markdown("---")

    # Maintenance Utility
    st.subheader("🧹 Database Maintenance")
    st.caption("Erase all logged detection events from the local SQLite database.")
    if st.button("⚠️ Clear All Detections Data"):
        clear_all_detections()
        st.warning("Database records cleared.")
        st.rerun()
