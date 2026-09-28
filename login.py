"""
AI Security Camera - Authentication & Login Module
Manages session authentication, credentials verification, and security login screen.
"""

import streamlit as st
import hashlib
from config import DEMO_USERNAME, DEMO_PASSWORD


def hash_password(password: str) -> str:
    """Helper for password hashing (SHA-256 baseline)."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_credentials(username: str, password: str) -> bool:
    """
    Verify user credentials.
    Structured to allow easy drop-in replacement with hashed database lookups.
    """
    if username.strip() == DEMO_USERNAME and password == DEMO_PASSWORD:
        return True
    return False


def init_auth_session() -> None:
    """Initialize session state authentication keys."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None


def login_user(username: str) -> None:
    """Mark session as authenticated."""
    st.session_state.authenticated = True
    st.session_state.user = username


def logout_user() -> None:
    """Clear authentication session and reset state."""
    st.session_state.authenticated = False
    st.session_state.user = None
    # Reset camera state if running
    if "camera_running" in st.session_state:
        st.session_state.camera_running = False
    st.rerun()


def render_login_screen() -> None:
    """Render a modern, high-tech security login screen."""
    init_auth_session()

    # Center column login card
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 2.5rem; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);">
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <div style="display: inline-block; background: #064e3b; padding: 12px; border-radius: 50%; margin-bottom: 10px;">
                    <span style="font-size: 2rem;">🛡️</span>
                </div>
                <h2 style="color: #f9fafb; margin: 0; font-family: sans-serif; letter-spacing: 0.5px;">AI Security Camera</h2>
                <p style="color: #9ca3af; font-size: 0.9rem; margin-top: 4px;">SURVEILLANCE & INTRUSION MONITORING</p>
                <div style="display: inline-block; background: #065f46; color: #34d399; font-size: 0.75rem; padding: 2px 10px; border-radius: 9999px; font-weight: 600; margin-top: 6px;">
                    SECURE ACCESS PORTAL
                </div>
            </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username_input = st.text_input("Operator Username", placeholder="e.g. admin")
            password_input = st.text_input("Access Key / Password", type="password", placeholder="••••••••")
            submit_button = st.form_submit_button("Authenticate Access", use_container_width=True)

            if submit_button:
                if verify_credentials(username_input, password_input):
                    login_user(username_input.strip())
                    st.success("Access Granted. Redirecting to Security Dashboard...")
                    st.rerun()
                else:
                    st.error("Authentication Failed: Invalid username or password.")

        st.markdown("""
            <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #1f2937; font-size: 0.8rem; color: #6b7280; text-align: center;">
                <p style="margin: 0;"><b>Default Development Credentials:</b></p>
                <p style="margin: 2px 0 0 0;">Username: <code>admin</code> | Password: <code>admin123</code></p>
                <p style="margin-top: 6px; font-size: 0.75rem; color: #ef4444;">Change credentials prior to production deployment.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
