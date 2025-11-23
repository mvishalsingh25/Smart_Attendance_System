import streamlit as st
import db as db_module
import utils
from views_dashboard import show_dashboard
from views_monitor import show_monitor
from views_analytics import show_analytics

# --- 1. CONFIG & SETUP ---
st.set_page_config(layout="wide", page_title="VisionGuard AI", page_icon="🛡️")

# Load CSS and DB
utils.load_css()
db_module.init_db()

# Load AI Models
model, encoder, ref_embeddings, ref_names, detector = utils.load_resources()

# Initialize Session State
if 'running' not in st.session_state: st.session_state.running = False
if 'recent_logs' not in st.session_state: st.session_state.recent_logs = []

# --- 2. SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="margin-bottom: 0px; font-size: 2.5rem;">VisionGuard</h1>
            <p style="font-style: italic; color: #808080; margin-top: 5px; font-size: 1rem;">
                Smart Attendance System
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    nav = st.radio("Menu", ["Dashboard", "Live Monitor", "Analytics & Reports"], label_visibility="collapsed")
    st.markdown("---")
    
    st.caption("System Status")
    status_placeholder = st.empty()
    if st.session_state.running:
        status_placeholder.markdown('<div class="status-pill online">● Monitoring Active</div>', unsafe_allow_html=True)
    else:
        status_placeholder.markdown('<div class="status-pill offline">● System Idle</div>', unsafe_allow_html=True)
        
    with st.expander("⚙️ Calibration"):
        threshold = st.slider("Sensitivity", 0.30, 0.60, 0.45, 0.01)

# --- 3. ROUTING ---
if nav == "Dashboard":
    show_dashboard()

elif nav == "Live Monitor":
    # Pass model and threshold to the monitor view
    show_monitor(model, encoder, ref_embeddings, ref_names, detector, threshold)

elif nav == "Analytics & Reports":
    show_analytics()