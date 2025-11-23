import streamlit as st
import cv2
import numpy as np
import pickle
import datetime
import mysql.connector
import os
import pandas as pd
import altair as alt
from deepface import DeepFace
from scipy.spatial.distance import cosine
import db as db_module  # Assumes you have db.py in the same folder

# ============================================================
# --- 1. PAGE CONFIGURATION ---
# ============================================================
st.set_page_config(layout="wide", page_title="VisionGuard AI", page_icon="🛡️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    div[data-testid="metric-container"] { background-color: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .status-pill { padding: 4px 12px; border-radius: 20px; font-size: 14px; font-weight: 600; text-align: center; display: inline-block;}
    .online { background-color: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .offline { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    
    /* Log Feed Styling */
    .feed-container { height: 450px; overflow-y: auto; border: 1px solid #e5e7eb; border-radius: 12px; padding: 15px; background: #f9fafb; }
    .feed-item { background: white; border-left: 4px solid #6366f1; padding: 12px; margin-bottom: 10px; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); transition: all 0.2s; }
    .feed-user { font-weight: 600; color: #1f2937; font-size: 1.05rem; }
    .feed-time { color: #6b7280; font-size: 0.85rem; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# --- 2. SYSTEM SETUP ---
# ============================================================
MODEL_DIR = "." 
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ✅ AUTOMATIC DB SETUP
db_module.init_db()

if 'running' not in st.session_state: st.session_state.running = False
if 'recent_logs' not in st.session_state: st.session_state.recent_logs = []

@st.cache_resource
def load_resources():
    try:
        with open(os.path.join(MODEL_DIR, "face_model.pkl"), "rb") as f: model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f: encoder = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "embeddings.pkl"), "rb") as f:
            data = pickle.load(f)
            ref_embeddings = data["embeddings"]
            ref_names = data["names"]
        detector = cv2.CascadeClassifier(HAAR_PATH)
        return model, encoder, ref_embeddings, ref_names, detector
    except Exception as e:
        st.error(f"System Error: {e}")
        return None, None, None, None, None

model, encoder, ref_embeddings, ref_names, detector = load_resources()

# ============================================================
# --- 3. SIDEBAR ---
# ============================================================
with st.sidebar:
    # IMPROVED TITLE SECTION
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

# ============================================================
# --- 4. MAIN LOGIC ---
# ============================================================

# Helper to generate HTML for the log list
def get_log_html(logs):
    html = '<div class="feed-container">'
    if logs:
        for log in logs:
            html += f"""
            <div class="feed-item">
                <div class="feed-user">{log['Name']}</div>
                <div class="feed-time">Checked in at {log['Time']}</div>
            </div>
            """
    else:
        html += '<div style="text-align:center; color:#9ca3af; padding-top:40%;">Waiting for check-ins...</div>'
    html += "</div>"
    return html

today = datetime.date.today()

# --- DASHBOARD ---
if nav == "Dashboard":
    # (Existing Dashboard Code)
    db = db_module.get_connection()
    present_count = 0
    last_arrival, last_time = "None", "--:--"

    if db:
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(DISTINCT name) FROM attendance WHERE date = %s", (today,))
        present_count = cursor.fetchone()[0]
        cursor.execute("SELECT name, time FROM attendance WHERE date = %s ORDER BY time DESC LIMIT 1", (today,))
        last_row = cursor.fetchone()
        if last_row:
            last_arrival = last_row[0]
            t = last_row[1]
            if isinstance(t, datetime.timedelta): t = (datetime.datetime.min + t).time()
            last_time = t.strftime("%I:%M %p")
        db.close()

    st.subheader("👋 Overview")
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Present Today", present_count)
    with m2: st.metric("Latest Arrival", last_arrival, delta=last_time)
    with m3: st.metric("System Health", "Optimal", delta="Online")

    st.markdown("### 📅 Quick Activity View")
    db = db_module.get_connection()
    if db:
        df = pd.read_sql("SELECT date, COUNT(DISTINCT name) as count FROM attendance GROUP BY date ORDER BY date DESC LIMIT 7", db)
        db.close()
        if not df.empty:
            chart = alt.Chart(df).mark_bar(color="#6366f1", cornerRadiusEnd=4).encode(
                x=alt.X('date:T', axis=alt.Axis(format="%b %d", title="Date")),
                y=alt.Y('count:Q', title="Attendees")
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else: st.info("No recent activity.")

# --- LIVE MONITOR ---
elif nav == "Live Monitor":
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📹 Camera Feed")
        feed_placeholder = st.empty()
        
        # Control Buttons
        b1, b2 = st.columns(2)
        with b1: 
            if st.button("▶ Start Monitoring", type="primary", use_container_width=True): 
                st.session_state.running = True
                st.rerun()
        with b2:
            if st.button("⏹ Stop Monitoring", type="secondary", use_container_width=True): 
                st.session_state.running = False
                st.rerun()
                
        if not st.session_state.running:
            feed_placeholder.markdown("""
                <div style='background:#f3f4f6; height:400px; display:flex; align-items:center; justify-content:center; border-radius:12px; color:#9ca3af;'>
                    <h3>Camera is Offline</h3>
                </div>
            """, unsafe_allow_html=True)

    with c2:
        st.subheader("🕒 Real-time Log")
        # Create a placeholder that we can update from inside the loop
        log_placeholder = st.empty()
        # Initial render of logs
        log_placeholder.markdown(get_log_html(st.session_state.recent_logs), unsafe_allow_html=True)

    # THE MONITORING LOOP
    if st.session_state.running and model:
        db = db_module.get_connection()
        if db:
            cursor = db.cursor()
            cap = cv2.VideoCapture(0) # Standard webcam
            cap.set(3, 640)
            cap.set(4, 480)
            
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret: 
                    st.error("Camera not detected.")
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                
                for (x, y, w, h) in faces:
                    roi = frame[y:y+h, x:x+w]
                    color, text = (200, 200, 200), ""
                    
                    try:
                        # Face Recognition Logic
                        emb = DeepFace.represent(roi, model_name="Facenet512", enforce_detection=False, detector_backend="skip")[0]["embedding"]
                        pred = model.predict([emb])[0]
                        name = encoder.inverse_transform([pred])[0]
                        
                        idxs = [i for i, n in enumerate(ref_names) if n == name]
                        dists = [cosine(emb, ref_embeddings[i]) for i in idxs]
                        min_dist = min(dists) if dists else 1.0
                        
                        if min_dist < threshold:
                            # Check if already marked today
                            cursor.execute("SELECT COUNT(*) FROM attendance WHERE name=%s AND date=%s", (name, today))
                            if cursor.fetchone()[0] > 0:
                                color = (0, 255, 255) # Yellow for already marked
                                text = f"{name} (Done)"
                            else:
                                # NEW ATTENDANCE MARKED
                                now = datetime.datetime.now()
                                cursor.execute("INSERT INTO attendance (name, date, time) VALUES (%s, %s, %s)", (name, today, now.time()))
                                db.commit()
                                
                                color = (0, 255, 0) # Green for success
                                text = f"{name} (Marked!)"
                                
                                # Update Session State Log
                                st.session_state.recent_logs.insert(0, {"Name": name, "Time": now.strftime("%I:%M %p")})
                                if len(st.session_state.recent_logs) > 20: 
                                    st.session_state.recent_logs.pop()
                                
                                # ⚡ CRITICAL: Update Log Placeholder immediately without RERUN
                                log_placeholder.markdown(get_log_html(st.session_state.recent_logs), unsafe_allow_html=True)
                                
                    except Exception as e:
                        print(e)
                        pass
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    if text: 
                        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Update Video Feed
                feed_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            cap.release()
            db.close()

# --- ANALYTICS & REPORTS ---
elif nav == "Analytics & Reports":
    # (Existing Analytics Code)
    st.subheader("📊 Data Analysis")
    col_type, col_date = st.columns([1, 2])
    with col_type:
        report_type = st.selectbox("Select Report Type", ["Today's Activity", "Weekly Report", "Monthly Overview", "Specific Date", "Date Range"])
    
    db = db_module.get_connection()
    if db:
        try:
            df_all = pd.read_sql("SELECT name, date, time FROM attendance", db)
            db.close()
            if not df_all.empty:
                df_all['time'] = df_all['time'].astype(str).str.replace('0 days ', '', regex=False)
                df_all['date'] = pd.to_datetime(df_all['date'])
                
                filtered_df = pd.DataFrame()
                
                if report_type == "Today's Activity":
                    filtered_df = df_all[df_all['date'].dt.date == today]
                    if not filtered_df.empty:
                        filtered_df['hour'] = pd.to_datetime(filtered_df['time'], format='%H:%M:%S').dt.hour
                        chart = alt.Chart(filtered_df.groupby('hour')['name'].count().reset_index()).mark_bar(color='#6366f1').encode(x='hour:O', y='name:Q')
                        st.altair_chart(chart, use_container_width=True)
                
                elif report_type == "Weekly Report":
                    start = pd.to_datetime(today) - pd.Timedelta(days=7)
                    filtered_df = df_all[df_all['date'] >= start]
                    if not filtered_df.empty:
                        chart = alt.Chart(filtered_df.groupby('date')['name'].nunique().reset_index()).mark_bar(color='#10b981').encode(x=alt.X('date:T', axis=alt.Axis(format="%a %d")), y='name:Q')
                        st.altair_chart(chart, use_container_width=True)

                elif report_type == "Monthly Overview":
                    filtered_df = df_all[(df_all['date'].dt.month == today.month) & (df_all['date'].dt.year == today.year)]
                    if not filtered_df.empty:
                        chart = alt.Chart(filtered_df.groupby('date')['name'].nunique().reset_index()).mark_line(point=True, color='#f59e0b').encode(x=alt.X('date:T', axis=alt.Axis(format="%d")), y='name:Q')
                        st.altair_chart(chart, use_container_width=True)

                elif report_type == "Specific Date":
                    with col_date: sel_date = st.date_input("Select Date", today)
                    filtered_df = df_all[df_all['date'].dt.date == sel_date]

                elif report_type == "Date Range":
                    with col_date: d_range = st.date_input("Select Range", [today - datetime.timedelta(days=7), today])
                    if len(d_range) == 2: filtered_df = df_all[(df_all['date'].dt.date >= d_range[0]) & (df_all['date'].dt.date <= d_range[1])]
                        
                if not filtered_df.empty:
                    st.markdown("### 📋 Detailed Records")
                    st.dataframe(filtered_df[['name', 'date', 'time']], use_container_width=True, hide_index=True)
                    st.download_button("Download CSV", filtered_df.to_csv(index=False).encode('utf-8'), "report.csv", "text/csv")
                else: st.warning("No data found for selection.")
        except Exception as e: st.error(f"Data Error: {e}")