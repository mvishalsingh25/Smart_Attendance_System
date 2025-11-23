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

# ============================================================
# --- 1. PAGE CONFIGURATION & MODERN CSS ---
# ============================================================
st.set_page_config(
    layout="wide", 
    page_title="VisionGuard AI", 
    page_icon="👁️",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for "Pro" Dashboard Look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-color: #4F46E5;
    }
    [data-testid="stMetricValue"] {
        font-size: 26px;
        color: #4F46E5;
        font-weight: 700;
    }
    
    /* Event Feed */
    .event-card {
        background-color: #f8fafc;
        border-left: 4px solid #4F46E5;
        padding: 10px;
        margin-bottom: 8px;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .event-name {
        font-size: 1rem;
        font-weight: 600;
        color: #1e293b;
    }
    .event-time {
        font-size: 0.8rem;
        color: #64748b;
    }
    
    /* Status Badges */
    .status-online { color: #10B981; font-weight: bold; border: 1px solid #10B981; padding: 5px 10px; border-radius: 20px; background: rgba(16, 185, 129, 0.1); }
    .status-offline { color: #EF4444; font-weight: bold; border: 1px solid #EF4444; padding: 5px 10px; border-radius: 20px; background: rgba(239, 68, 68, 0.1); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# --- 2. BACKEND SETUP ---
# ============================================================
MODEL_DIR = "." 
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Vishal@1204",
    "database": "smart_attendance"
}

if 'running' not in st.session_state: st.session_state.running = False
if 'recent_logs' not in st.session_state: st.session_state.recent_logs = []

@st.cache_resource
def load_resources():
    try:
        with open(os.path.join(MODEL_DIR, "face_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
            encoder = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "embeddings.pkl"), "rb") as f:
            data = pickle.load(f)
            ref_embeddings = data["embeddings"]
            ref_names = data["names"]
        detector = cv2.CascadeClassifier(HAAR_PATH)
        return model, encoder, ref_embeddings, ref_names, detector
    except Exception as e:
        st.error(f"❌ System Error: {e}")
        return None, None, None, None, None

model, encoder, ref_embeddings, ref_names, detector = load_resources()

def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except:
        return None

# ============================================================
# --- 3. SIDEBAR ---
# ============================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/11667/11667357.png", width=60)
    st.title("VisionGuard AI")
    st.markdown("---")
    
    nav = st.radio("Navigation", ["Dashboard", "Live Camera", "Records & Analytics"], label_visibility="collapsed")
    
    st.markdown("---")
    st.subheader("System Status")
    if st.session_state.running:
        st.markdown('<span class="status-online">● Camera Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-offline">● Camera Offline</span>', unsafe_allow_html=True)
    
    st.markdown("### Settings")
    threshold = st.slider("AI Sensitivity", 0.30, 0.60, 0.45, 0.01, help="Lower = Stricter Security")

# ============================================================
# --- 4. MAIN CONTENT ---
# ============================================================
db = get_db_connection()
today = datetime.date.today()
present_count = 0
last_person_name = "Waiting..."
last_person_time = "--:--"

if db:
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(DISTINCT name) FROM attendance WHERE date = %s", (today,))
    present_count = cursor.fetchone()[0]
    cursor.execute("SELECT name, time FROM attendance WHERE date = %s ORDER BY time DESC LIMIT 1", (today,))
    last_entry = cursor.fetchone()
    if last_entry:
        last_person_name = last_entry[0]
        t = last_entry[1]
        if isinstance(t, datetime.timedelta): t = (datetime.datetime.min + t).time()
        last_person_time = t.strftime("%I:%M %p")
    db.close()

# --- DASHBOARD ---
if nav == "Dashboard":
    st.subheader(f"👋 Welcome, Admin")
    st.markdown(f"Today is **{today.strftime('%A, %B %d, %Y')}**")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("👥 Present Today", present_count, delta="Live")
    with c2: st.metric("⚡ Latest Arrival", last_person_name, delta=last_person_time)
    with c3: st.metric("🤖 AI Model", "FaceNet512", delta="Active" if model else "Error", delta_color="normal")
    with c4: st.metric("🛡️ Threshold", f"{threshold}", delta="Strictness")
    
    st.markdown("---")
    st.info("🚀 Navigate to **Live Camera** to start taking attendance.")

# --- LIVE CAMERA ---
elif nav == "Live Camera":
    col_cam, col_feed = st.columns([2, 1])
    with col_cam:
        st.markdown("### 📷 Surveillance Feed")
        feed_placeholder = st.empty()
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("▶ ACTIVATE CAMERA", type="primary", use_container_width=True): st.session_state.running = True
        with btn_col2:
            if st.button("⏹ STOP CAMERA", type="secondary", use_container_width=True): st.session_state.running = False
                
        if not st.session_state.running:
            feed_placeholder.markdown("""<div style='background:#f1f5f9; border-radius:10px; height:350px; display:flex; align-items:center; justify-content:center; color:#64748b;'><h4>Camera is currently OFF</h4></div>""", unsafe_allow_html=True)

    with col_feed:
        st.markdown("### 🕒 Real-Time Stream")
        log_container = st.container(height=400, border=True)
        if not st.session_state.recent_logs:
            log_container.info("Waiting for arrivals...")
        else:
            for log in st.session_state.recent_logs:
                log_container.markdown(f"""<div class="event-card"><div class="event-name">{log['Name']}</div><div class="event-time">✅ Checked in at {log['Time']}</div></div>""", unsafe_allow_html=True)

    if st.session_state.running and model:
        db = get_db_connection()
        cursor = db.cursor()
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                color, text = (0, 0, 255), ""
                try:
                    embedding = DeepFace.represent(face_roi, model_name="Facenet512", enforce_detection=False, detector_backend="skip")[0]["embedding"]
                    pred_idx = model.predict([embedding])[0]
                    name = encoder.inverse_transform([pred_idx])[0]
                    indices = [i for i, n in enumerate(ref_names) if n == name]
                    dists = [cosine(embedding, ref_embeddings[i]) for i in indices]
                    min_dist = min(dists) if dists else 1.0
                    if min_dist < threshold:
                        cursor.execute("SELECT COUNT(*) FROM attendance WHERE name=%s AND date=%s", (name, today))
                        if cursor.fetchone()[0] > 0:
                            color = (0, 255, 255); text = name 
                        else:
                            now = datetime.datetime.now()
                            cursor.execute("INSERT INTO attendance (name, date, time) VALUES (%s, %s, %s)", (name, today, now.time()))
                            db.commit()
                            color = (0, 255, 0); text = name
                            st.session_state.recent_logs.insert(0, {"Name": name, "Time": now.strftime("%I:%M %p")})
                            if len(st.session_state.recent_logs) > 20: st.session_state.recent_logs.pop()
                            st.rerun()
                except: pass
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                if text: cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            feed_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        cap.release()
        db.close()

# --- RECORDS & ANALYTICS ---
elif nav == "Records & Analytics":
    st.subheader("📊 Database & Insights")
    tab1, tab2 = st.tabs(["🗃️ Search Records", "📈 Analytics Graphs"])
    
    with tab1:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: date_opt = st.selectbox("Date Filter", ["Today", "Last 7 Days", "All Time", "Specific Date"])
        with c2: spec_date = st.date_input("Select Date", today) if date_opt == "Specific Date" else None
        with c3: 
            st.write("") 
            st.write("")
            run_search = st.button("Search", use_container_width=True)
        
        if run_search or date_opt:
            db = get_db_connection()
            if db:
                query = "SELECT name, date, time FROM attendance "
                params = []
                if date_opt == "Today": query += "WHERE date = CURDATE() "
                elif date_opt == "Last 7 Days": query += "WHERE date >= CURDATE() - INTERVAL 7 DAY "
                elif date_opt == "Specific Date" and spec_date: query += "WHERE date = %s "; params.append(spec_date)
                query += "ORDER BY date DESC, time DESC"
                df = pd.read_sql(query, db, params=params)
                db.close()
                if not df.empty:
                    st.dataframe(df, use_container_width=True, column_config={"name": "Student Name", "date": st.column_config.DateColumn("Date", format="MMM DD, YYYY"), "time": "Time"}, hide_index=True)
                    st.download_button("Download CSV", df.to_csv(index=False), "attendance.csv")
                else: st.warning("No records found.")

    with tab2:
        db = get_db_connection()
        if db:
            df_all = pd.read_sql("SELECT date, name FROM attendance", db)
            db.close()
            
            if not df_all.empty:
                df_all['date'] = pd.to_datetime(df_all['date'])
                
                col_graph, col_leader = st.columns([2, 1])
                
                with col_graph:
                    st.markdown("#### 📅 Daily Attendance Trend")
                    daily = df_all.groupby('date')['name'].nunique().reset_index()
                    chart = alt.Chart(daily).mark_area(
                        line={'color':'#4F46E5'},
                        color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='#4F46E5', offset=0), alt.GradientStop(color='white', offset=1)], x1=1, y1=1, x2=1, y2=0)
                    ).encode(x='date:T', y='name:Q', tooltip=['date', 'name']).properties(height=350)
                    st.altair_chart(chart, use_container_width=True)
                
                with col_leader:
                    st.markdown("#### 🏆 Top Attendees")
                    
                    # --- NEW: TIME PERIOD SELECTOR FOR LEADERBOARD ---
                    period = st.selectbox("Period", ["All Time", "This Month", "This Week"])
                    
                    # Filter Logic
                    df_filtered = df_all.copy()
                    curr_date = pd.to_datetime(datetime.date.today())
                    
                    if period == "This Month":
                        df_filtered = df_filtered[
                            (df_filtered['date'].dt.month == curr_date.month) & 
                            (df_filtered['date'].dt.year == curr_date.year)
                        ]
                    elif period == "This Week":
                        # Last 7 days approximation
                        start_date = curr_date - pd.Timedelta(days=7)
                        df_filtered = df_filtered[df_filtered['date'] >= start_date]
                    
                    # Generate Leaderboard
                    if not df_filtered.empty:
                        top = df_filtered['name'].value_counts().reset_index().head(10)
                        top.columns = ['Name', 'Days']
                        
                        chart_top = alt.Chart(top).mark_bar(color='#10B981', cornerRadiusEnd=5).encode(
                            x=alt.X('Days:Q', title="Days Present"),
                            y=alt.Y('Name:N', sort='-x', title=""),
                            tooltip=['Name', 'Days']
                        ).properties(height=350)
                        st.altair_chart(chart_top, use_container_width=True)
                    else:
                        st.info("No data for this period.")