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

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide", page_title="Smart Attendance AI")

# --- Configuration ---
MODEL_DIR = "." 
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
REINSERT_DELAY_SECONDS = 60 * 5 

# --- Database Config ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Vishal@1204",
    "database": "smart_attendance"
}

# --- Load Resources ---
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
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

model, encoder, ref_embeddings, ref_names, detector = load_resources()

# --- Helper Functions ---
def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except:
        return None

def get_todays_attendees(cursor):
    """Fetches list of names present today for the live counter."""
    today = datetime.date.today()
    cursor.execute("SELECT name, time FROM attendance WHERE date = %s ORDER BY time DESC", (today,))
    return cursor.fetchall()

# --- Initialize Session State for Logs ---
if 'recent_logs' not in st.session_state:
    st.session_state.recent_logs = []  # Stores recent marks for the UI table

# --- UI: Sidebar Controls ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/10628/10628970.png", width=80)
    st.title("Control Panel")
    
    # Camera Controls
    start_cam = st.button("▶ Start Camera", type="primary", use_container_width=True)
    stop_cam = st.button("⏹ Stop Camera", type="secondary", use_container_width=True)
    
    if start_cam: st.session_state.running = True
    if stop_cam: st.session_state.running = False

    st.markdown("---")
    st.markdown("### ⚙️ AI Settings")
    # Distance Threshold
    threshold = st.slider("Strictness (Distance)", 0.30, 0.60, 0.45, 0.01, help="Lower is stricter. 0.45 is standard.")

# --- UI: Main Dashboard ---
# Top Metric Row
db = get_db_connection()
if db:
    cursor = db.cursor()
    attendees = get_todays_attendees(cursor)
    count = len(attendees)
    latest_person = attendees[0][0] if count > 0 else "None"
    db.close()
else:
    count = 0
    latest_person = "DB Error"

col_metric1, col_metric2, col_metric3 = st.columns(3)
col_metric1.metric("📅 Date", str(datetime.date.today()))
col_metric2.metric("👥 Total Present", f"{count}")
col_metric3.metric("⚡ Last Marked", latest_person)

st.markdown("---")

# Tabs
tab_live, tab_records, tab_analytics = st.tabs(["📸 Live Attendance", "📋 Database Records", "📈 Analytics"])

# ============================================================
# --- TAB 1: LIVE ATTENDANCE ---
# ============================================================
with tab_live:
    col_cam, col_list = st.columns([2, 1])

    with col_cam:
        st.markdown("### Camera Feed")
        feed_placeholder = st.empty()
        
        if not st.session_state.get('running'):
            feed_placeholder.info("Click 'Start Camera' in the sidebar to begin.")

    with col_list:
        st.markdown("### 🕒 Recent Arrivals")
        # We use a dataframe container that updates dynamically
        log_placeholder = st.empty()

    # --- MAIN LOOP ---
    if st.session_state.get('running'):
        db = get_db_connection()
        cursor = db.cursor()
        cap = cv2.VideoCapture(0)
        
        # Optimization
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret: break

            # 1. Detect
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                
                # Default Colors (Unknown)
                color = (0, 0, 255) # Red
                text = "Unknown"

                try:
                    # 2. AI Recognition
                    embedding_objs = DeepFace.represent(img_path=face_roi, model_name="Facenet512", enforce_detection=False, detector_backend="skip")
                    embedding = embedding_objs[0]["embedding"]
                    
                    test_input = np.array(embedding).reshape(1, -1)
                    prediction_idx = model.predict(test_input)[0]
                    name = encoder.inverse_transform([prediction_idx])[0]
                    
                    # 3. Verification
                    indices = [i for i, n in enumerate(ref_names) if n == name]
                    person_embeddings = [ref_embeddings[i] for i in indices]
                    
                    dists = [cosine(embedding, emb) for emb in person_embeddings]
                    min_dist = min(dists) if dists else 1.0

                    if min_dist < threshold:
                        # --- LOGIC FIX: CHECK STATUS ONLY ---
                        today = datetime.date.today()
                        cursor.execute("SELECT COUNT(*) FROM attendance WHERE name=%s AND date=%s", (name, today))
                        already_present = cursor.fetchone()[0] > 0
                        
                        if already_present:
                            # User asked: NO LOGS for already marked. Just Yellow Box.
                            color = (0, 255, 255) # Yellow
                            text = f"{name} (Present)"
                        else:
                            # NEW MARKING
                            now = datetime.datetime.now()
                            cursor.execute("INSERT INTO attendance (name, date, time) VALUES (%s, %s, %s)", (name, today, now.time()))
                            db.commit()
                            
                            # Visuals
                            color = (0, 255, 0) # Green
                            text = f"MARKED: {name}"
                            
                            # Toast Notification
                            st.toast(f"✅ Attendance Marked: {name}!", icon="🎉")
                            
                            # Update "Recent Arrivals" Table in Session State
                            time_str = now.strftime("%I:%M:%S %p")
                            st.session_state.recent_logs.insert(0, {"Name": name, "Time": time_str})
                            if len(st.session_state.recent_logs) > 10: # Keep last 10
                                st.session_state.recent_logs.pop()

                    else:
                        text = f"Unknown ({min_dist:.2f})"

                except:
                    pass

                # Draw
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Update Feed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feed_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update "Recent Arrivals" Table
            if st.session_state.recent_logs:
                df_recent = pd.DataFrame(st.session_state.recent_logs)
                log_placeholder.dataframe(df_recent, hide_index=True, use_container_width=True)
            else:
                log_placeholder.info("No new attendance yet.")

        cap.release()
        db.close()

# ============================================================
# --- TAB 2: RECORDS ---
# ============================================================
with tab_records:
    st.subheader("Search Database")
    col_filter, col_btn = st.columns([3, 1])
    
    with col_filter:
        option = st.selectbox("Filter By", ["Today", "Last 7 Days", "All Records"])
    
    with col_btn:
        st.write("") # Spacing
        st.write("") 
        load_btn = st.button("Load Data")

    if load_btn:
        db = get_db_connection()
        cursor = db.cursor()
        query = "SELECT name, date, time FROM attendance "
        
        if option == "Today":
            query += "WHERE date = CURDATE() "
        elif option == "Last 7 Days":
            query += "WHERE date >= CURDATE() - INTERVAL 7 DAY "
            
        query += "ORDER BY date DESC, time DESC"
        cursor.execute(query)
        rows = cursor.fetchall()
        db.close()
        
        if rows:
            # Convert timedelta to string
            clean_rows = []
            for r in rows:
                t = r[2]
                if isinstance(t, datetime.timedelta):
                    t = (datetime.datetime.min + t).time()
                clean_rows.append((r[0], r[1], t.strftime("%I:%M:%S %p")))
                
            df = pd.DataFrame(clean_rows, columns=["Name", "Date", "Time"])
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No records found.")

# ============================================================
# --- TAB 3: ANALYTICS ---
# ============================================================
with tab_analytics:
    st.subheader("Attendance Trends")
    
    db = get_db_connection()
    if db:
        cursor = db.cursor()
        cursor.execute("SELECT date, COUNT(DISTINCT name) FROM attendance GROUP BY date ORDER BY date")
        data = cursor.fetchall()
        db.close()
        
        if data:
            df_chart = pd.DataFrame(data, columns=["Date", "Count"])
            
            # Altair Chart
            chart = alt.Chart(df_chart).mark_bar(color='#4F46E5').encode(
                x=alt.X('Date:T', axis=alt.Axis(format="%b %d")),
                y='Count:Q',
                tooltip=['Date', 'Count']
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Not enough data to show charts.")