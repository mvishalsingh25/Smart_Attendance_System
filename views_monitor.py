import streamlit as st
import cv2
import datetime
import threading  # ✅ NEW: Allows background tasks
from scipy.spatial.distance import cosine
import db as db_module
from utils import get_log_html

# --- HELPER: Save to DB in Background ---
# This runs in a separate thread so the video never freezes
def save_attendance_background(name, current_date, current_time):
    try:
        db = db_module.get_connection()
        if db:
            cursor = db.cursor()
            # Check for duplicates one last time safely
            cursor.execute("SELECT COUNT(*) FROM attendance WHERE name=%s AND date=%s", (name, current_date))
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO attendance (name, date, time) VALUES (%s, %s, %s)", (name, current_date, current_time))
                db.commit()
            db.close()
            print(f"✅ Saved {name} to DB in background.")
    except Exception as e:
        print(f"❌ Background DB Error: {e}")

def show_monitor(model, encoder, ref_embeddings, ref_names, detector, threshold):
    # ✅ Lazy Import (Speeds up App Startup)
    from deepface import DeepFace

    today = datetime.date.today()
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📹 Camera Feed")
        feed_placeholder = st.empty()
        
        # Control Buttons
        b1, b2 = st.columns(2)
        with b1: 
            if st.button("▶ Start Monitoring", type="primary", width="stretch"): 
                st.session_state.running = True
                st.rerun()
        with b2:
            if st.button("⏹ Stop Monitoring", type="secondary", width="stretch"): 
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
        log_placeholder = st.empty()
        log_placeholder.markdown(get_log_html(st.session_state.recent_logs), unsafe_allow_html=True)

    # THE MONITORING LOOP
    if st.session_state.running and model:
        # ✅ OPTIMIZATION: Don't open DB here. We open it only when needed in the thread.
        
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        
        # --- PERFORMANCE VARIABLES ---
        frame_count = 0
        process_every_n_frames = 5 
        scale_factor = 0.5  # ✅ NEW: Process at 50% resolution (4x faster)
        last_faces = []
        
        # ✅ NEW: Local Cache to prevent constant DB hits
        local_attendance_cache = set()
        
        # Pre-load today's attendance into cache
        try:
            db = db_module.get_connection()
            if db:
                cursor = db.cursor()
                cursor.execute("SELECT name FROM attendance WHERE date=%s", (today,))
                for row in cursor.fetchall():
                    local_attendance_cache.add(row[0])
                db.close()
        except Exception as e:
            print(f"Cache Error: {e}")

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret: 
                st.error("Camera not detected.")
                break
            
            frame_count += 1
            
            # --- HEAVY PROCESSING (Every 5th Frame) ---
            if frame_count % process_every_n_frames == 0:
                
                # 1. Resize frame for faster detection
                small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces on SMALL frame
                faces = detector.detectMultiScale(gray_small, 1.1, 5, minSize=(30, 30))
                
                current_faces_data = []
                
                for (x, y, w, h) in faces:
                    # 2. Scale coordinates back UP to original size
                    big_x = int(x / scale_factor)
                    big_y = int(y / scale_factor)
                    big_w = int(w / scale_factor)
                    big_h = int(h / scale_factor)
                    
                    # Crop face from ORIGINAL High-Q frame
                    roi = frame[big_y:big_y+big_h, big_x:big_x+big_w]
                    color, text = (200, 200, 200), ""
                    
                    try:
                        # Recognition
                        emb = DeepFace.represent(roi, model_name="Facenet512", enforce_detection=False, detector_backend="skip")[0]["embedding"]
                        pred = model.predict([emb])[0]
                        name = encoder.inverse_transform([pred])[0]
                        
                        idxs = [i for i, n in enumerate(ref_names) if n == name]
                        dists = [cosine(emb, ref_embeddings[i]) for i in idxs]
                        min_dist = min(dists) if dists else 1.0
                        
                        if min_dist < threshold:
                            # 3. Check Local Cache (Instant)
                            if name in local_attendance_cache:
                                color = (0, 255, 255)
                                text = f"{name} (Done)"
                            else:
                                now = datetime.datetime.now()
                                
                                # 4. Threading: Save to DB in background
                                threading.Thread(target=save_attendance_background, args=(name, today, now.time())).start()
                                
                                # Update Local Cache & UI
                                local_attendance_cache.add(name)
                                color = (0, 255, 0)
                                text = f"{name} (Marked!)"
                                
                                st.session_state.recent_logs.insert(0, {"Name": name, "Time": now.strftime("%I:%M %p")})
                                if len(st.session_state.recent_logs) > 20: 
                                    st.session_state.recent_logs.pop()
                                log_placeholder.markdown(get_log_html(st.session_state.recent_logs), unsafe_allow_html=True)
                        else:
                            color = (0, 0, 255)
                            text = "Unknown"

                        current_faces_data.append((big_x, big_y, big_w, big_h, color, text))
                                
                    except Exception:
                        pass
                
                last_faces = current_faces_data
            
            # --- DRAWING (Every Frame) ---
            for (x, y, w, h, color, text) in last_faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                if text:
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            feed_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", )
        
        cap.release()
            


#             Here is the breakdown of exactly what is different between your previous code and this new code.

# 1. The "Laggy Video" Fix: Frame Skipping
# The Problem: DeepFace (the AI) is very heavy. It takes about 0.1 to 0.3 seconds to recognize a face.

# Previous Code: You forced the computer to run the AI on every single frame.

# Result: If the camera sends 30 frames per second, but the AI takes 0.2s to think, your video slows down to 5 frames per second. It looks like a slideshow.

# This Code: We use Frame Skipping logic.

# Logic: We run the heavy AI on Frame #1. We save the result (name & box location). For Frames #2, #3, #4, and #5, we don't run the AI. We just draw the saved box from Frame #1.

# Result: The heavy math runs 80% less often. The video stays smooth because drawing a box is instant, even if we aren't recalculating the face identity every millisecond.

# Visualizing the Difference:

# Previous: [Calculate] -> [Calculate] -> [Calculate] -> [Calculate] -> [Calculate] (CPU 100%, Laggy)

# This Code: [Calculate] -> [Draw Copy] -> [Draw Copy] -> [Draw Copy] -> [Draw Copy] (CPU 20%, Smooth)

# 2. The "Slow Loading" Fix: Model Pre-building
# The Problem: DeepFace needs to build a massive TensorFlow graph in the background before it can work.

# Previous Code: You loaded the pickle files, but you didn't initialize DeepFace until the loop started.

# Result: When you clicked "Start Monitoring", the button would freeze for 5–10 seconds while the AI woke up, making it feel like the camera was broken.

# This Code: I added DeepFace.build_model("Facenet512") inside the load_resources function in utils.py.

# Result: The heavy lifting happens once when you first open the webpage (cached). When you click "Start Monitoring", the model is already ready in RAM, so the camera opens instantly.