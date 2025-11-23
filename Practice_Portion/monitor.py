# import streamlit as st
# import cv2
# import datetime
# from deepface import DeepFace
# from scipy.spatial.distance import cosine
# import db as db_module
# from utils import get_log_html

# def show_monitor(model, encoder, ref_embeddings, ref_names, detector, threshold):
#     today = datetime.date.today()
#     c1, c2 = st.columns([2, 1])
    
#     with c1:
#         st.subheader("📹 Camera Feed")
#         feed_placeholder = st.empty()
        
#         # Control Buttons
#         b1, b2 = st.columns(2)
#         with b1: 
#             if st.button("▶ Start Monitoring", type="primary", use_container_width=True): 
#                 st.session_state.running = True
#                 st.rerun()
#         with b2:
#             if st.button("⏹ Stop Monitoring", type="secondary", use_container_width=True): 
#                 st.session_state.running = False
#                 st.rerun()
                
#         if not st.session_state.running:
#             feed_placeholder.markdown("""
#                 <div style='background:#f3f4f6; height:400px; display:flex; align-items:center; justify-content:center; border-radius:12px; color:#9ca3af;'>
#                     <h3>Camera is Offline</h3>
#                 </div>
#             """, unsafe_allow_html=True)

#     with c2:
#         st.subheader("🕒 Real-time Log")
#         log_placeholder = st.empty()
#         log_placeholder.markdown(get_log_html(st.session_state.recent_logs), unsafe_allow_html=True)

#     # THE MONITORING LOOP
#     if st.session_state.running and model:
#         db = db_module.get_connection()
#         if db:
#             cursor = db.cursor()
#             cap = cv2.VideoCapture(0)
#             cap.set(3, 640)
#             cap.set(4, 480)
            
#             while st.session_state.running:
#                 ret, frame = cap.read()
#                 if not ret: 
#                     st.error("Camera not detected.")
#                     break
                
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                
#                 for (x, y, w, h) in faces:
#                     roi = frame[y:y+h, x:x+w]
#                     color, text = (200, 200, 200), ""
                    
#                     try:
#                         emb = DeepFace.represent(roi, model_name="Facenet512", enforce_detection=False, detector_backend="skip")[0]["embedding"]
#                         pred = model.predict([emb])[0]
#                         name = encoder.inverse_transform([pred])[0]
                        
#                         idxs = [i for i, n in enumerate(ref_names) if n == name]
#                         dists = [cosine(emb, ref_embeddings[i]) for i in idxs]
#                         min_dist = min(dists) if dists else 1.0
                        
#                         if min_dist < threshold:
#                             # Check if already marked today
#                             cursor.execute("SELECT COUNT(*) FROM attendance WHERE name=%s AND date=%s", (name, today))
#                             if cursor.fetchone()[0] > 0:
#                                 color = (0, 255, 255)
#                                 text = f"{name} (Done)"
#                             else:
#                                 now = datetime.datetime.now()
#                                 cursor.execute("INSERT INTO attendance (name, date, time) VALUES (%s, %s, %s)", (name, today, now.time()))
#                                 db.commit()
                                
#                                 color = (0, 255, 0)
#                                 text = f"{name} (Marked!)"
                                
#                                 st.session_state.recent_logs.insert(0, {"Name": name, "Time": now.strftime("%I:%M %p")})
#                                 if len(st.session_state.recent_logs) > 20: 
#                                     st.session_state.recent_logs.pop()
                                
#                                 log_placeholder.markdown(get_log_html(st.session_state.recent_logs), unsafe_allow_html=True)
                                
#                     except Exception:
#                         pass
                    
#                     cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#                     if text: 
#                         cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
#                 feed_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
#             cap.release()
#             db.close()