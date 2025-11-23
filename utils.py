import streamlit as st
import pickle
import cv2
import os

# --- CONFIGURATION ---
MODEL_DIR = "."
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def load_css():
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

def get_log_html(logs):
    html = '<div class="feed-container">'
    if logs:
        for log in logs:
            # ✅ FIXED: All in one line guarantees Streamlit renders it as HTML
            html += f'<div class="feed-item"><div class="feed-user">{log["Name"]}</div><div class="feed-time">Checked in at {log["Time"]}</div></div>'
    else:
        html += '<div style="text-align:center; color:#9ca3af; padding-top:40%;">Waiting for check-ins...</div>'
    html += "</div>"
    return html

@st.cache_resource
def load_resources():
    from deepface import DeepFace # Import DeepFace here to build model
    try:
        with open(os.path.join(MODEL_DIR, "face_model.pkl"), "rb") as f: model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f: encoder = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "embeddings.pkl"), "rb") as f:
            data = pickle.load(f)
            ref_embeddings = data["embeddings"]
            ref_names = data["names"]
        detector = cv2.CascadeClassifier(HAAR_PATH)
        
        # --- OPTIMIZATION: Pre-build DeepFace Model ---
        # This prevents the "freeze" when you first click Start Monitoring
        print("Building DeepFace model...")
        _ = DeepFace.build_model("Facenet512")
        print("DeepFace model built.")
        
        return model, encoder, ref_embeddings, ref_names, detector
    except Exception as e:
        st.error(f"System Error loading resources: {e}")
        return None, None, None, None, None