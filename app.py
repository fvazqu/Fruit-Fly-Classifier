import streamlit as st
import cv2
import numpy as np
import os
import sqlite3
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# --- DATABASE & FOLDER SETUP ---
DB_PATH = "fly_data.db"
UPLOAD_DIR = "stored_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     AUTOINCREMENT,
                     timestamp
                     TEXT,
                     filename
                     TEXT,
                     total
                     INTEGER,
                     immature
                     INTEGER,
                     mature
                     INTEGER
                 )''')
    conn.commit()
    conn.close()


init_db()

# --- APP CONFIG & MODEL ---
st.set_page_config(page_title="USDA Fly Counter", layout="wide")
st.title("🪰 Fruit Fly Management System")


@st.cache_resource()
def load_yolo_model():
    model_path = r"model_path"
    return YOLO(model_path)


model = load_yolo_model()

# --- STEP 4: CREATE TABS ---
tab1, tab2 = st.tabs(["🚀 New Analysis", "📊 History & Analytics"])

# ==========================================
# TAB 1: NEW ANALYSIS
# ==========================================
with tab1:
    uploaded_file = st.file_uploader("Upload pupae image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        with st.spinner('Analyzing and Saving...'):
            results = model.predict(source=opencv_image, conf=0.5)
            res = results[0]

            classes = res.boxes.cls.cpu().numpy()
            total = len(classes)
            immature = int(np.count_nonzero(classes == 0))
            mature = int(np.count_nonzero(classes == 1))
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            img_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
            save_path = os.path.join(UPLOAD_DIR, img_filename)
            cv2.imwrite(save_path, opencv_image)

            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO detections (timestamp, filename, total, immature, mature) VALUES (?, ?, ?, ?, ?)",
                      (timestamp, img_filename, total, immature, mature))
            conn.commit()
            conn.close()

        col_img, col_metrics = st.columns([2, 1])
        with col_img:
            st.image(cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB), caption="Detection Results", use_container_width=True)
        with col_metrics:
            st.subheader("Current Results")
            st.metric("Total", total)
            st.metric("Immature (0)", immature)
            st.metric("Mature (1)", mature)
            st.success(f"Data logged successfully.")

# ==========================================
# TAB 2: HISTORY & ANALYTICS
# ==========================================
with tab2:
    st.header("Detection History")

    # Load data from SQLite
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM detections ORDER BY id DESC", conn)
    conn.close()

    if not df.empty:
        # Show Summary Stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Uploads", len(df))
        c2.metric("Total Flies (All Time)", df['total'].sum())
        c3.metric("Avg Mature/Img", f"{df['mature'].mean():.1f}")

        # Show Interactive Table
        st.dataframe(df, use_container_width=True)

        # Download Data Button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full History as CSV", data=csv, file_name="fly_history.csv", mime="text/csv")

        # # Optional: Visualization
        # st.subheader("Maturity Trends")
        # st.bar_chart(df.set_index('timestamp')[['immature', 'mature']])
    else:
        st.info("No history found. Run an analysis to populate the database.")
