# -------------------------------
# IMPORTS
# -------------------------------
import os
import gdown
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import mysql.connector

# -------------------------------
# 1. SETUP: DOWNLOAD + LOAD MODEL
# -------------------------------
@st.cache_resource
def setup_model():
    files = {
        "binary_pest_model.pkl": "1RykUn9QaRHoj8tLWo5RYB2ZbNeHDIQeE",
        "binary_metadata.pkl": "1Tehx_VfR_RPhaEd5-LZpqZRfALmABFss"
    }

    for file, file_id in files.items():
        if not os.path.exists(file):
            st.info(f"Downloading {file}...")
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                file,
                quiet=False
            )

            # ✅ Ensure file downloaded
            if not os.path.exists(file):
                raise Exception(f"{file} download failed!")

    try:
        model = joblib.load("binary_pest_model.pkl")
        meta = joblib.load("binary_metadata.pkl")
        return model, meta
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None


model, meta = setup_model()

# -------------------------------
# 2. DATABASE CONNECTION (SAFE)
# -------------------------------
def connect_db():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            port=int(os.getenv("DB_PORT", 3306))
        )
        return conn, conn.cursor()
    except:
        st.warning("Database not connected.")
        return None, None

conn, cursor = connect_db()

# -------------------------------
# 3. CONFIG
# -------------------------------
kharif_months = {
    "June": [22, 23, 24, 25, 26],
    "July": [27, 28, 29, 30],
    "August": [31, 32, 33, 34, 35],
    "September": [36, 37, 38, 39],
    "October": [40, 41, 42, 43, 44],
    "November": [45, 46, 47, 48]
}

location_display = {
    0: "Ratnagiri (Chiplun)",
    1: "Raigad (Mahad)",
    2: "Bhandara (Tumsar)",
    3: "Chandrapur (Nagbhid)",
    4: "Kolhapur (Shahuwadi)"
}

def get_location_weather(loc):
    profiles = {
        "Ratnagiri (Chiplun)":  {"maxt": 28.2, "rh1": 94.0, "rf": 38.0},
        "Raigad (Mahad)":       {"maxt": 29.5, "rh1": 91.0, "rf": 32.0},
        "Bhandara (Tumsar)":    {"maxt": 31.0, "rh1": 87.0, "rf": 18.0},
        "Chandrapur (Nagbhid)": {"maxt": 32.8, "rh1": 83.0, "rf": 14.0},
        "Kolhapur (Shahuwadi)": {"maxt": 27.0, "rh1": 86.0, "rf": 22.0}
    }
    return profiles.get(loc, {"maxt": 30, "rh1": 85, "rf": 15})

# -------------------------------
# 4. UI
# -------------------------------
st.set_page_config(page_title="Rice Pest AI", layout="centered")
st.title("🌾 Rice Pest Outbreak Prediction")

if model and meta:

    st.subheader("Input Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.selectbox("Crop", ["Rice"], disabled=True)
        selected_loc = st.selectbox("Location", list(location_display.values()))
        year = st.selectbox("Year", [2025, 2026])

    with col2:
        month = st.selectbox("Month", list(kharif_months.keys()))
        week = st.selectbox("Week", kharif_months[month])
        date_est = datetime(year, 1, 1) + timedelta(weeks=week - 1)
        st.info(f"Target Week: {date_est.strftime('%d %B %Y')}")

    # -------------------------------
    # 5. PREDICTION
    # -------------------------------
    if st.button("Run Analysis"):

        loc_id = [k for k, v in location_display.items() if v == selected_loc][0]
        w = get_location_weather(selected_loc)

        input_data = pd.DataFrame([[
            year, week, w['maxt'], 22.5, w['rh1'],
            w['rh1'] - 10, w['rf'], 4.5, 6.0, 4.0, loc_id
        ]], columns=meta['features'])

        prob = model.predict_proba(input_data)[0][1] * 100

        # -------------------------------
        # RISK CLASSIFICATION
        # -------------------------------
        if prob <= 35:
            level = "LOW"
            color = "green"
            advice = "No immediate action needed. Continue monitoring."
        elif prob <= 65:
            level = "MODERATE"
            color = "orange"
            advice = "Monitor crop closely and use preventive measures."
        else:
            level = "HIGH"
            color = "red"
            advice = "⚠️ Take immediate action. Consider pest control measures."

        # -------------------------------
        # OUTPUT
        # -------------------------------
        st.subheader("Risk Assessment")

        st.metric("Outbreak Probability", f"{prob:.2f}%")
        st.progress(prob / 100)

        st.markdown(
            f"### <span style='color:{color}'>Risk Level: {level}</span>",
            unsafe_allow_html=True
        )

        st.info(f"Recommendation: {advice}")

        # -------------------------------
        # SAVE TO DATABASE
        # -------------------------------
        if conn:
            try:
                cursor.execute(
                    "INSERT INTO predictions (location, probability, risk, date) VALUES (%s,%s,%s,%s)",
                    (selected_loc, float(prob), level, date_est)
                )
                conn.commit()
            except:
                st.warning("Could not save to database.")

else:
    st.error("Model not loaded properly.")