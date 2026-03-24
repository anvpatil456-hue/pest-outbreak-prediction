import os
import gdown
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta

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
            # Using st.toast instead of st.info for a cleaner UI
            st.toast(f"Downloading {file}...", icon="📥")
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                file,
                quiet=True
            )

    try:
        model = joblib.load("binary_pest_model.pkl")
        meta = joblib.load("binary_metadata.pkl")
        return model, meta
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None

model, meta = setup_model()

# -------------------------------
# 2. DATABASE CONNECTION (STREAMLIT NATIVE)
# -------------------------------
# This uses Streamlit's built-in SQL connection manager
try:
    conn = st.connection("mysql", type="sql")
except Exception as e:
    st.warning("Database configuration missing in Secrets.")
    conn = None

# -------------------------------
# 3. CONFIG & UI
# -------------------------------
st.set_page_config(page_title="KRISHISETU | Rice Pest AI", layout="centered", page_icon="🌾")
st.title("🌾 KRISHISETU: Pest Outbreak Prediction")

kharif_months = {
    "June": [22, 23, 24, 25, 26], "July": [27, 28, 29, 30],
    "August": [31, 32, 33, 34, 35], "September": [36, 37, 38, 39],
    "October": [40, 41, 42, 43, 44], "November": [45, 46, 47, 48]
}

location_display = {
    0: "Ratnagiri (Chiplun)", 1: "Raigad (Mahad)",
    2: "Bhandara (Tumsar)", 3: "Chandrapur (Nagbhid)",
    4: "Kolhapur (Shahuwadi)"
}

def get_location_weather(loc):
    profiles = {
        "Ratnagiri (Chiplun)":  {"maxt": 28.2, "rh1": 94.0, "rf": 38.0},
        "Raigad (Mahad)":        {"maxt": 29.5, "rh1": 91.0, "rf": 32.0},
        "Bhandara (Tumsar)":    {"maxt": 31.0, "rh1": 87.0, "rf": 18.0},
        "Chandrapur (Nagbhid)": {"maxt": 32.8, "rh1": 83.0, "rf": 14.0},
        "Kolhapur (Shahuwadi)": {"maxt": 27.0, "rh1": 86.0, "rf": 22.0}
    }
    return profiles.get(loc, {"maxt": 30, "rh1": 85, "rf": 15})

# -------------------------------
# 4. MAIN INTERFACE
# -------------------------------
if model and meta:
    st.subheader("Analyze Current Conditions")
    col1, col2 = st.columns(2)

    with col1:
        selected_loc = st.selectbox("Select Location", list(location_display.values()))
        year = st.selectbox("Year", [2025, 2026])

    with col2:
        month = st.selectbox("Month", list(kharif_months.keys()))
        week = st.selectbox("Week Number", kharif_months[month])
        date_est = datetime(year, 1, 1) + timedelta(weeks=week - 1)
        st.caption(f"Predicted Week: {date_est.strftime('%d %B %Y')}")

    if st.button("Generate Prediction", type="primary"):
        loc_id = [k for k, v in location_display.items() if v == selected_loc][0]
        w = get_location_weather(selected_loc)

        input_data = pd.DataFrame([[
            year, week, w['maxt'], 22.5, w['rh1'],
            w['rh1'] - 10, w['rf'], 4.5, 6.0, 4.0, loc_id
        ]], columns=meta['features'])

        prob = model.predict_proba(input_data)[0][1] * 100

        # Risk Logic
        if prob <= 35:
            level, color, advice = "LOW", "green", "No immediate action needed. Continue monitoring."
        elif prob <= 65:
            level, color, advice = "MODERATE", "orange", "Monitor crop closely and use preventive measures."
        else:
            level, color, advice = "HIGH", "red", "⚠️ Take immediate action. Apply pest control."

        st.divider()
        st.metric("Outbreak Probability", f"{prob:.2f}%")
        st.progress(prob / 100)
        st.markdown(f"### Risk Level: :{color}[{level}]")
        st.info(f"**Expert Advice:** {advice}")

        # Save to Cloud Database
        if conn:
            try:
                with conn.session as session:
                    session.execute(
                        "INSERT INTO predictions (location, probability, risk, date) VALUES (:loc, :prob, :risk, :date)",
                        {"loc": selected_loc, "prob": float(prob), "risk": level, "date": date_est}
                    )
                    session.commit()
                st.success("Analysis saved to cloud database.")
            except Exception as e:
                st.error(f"Database error: {e}")

    # -------------------------------
    # 5. VIEW HISTORY (NEW FEATURE)
    # -------------------------------
    with st.expander("View Historical Predictions"):
        if conn:
            try:
                history_df = conn.query("SELECT location, probability, risk, date FROM predictions ORDER BY id DESC LIMIT 5")
                st.table(history_df)
            except:
                st.write("No historical data available yet.")
        else:
            st.write("Connect a database in Secrets to see history.")

else:
    st.error("Model files could not be loaded from Google Drive. Check file IDs.")
