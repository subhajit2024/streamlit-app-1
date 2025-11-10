import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="😴 Sleep Pattern Estimator", layout="wide")

# -----------------------------
# LOAD MODEL ASSETS
# -----------------------------
@st.cache_resource
def load_assets():
    model = joblib.load('kmeans_model.joblib')
    scaler = joblib.load('scaler.joblib')
    centers = joblib.load('cluster_centers.joblib')
    return model, scaler, centers

kmeans_model, scaler, centers = load_assets()

# -----------------------------
# CLUSTER → LABEL
# -----------------------------
CLUSTER_MAP = {
    2: 'Likely Low Sleep',
    1: 'Likely Moderate Sleep',
    0: 'Likely High Sleep'
}

# -----------------------------
# ESTIMATION LOGIC
# -----------------------------
def assign_hours(cluster, norm_distance):
    if cluster == 2:
        return 4 + (2 * (1 - norm_distance))
    elif cluster == 1:
        return 6 + (2 * (1 - norm_distance))
    else:
        return 8 + (1.5 * (1 - norm_distance))

# -----------------------------
# UI TITLE ---------------------
# -----------------------------
st.title("😴 Sleep Pattern Estimator")
st.markdown("### A smarter way to estimate your sleep schedule based on lifestyle factors.")
st.markdown("---")

# SIDEBAR INPUTS
st.sidebar.header("🧩 Enter Your Lifestyle Details")

daily_social_media = st.sidebar.slider(
    'Daily Social Media Minutes', 0, 600, 180, 10,
)

gaming_hours_week = st.sidebar.slider(
    'Gaming Hours Per Week', 0, 50, 5, 1,
)

intro_extro = st.sidebar.slider(
    'Introversion (1) → Extraversion (10)', 1, 10, 5, 1,
)

input_data = np.array([
    daily_social_media,
    gaming_hours_week,
    intro_extro
]).reshape(1, -1)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button('Estimate Sleep Hours 😴'):
    cluster = kmeans_model.predict(input_data)[0]
    center = centers[cluster]
    dist = np.linalg.norm(input_data[0] - center)
    norm_distance = scaler.transform(np.array([[dist]]))[0][0]
    estimated_hours = assign_hours(cluster, norm_distance)

    # RESULT AREA
    st.markdown("---")
    st.header("✅ Your Sleep Estimate")
    st.metric("Estimated Sleep Per Night", f"{estimated_hours:.1f} hours")
    st.success(f"Detected Group: **{CLUSTER_MAP[cluster]}**")

    # -----------------------------
    # PLOT 1 — DISTANCE VISUALIZATION
    # -----------------------------
    st.subheader("📊 Distance From Cluster Center")
    fig, ax = plt.subplots()
    labels = ['Your Distance', 'Max Normalized Distance']
    values = [1 - norm_distance, norm_distance]
    ax.bar(labels, values)
    ax.set_ylabel("Value")
    ax.set_title("How close your lifestyle pattern is to the cluster center")
    st.pyplot(fig)

    # -----------------------------
    # PLOT 2 — INPUT DISTRIBUTION
    # -----------------------------
    st.subheader("📈 Your Inputs Compared to Typical Ranges")
    fig2, ax2 = plt.subplots()
    categories = ['Social Media (min)', 'Gaming (hrs/week)', 'Personality Score']
    user_vals = [daily_social_media, gaming_hours_week, intro_extro]
    ax2.bar(categories, user_vals)
    ax2.set_title("Your Lifestyle Factors")
    st.pyplot(fig2)
