# app.py (CORRECTED TOP SECTION)

# 1. First, only import streamlit
import streamlit as st

# 🚨 This MUST be the first Streamlit command executed 🚨
st.set_page_config(page_title="Sleep Pattern Estimator", layout="sidebar") 
# ------------------------------------------------------------------------

# 2. Then, import other libraries
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans 
# ... and the rest of your script

# --- 1. CONFIGURATION AND CACHED ASSET LOADING ---

# 🚨 CRITICAL FIX: st.set_page_config MUST be the first Streamlit command 🚨
st.set_page_config(page_title="Sleep Pattern Estimator", layout="sidebar") 
# -------------------------------------------------------------------------

# Map clusters to sleep groups (based on your notebook's logic)
CLUSTER_MAP = {
    2: 'Likely Low Sleep',
    1: 'Likely Moderate Sleep',
    0: 'Likely High Sleep'
}

# Use st.cache_resource for heavy, unchanging resources like models
@st.cache_resource
def load_assets():
    """Loads the trained model, scaler, and cluster centers."""
    try:
        # These filenames must match your downloaded .joblib files
        model = joblib.load('kmeans_model.joblib')
        scaler = joblib.load('scaler.joblib')
        centers = joblib.load('cluster_centers.joblib')
        return model, scaler, centers
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'kmeans_model.joblib', 'scaler.joblib', and 'cluster_centers.joblib' are in the same directory.")
        st.stop()
        
kmeans_model, scaler, centers = load_assets()

# --- 2. CORE ESTIMATION LOGIC ---

def assign_hours(cluster, norm_distance):
    """Calculates estimated sleep hours based on cluster and normalized distance."""
    # Logic copied directly from your Jupyter Notebook
    if cluster == 2:  # Low (4-6 hrs range)
        return 4 + (2 * (1 - norm_distance))
    elif cluster == 1:  # Moderate (6-8 hrs range)
        return 6 + (2 * (1 - norm_distance))
    else:  # High (8-9.5 hrs range)
        return 8 + (1.5 * (1 - norm_distance))

# --- 3. STREAMLIT UI AND INPUTS ---

st.title("😴 Sleep Pattern Estimator")
st.markdown("Estimate your sleep hours based on **social media, gaming, and personality.**")

st.sidebar.header("Input Your Lifestyle Data")

# User Inputs using Streamlit Widgets (Sliders)
daily_social_media = st.sidebar.slider(
    'Daily Social Media Minutes', 
    min_value=0, 
    max_value=600, 
    value=180, 
    step=10,
    help="Provide value between 0 and 600 minutes."
)

gaming_hours_week = st.sidebar.slider(
    'Gaming Hours Per Week', 
    min_value=0, 
    max_value=50, 
    value=5, 
    step=1,
    help="Provide value between 0 and 50 hours."
)

intro_extro = st.sidebar.slider(
    'Introversion (1) to Extraversion (10)', 
    min_value=1, 
    max_value=10, 
    value=5, 
    step=1,
    help="Rate yourself on a scale from 1 (Introvert) to 10 (Extrovert)."
)

# Organize input data for the model
input_data = np.array([
    daily_social_media, 
    gaming_hours_week, 
    intro_extro
]).reshape(1, -1) # Reshape for the model

# --- 4. PREDICTION AND DISPLAY ---

if st.button('Estimate Sleep Hours'):
    
    # 1. Predict Cluster
    cluster = kmeans_model.predict(input_data)[0]
    
    # 2. Calculate Distance from Cluster Center
    center = centers[cluster]
    dist = np.linalg.norm(input_data[0] - center) # Euclidean distance

    # 3. Normalize the Distance
    # Reshape the single distance value for the scaler: np.array([[dist]])
    norm_distance = scaler.transform(np.array([[dist]]))[0][0]

    # 4. Estimate Sleep Hours
    estimated_hours = assign_hours(cluster, norm_distance)

    # --- Display Results ---
    st.markdown("---")
    st.header("Your Estimated Sleep")
    
    st.metric(label="Estimated Hours per Night", value=f"{estimated_hours:.1f} hours")
    
    st.success(f"**Sleep Pattern Group:** {CLUSTER_MAP[cluster]}")

    st.markdown(f"""
        <div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-top: 15px;'>
            **Details:**
            The model clustered your data into **Cluster {cluster}** ({CLUSTER_MAP[cluster]}).
            The final hour estimate is adjusted based on your inputs' distance from the cluster's center.
        </div>
        """, unsafe_allow_html=True)

