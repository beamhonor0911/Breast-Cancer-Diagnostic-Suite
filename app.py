import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Cancer Detection AI Diagnostic Suite", layout="wide")

st.title("🧬 Breast Cancer Diagnostic & Triage System")
st.markdown("Automated clinical risk stratification based on cellular morphometric biomarkers.")

MODEL_PATH = "cancer_model.joblib"
DATA_PATH = "breast-cancer.csv"

@st.cache_resource
def load_model_and_reference():
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    df = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None
    if df is not None:
        df.drop(columns=[c for c in ["id", "Unnamed: 32"] if c in df.columns], inplace=True)
    return model, df

model, df = load_model_and_reference()

if model is None or df is None:
    st.error("Missing model artifact or dataset. Please run `production_pipeline.py` first.")
    st.stop()

features = [c for c in df.columns if c != "diagnosis"]

st.sidebar.header("Biomarker Input Panel")
st.sidebar.markdown("Adjust cellular features or use sample presets:")

preset = st.sidebar.selectbox("Load Patient Preset", ["Custom Adjustments", "Sample Benign Case", "Sample Malignant Case"])

if preset == "Sample Benign Case":
    sample_values = df[df["diagnosis"] == "B"].iloc[0][features].to_dict()
elif preset == "Sample Malignant Case":
    sample_values = df[df["diagnosis"] == "M"].iloc[0][features].to_dict()
else:
    sample_values = df[features].mean().to_dict()

input_values = []
col1, col2, col3 = st.columns(3)

for idx, feat in enumerate(features):
    min_v = float(df[feat].min())
    max_v = float(df[feat].max())
    default_v = float(sample_values[feat])
    
    target_col = col1 if idx % 3 == 0 else (col2 if idx % 3 == 1 else col3)
    val = target_col.slider(f"{feat}", min_v, max_v, default_v)
    input_values.append(val)

st.markdown("---")
if st.button("Run Diagnostic Inference", type="primary"):
    input_arr = np.array(input_values).reshape(1, -1)
    prob = float(model.predict_proba(input_arr)[0][1])
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(label="Malignancy Probability", value=f"{prob * 100:.2f}%")
        st.progress(prob)
        
    with res_col2:
        if prob >= 0.65:
            st.error("🚨 **Classification: Malignant (High Risk)**\n\nImmediate biopsy and clinical consultation required.")
        elif prob >= 0.35:
            st.warning("⚠️ **Classification: Borderline / Uncertain**\n\nConfidence in uncertainty band. Requires pathologist review.")
        else:
            st.success("✅ **Classification: Benign (Low Risk)**\n\nNormal cellular characteristics. Routine follow-up scheduled.")