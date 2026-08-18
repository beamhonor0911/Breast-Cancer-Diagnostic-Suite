import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Oncology AI: Cancer Detection Inference API",
    description="REST API for predicting breast cancer malignancy from cellular morphometry features.",
    version="1.0.0"
)

MODEL_PATH = "cancer_model.joblib"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        print("Warning: cancer_model.joblib not found. Run production_pipeline.py first.")

class DiagnosticRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="30 numerical biomarkers (radius_mean, texture_mean, perimeter_mean, etc.)",
        min_items=30,
        max_items=30
    )

class DiagnosticResponse(BaseModel):
    malignancy_probability: float
    prediction_label: str
    triage_tier: str
    action_protocol: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=DiagnosticResponse)
def predict_diagnosis(payload: DiagnosticRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")

    input_data = np.array(payload.features).reshape(1, -1)
    prob = float(model.predict_proba(input_data)[0][1])
    
    if prob >= 0.65:
        tier = "CRITICAL / HIGH RISK"
        protocol = "Immediate specialist biopsy and ultrasound correlation recommended."
        label = "Malignant"
    elif prob >= 0.35:
        tier = "UNCERTAIN / REVIEW REQUIRED"
        protocol = "Flagged for human-in-the-loop pathologist secondary examination."
        label = "Borderline"
    else:
        tier = "LOW RISK"
        protocol = "Routine standard annual screening."
        label = "Benign"

    return DiagnosticResponse(
        malignancy_probability=round(prob, 4),
        prediction_label=label,
        triage_tier=tier,
        action_protocol=protocol
    )