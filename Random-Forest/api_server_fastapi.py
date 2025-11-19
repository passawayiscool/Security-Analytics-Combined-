"""
FastAPI server for Random Forest phishing detection
Minimal REST API similar to XGBoost implementation
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
from feature_extraction_rf import features_from_text
import uvicorn
import os
from dotenv import load_dotenv
import requests
from datetime import datetime
import hashlib
import socket
import io

# Load environment variables from .env file
load_dotenv()

# Splunk configuration - ADD YOUR VALUES IN .env FILE
SPLUNK_HEC_URL = os.getenv("SPLUNK_HEC_URL")
SPLUNK_TOKEN = os.getenv("SPLUNK_TOKEN")
SPLUNK_INDEX = os.getenv("SPLUNK_INDEX", "security")

# Load model at startup
MODEL_PATH = "checkpoints/phishing_detector/rf_phishing_detector.joblib"
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]

app = FastAPI(
    title="Random Forest Phishing Detection API",
    description="REST API for phishing email detection using Random Forest",
    version="1.0.0"
)


class EmailRequest(BaseModel):
    subject: str
    body: str


class EmailBatchRequest(BaseModel):
    emails: List[EmailRequest]


class PredictionResponse(BaseModel):
    is_phishing: bool
    phishing_probability: float
    confidence: float
    risk_score: int
    label: str
    recommended_action: str


def send_to_splunk(email_data: dict, prediction_result: dict):
    """Send alert to Splunk HTTP Event Collector"""
    if not SPLUNK_HEC_URL or not SPLUNK_TOKEN:
        return  # Splunk not configured, skip silently
    
    try:
        email_hash = hashlib.sha256(
            f"{email_data.get('subject', '')}{email_data.get('body', '')}".encode()
        ).hexdigest()
        
        timestamp = datetime.utcnow().timestamp()
        
        # Determine severity based on risk score
        risk_score = prediction_result["risk_score"]
        if risk_score > 90:
            severity = "CRITICAL"
        elif risk_score > 70:
            severity = "HIGH"
        elif risk_score > 50:
            severity = "MEDIUM"
        elif risk_score > 30:
            severity = "LOW"
        else:
            severity = "INFO"
        
        # Build Splunk HEC event payload
        event = {
            "time": timestamp,
            "host": socket.gethostname(),
            "source": "rf-phishing-detector",
            "sourcetype": "phishing_detection",
            "index": SPLUNK_INDEX,
            "event": {
                "alert_id": f"phish-rf-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "event_type": "email_analysis",
                "severity": severity,
                "email": {
                    "subject": email_data.get("subject", ""),
                    "body_preview": email_data.get("body", "")[:200],
                    "hash": email_hash,
                    "size_bytes": len(email_data.get("body", ""))
                },
                "detection": {
                    "classification": prediction_result["label"],
                    "is_phishing": prediction_result["is_phishing"],
                    "phishing_probability": prediction_result["phishing_probability"],
                    "confidence": prediction_result["confidence"],
                    "risk_score": prediction_result["risk_score"],
                    "recommended_action": prediction_result["recommended_action"],
                    "model_type": "RandomForest"
                },
                "actions_taken": {
                    "email_quarantined": risk_score > 70,
                    "sender_blocked": prediction_result["recommended_action"] == "BLOCK",
                    "soc_alerted": risk_score > 80
                }
            }
        }
        
        # Send to Splunk HEC
        headers = {
            "Authorization": f"Splunk {SPLUNK_TOKEN}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            SPLUNK_HEC_URL,
            json=event,
            headers=headers,
            timeout=5,
            verify=False  # Set to True in production with proper SSL certs
        )
        
        if response.status_code == 200:
            print(f"✓ Alert sent to Splunk: {event['event']['alert_id']}")
        else:
            print(f"✗ Splunk error: {response.status_code}")
            
    except Exception as e:
        print(f"✗ Splunk send failed: {e}")


@app.get("/")
def root():
    return {
        "service": "Random Forest Phishing Detection API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(email: EmailRequest):
    """Predict if a single email is phishing"""
    try:
        # Combine subject and body
        email_text = f"Subject: {email.subject}\n\n{email.body}"
        
        # Extract features
        features_dict = features_from_text(email_text)
        feature_vector = np.array([features_dict.get(name, 0.0) for name in feature_names])
        X = feature_vector.reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        
        # Determine recommended action
        if proba[1] > 0.9:
            action = "BLOCK"
        elif proba[1] > 0.7:
            action = "QUARANTINE"
        elif proba[1] > 0.5:
            action = "REVIEW"
        else:
            action = "ALLOW"
        
        result = {
            "is_phishing": bool(prediction == 1),
            "phishing_probability": float(proba[1]),
            "confidence": float(proba[prediction]),
            "risk_score": int(proba[1] * 100),
            "label": "Phishing" if prediction == 1 else "Legitimate",
            "recommended_action": action
        }
        
        # Send to Splunk if phishing detected or high risk
        if result["is_phishing"] or result["risk_score"] > 50:
            send_to_splunk(
                email_data={"subject": email.subject, "body": email.body},
                prediction_result=result
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(batch: EmailBatchRequest):
    """Predict multiple emails at once"""
    try:
        results = []
        for email in batch.emails:
            email_text = f"Subject: {email.subject}\n\n{email.body}"
            
            features_dict = features_from_text(email_text)
            feature_vector = np.array([features_dict.get(name, 0.0) for name in feature_names])
            X = feature_vector.reshape(1, -1)
            X_scaled = scaler.transform(X)
            
            prediction = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            
            if proba[1] > 0.9:
                action = "BLOCK"
            elif proba[1] > 0.7:
                action = "QUARANTINE"
            elif proba[1] > 0.5:
                action = "REVIEW"
            else:
                action = "ALLOW"
            
            results.append({
                "is_phishing": bool(prediction == 1),
                "phishing_probability": float(proba[1]),
                "confidence": float(proba[prediction]),
                "risk_score": int(proba[1] * 100),
                "label": "Phishing" if prediction == 1 else "Legitimate",
                "recommended_action": action
            })
        
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Upload CSV file with 'subject' and 'body' columns for batch prediction"""
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate columns
        if 'subject' not in df.columns or 'body' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'subject' and 'body' columns"
            )
        
        results = []
        phishing_count = 0
        
        # Process each row
        for idx, row in df.iterrows():
            subject = str(row.get('subject', ''))
            body = str(row.get('body', ''))
            email_text = f"Subject: {subject}\n\n{body}"
            
            # Extract features and predict
            features_dict = features_from_text(email_text)
            feature_vector = np.array([features_dict.get(name, 0.0) for name in feature_names])
            X = feature_vector.reshape(1, -1)
            X_scaled = scaler.transform(X)
            
            prediction = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
            
            if proba[1] > 0.9:
                action = "BLOCK"
            elif proba[1] > 0.7:
                action = "QUARANTINE"
            elif proba[1] > 0.5:
                action = "REVIEW"
            else:
                action = "ALLOW"
            
            result = {
                "row_index": int(idx),
                "subject": subject,
                "is_phishing": bool(prediction == 1),
                "phishing_probability": float(proba[1]),
                "confidence": float(proba[prediction]),
                "risk_score": int(proba[1] * 100),
                "label": "Phishing" if prediction == 1 else "Legitimate",
                "recommended_action": action
            }
            
            results.append(result)
            
            # Send high-risk emails to Splunk
            if result["is_phishing"] or result["risk_score"] > 50:
                phishing_count += 1
                send_to_splunk(
                    email_data={"subject": subject, "body": body},
                    prediction_result=result
                )
        
        return {
            "status": "success",
            "total_emails": len(results),
            "phishing_detected": phishing_count,
            "legitimate_detected": len(results) - phishing_count,
            "predictions": results
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")


@app.get("/model/info")
def model_info():
    """Get information about the loaded model"""
    return {
        "model_type": "RandomForestClassifier",
        "n_estimators": model.n_estimators,
        "n_features": len(feature_names),
        "feature_names_sample": feature_names[:10],
        "metrics": model_data.get("metrics", {})
    }


if __name__ == "__main__":
    import sys
    
    # Allow custom port from command line
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except:
            port = 8000
    
    print("=" * 60)
    print("RANDOM FOREST PHISHING DETECTION API")
    print("=" * 60)
    print(f"\nStarting FastAPI server on port {port}...")
    print(f"API Documentation: http://localhost:{port}/docs")
    print(f"Health Check: http://localhost:{port}/health")
    print("\nPress CTRL+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
