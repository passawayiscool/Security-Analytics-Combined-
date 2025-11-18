"""Simplified phishing prediction script using trained Random Forest model."""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from feature_extraction_rf import features_from_text, features_from_dataframe

DEFAULT_MODEL = "checkpoints/phishing_detector/rf_phishing_detector.joblib"


def predict_email(email_text: str, model_path: str = DEFAULT_MODEL) -> dict:
    """Predict phishing classification for a single raw email text."""
    model_data = joblib.load(model_path)
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_names = model_data["feature_names"]
    
    # Extract features
    features_dict = features_from_text(email_text)
    feature_vector = np.array([features_dict.get(name, 0.0) for name in feature_names])
    X = feature_vector.reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    
    return {
        "classification": "PHISHING" if prediction == 1 else "LEGITIMATE",
        "phishing_probability": float(proba[1]),
        "confidence": float(proba[prediction]),
        "risk_score": int(proba[1] * 100),
        "recommended_action": "BLOCK" if proba[1] > 0.9 else "QUARANTINE" if proba[1] > 0.7 else "ALLOW"
    }


def predict_batch(csv_path: str, output_path: str = "rf_predictions.csv", model_path: str = DEFAULT_MODEL):
    """Batch predict phishing on a CSV containing an email text column.
    Accepts 'subject' and 'body' columns.
    """
    df = pd.read_csv(csv_path)
    
    if "subject" not in df.columns or "body" not in df.columns:
        raise ValueError("CSV must contain 'subject' and 'body' columns")
    
    model_data = joblib.load(model_path)
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_names = model_data["feature_names"]
    
    # Extract features
    X = features_from_dataframe(df[["subject", "body"]])
    
    # Ensure feature order matches training
    X_ordered = X[feature_names]
    X_scaled = scaler.transform(X_ordered)
    
    # Predict
    predictions = model.predict(X_scaled)
    probas = model.predict_proba(X_scaled)[:, 1]
    
    # Add to dataframe
    df["phishing_probability"] = probas
    df["prediction"] = predictions
    df["prediction_label"] = df["prediction"].map({0: "Legitimate", 1: "Phishing"})
    
    df.to_csv(output_path, index=False)
    print(f"Saved batch predictions to {output_path}")
    print(f"Total emails: {len(df)} | Phishing predicted: {(df['prediction']==1).sum()}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("RANDOM FOREST PHISHING PREDICTION EXAMPLES")
    print("=" * 60)

    sample_phish = """Subject: URGENT ACCOUNT LOCKED\n\nYour account has been suspended. Click here to unlock immediately: http://secure-verify-login.tk"""
    res1 = predict_email(sample_phish)
    print("\nExample 1 (Phishing-like):")
    print(res1)

    sample_legit = """Subject: Meeting notes\n\nHere are the notes from today's meeting. No action needed."""
    res2 = predict_email(sample_legit)
    print("\nExample 2 (Legitimate):")
    print(res2)

    print("\nFor batch prediction: predict_batch('emails.csv')")
