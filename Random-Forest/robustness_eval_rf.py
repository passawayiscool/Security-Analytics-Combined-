"""Simple robustness evaluation for Random Forest phishing model.
Applies lightweight text perturbations to assess prediction stability.
"""
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from feature_extraction_rf import features_from_text

MODEL_PATH = "checkpoints/phishing_detector/rf_phishing_detector.joblib"

def perturbations(email_text: str):
    """Generate text perturbations"""
    variants = []
    variants.append("URGENT! " + email_text)
    variants.append(email_text.lower())
    variants.append("IMPORTANT: " + email_text)
    variants.append(email_text.replace("http", "hxxp"))
    variants.append(email_text + "\nThis is a routine informational update.")
    return variants


def main():
    print("=" * 60)
    print("RANDOM FOREST ROBUSTNESS EVALUATION")
    print("=" * 60)

    # Load model
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_names = model_data["feature_names"]

    # Load and sample dataset
    df = pd.read_csv("Enron.csv").dropna(subset=["subject", "body", "label"]).copy()
    sample_df = df.sample(n=min(10, len(df)), random_state=42)

    rows = []
    for _, row in sample_df.iterrows():
        original_text = f"Subject: {row['subject']}\n\n{row['body']}"
        
        # Original prediction
        orig_features = features_from_text(original_text)
        feature_vector = np.array([orig_features.get(name, 0.0) for name in feature_names])
        X = feature_vector.reshape(1, -1)
        X_scaled = scaler.transform(X)
        orig_proba = model.predict_proba(X_scaled)[0, 1]

        # Test perturbations
        vars_texts = perturbations(original_text)
        drift = []
        for vt in vars_texts:
            feats = features_from_text(vt)
            vec = np.array([feats.get(name, 0.0) for name in feature_names])
            Xv = vec.reshape(1, -1)
            Xv_scaled = scaler.transform(Xv)
            p = model.predict_proba(Xv_scaled)[0, 1]
            drift.append(p - orig_proba)

        rows.append({
            'original_probability': float(orig_proba),
            'max_positive_shift': float(max(drift)),
            'max_negative_shift': float(min(drift)),
            'avg_abs_shift': float(sum(abs(d) for d in drift) / len(drift)),
            'n_perturbations': len(drift)
        })

    # Aggregate stats
    avg_abs = sum(r['avg_abs_shift'] for r in rows) / len(rows)
    max_pos = max(r['max_positive_shift'] for r in rows)
    max_neg = min(r['max_negative_shift'] for r in rows)

    print("\nRobustness Summary (probability shifts under perturbations):")
    print(f"   Avg absolute shift: {avg_abs:.4f}")
    print(f"   Max positive shift: {max_pos:.4f}")
    print(f"   Max negative shift: {max_neg:.4f}")
    print(f"   Samples analyzed:   {len(rows)}")

    import json
    with open("rf_robustness_report.json", "w", encoding="utf-8") as f:
        json.dump({
            'samples': rows,
            'summary': {
                'avg_abs_shift': avg_abs,
                'max_positive_shift': max_pos,
                'max_negative_shift': max_neg
            }
        }, f, indent=2)
    print("\nReport written to rf_robustness_report.json")


if __name__ == "__main__":
    main()
