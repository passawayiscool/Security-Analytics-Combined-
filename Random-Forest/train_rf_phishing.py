"""
Simplified training script for Random Forest phishing detection
Mirrors the concise style of the XgBoost training script.
"""
import time
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from feature_extraction_rf import features_from_dataframe
from tqdm import tqdm

print("=" * 60)
print("RANDOM FOREST PHISHING EMAIL DETECTION")
print("=" * 60)

# 1) Load + prepare dataset
print("\n[1] Loading and preparing dataset...")
df = pd.read_csv("Enron.csv").dropna(subset=["subject", "body", "label"]).copy()
print(f"   Total samples: {len(df)}")

y = df["label"].astype(int)
print(f"   Phishing ratio: {y.mean():.2%}")

# 2) Extract features
print("\n[2] Extracting features...")
X = features_from_dataframe(df[["subject", "body"]])
print(f"   Features extracted: {X.shape[1]}")
print(f"   Feature names: {list(X.columns)[:10]}...")

# 3) Train/validation/test split
print("\n[3] Splitting data...")
X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)

print(f"   Training set: {len(X_train)} samples")
print(f"   Validation set: {len(X_val)} samples")
print(f"   Test set: {len(X_holdout)} samples")

# 4) Class balance
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
print(f"\n[4] Class balance:")
print(f"   Negative (legitimate): {n_neg}")
print(f"   Positive (phishing): {n_pos}")

# 5) Build and train model
print("\n[5] Training Random Forest model...")
start = time.perf_counter()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_holdout)

# Initialize RF with good hyperparameters
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    bootstrap=True,
    oob_score=True,
    verbose=0
)

model.fit(X_train_scaled, y_train)
train_time = time.perf_counter() - start
print(f"   Training time: {train_time:.2f}s")
if hasattr(model, 'oob_score_'):
    print(f"   OOB score: {model.oob_score_:.4f}")

# 6) Validation metrics
print("\n[6] Validation Performance:")
val_pred = model.predict(X_val_scaled)
val_proba = model.predict_proba(X_val_scaled)[:, 1]
val_acc = accuracy_score(y_val, val_pred)
val_prec = precision_score(y_val, val_pred, zero_division=0)
val_rec = recall_score(y_val, val_pred, zero_division=0)
val_f1 = f1_score(y_val, val_pred, zero_division=0)
val_auc = roc_auc_score(y_val, val_proba)

print(f"   Accuracy:  {val_acc:.4f}")
print(f"   Precision: {val_prec:.4f}")
print(f"   Recall:    {val_rec:.4f}")
print(f"   F1 Score:  {val_f1:.4f}")
print(f"   ROC-AUC:   {val_auc:.4f}")

# 7) Test set evaluation
print("\n[7] Test Set Evaluation:")
test_pred = model.predict(X_test_scaled)
test_proba = model.predict_proba(X_test_scaled)[:, 1]
test_acc = accuracy_score(y_holdout, test_pred)
test_prec = precision_score(y_holdout, test_pred, zero_division=0)
test_rec = recall_score(y_holdout, test_pred, zero_division=0)
test_f1 = f1_score(y_holdout, test_pred, zero_division=0)
test_auc = roc_auc_score(y_holdout, test_proba)

print(f"   Accuracy:  {test_acc:.4f}")
print(f"   Precision: {test_prec:.4f}")
print(f"   Recall:    {test_rec:.4f}")
print(f"   F1 Score:  {test_f1:.4f}")
print(f"   ROC-AUC:   {test_auc:.4f}")

# 8) Feature importance
print("\n[8] Top 15 Feature Importances:")
feature_importance = model.feature_importances_
feature_names = X.columns.tolist()
importance_pairs = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)

for i, (fname, imp) in enumerate(importance_pairs[:15], 1):
    print(f"   {i:02d}. {fname}: {imp:.4f}")

# 9) Save model
print("\n[9] Saving model...")
output_dir = Path("checkpoints/phishing_detector")
output_dir.mkdir(parents=True, exist_ok=True)
model_path = output_dir / "rf_phishing_detector.joblib"

model_data = {
    "model": model,
    "scaler": scaler,
    "feature_names": feature_names,
    "metrics": {
        "test_accuracy": float(test_acc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
        "test_roc_auc": float(test_auc),
        "train_time_seconds": float(train_time)
    }
}

joblib.dump(model_data, model_path)
print(f"   Model saved: {model_path}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
