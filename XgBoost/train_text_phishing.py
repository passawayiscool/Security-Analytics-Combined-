# Text-based phishing detection training using XGBoost
# Uses email subject and body content for classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve, 
                            classification_report, confusion_matrix, accuracy_score, f1_score,
                            precision_score, recall_score, matthews_corrcoef, brier_score_loss)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBClassifier
import joblib
from feature_extraction_text import features_from_dataframe
import time
import os
import json

print("=" * 60)
print("TEXT-BASED PHISHING EMAIL DETECTION")
print("=" * 60)

# 1) Load data
print("\n[1] Loading Enron dataset...")
df = pd.read_csv("Enron.csv")
print(f"   Total samples: {len(df)}")
print(f"   Columns: {list(df.columns)}")

# Drop rows with missing subject or body
df = df.dropna(subset=["subject", "body", "label"])
print(f"   After dropping NaN: {len(df)} samples")

# Check label distribution
print(f"\n   Label distribution:")
print(df["label"].value_counts())
print(f"   Phishing ratio: {df['label'].mean():.2%}")

y = df["label"].astype(int)

# 2) Feature extraction
print("\n[2] Extracting text features...")
X = features_from_dataframe(df[["subject", "body"]])
print(f"   Features extracted: {X.shape[1]}")
print(f"   Feature names: {list(X.columns)}")

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

# 4) Compute scale_pos_weight (for imbalanced datasets)
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / max(1, n_pos)
print(f"\n[4] Class balance:")
print(f"   Negative (legitimate): {n_neg}")
print(f"   Positive (phishing): {n_pos}")
print(f"   scale_pos_weight: {scale_pos_weight:.2f}")

# 5) Build pipeline
print("\n[5] Building XGBoost pipeline...")
pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),  # Feature interactions
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=3000,  # Increase for better learning
        learning_rate=0.03,  # Lower learning rate for better generalization
        max_depth=10,  # Deeper trees to capture complex patterns
        min_child_weight=3,  # Prevent overfitting on small groups
        subsample=0.85,
        colsample_bytree=0.85,
        colsample_bylevel=0.85,  # Additional regularization
        gamma=0.1,  # Min loss reduction for split
        reg_alpha=0.5,  # L1 regularization (less aggressive)
        reg_lambda=2,  # L2 regularization (more aggressive)
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        early_stopping_rounds=150  # More patience
    ))
])

# 6) Train with early stopping
print("\n[6] Training model with early stopping...")
print("   (This may take a few minutes...)")
train_start = time.perf_counter()

# Transform training data through the pipeline steps (except xgb)
X_train_transformed = pipeline.named_steps["poly"].fit_transform(X_train)
X_train_transformed = pipeline.named_steps["scaler"].fit_transform(X_train_transformed)

# Transform validation data
X_val_transformed = pipeline.named_steps["poly"].transform(X_val)
X_val_transformed = pipeline.named_steps["scaler"].transform(X_val_transformed)

print(f"   Feature count after polynomial interactions: {X_train_transformed.shape[1]}")

pipeline.named_steps["xgb"].fit(
    X_train_transformed,
    y_train,
    eval_set=[(X_val_transformed, y_val)],
    verbose=50
)
train_time_s = time.perf_counter() - train_start

# 7) Evaluate on validation set
print("\n[7] Validation set performance:")
# Transform validation data again for prediction
X_val_for_pred = pipeline.named_steps["poly"].transform(X_val)
X_val_for_pred = pipeline.named_steps["scaler"].transform(X_val_for_pred)
val_probs = pipeline.named_steps["xgb"].predict_proba(X_val_for_pred)[:, 1]
val_avg_prec = average_precision_score(y_val, val_probs)
val_rocauc = roc_auc_score(y_val, val_probs)
print(f"   AUC-PR: {val_avg_prec:.4f}")
print(f"   ROC-AUC: {val_rocauc:.4f}")

# 8) Evaluate on holdout test set
print("\n[8] Test set performance:")
X_holdout_transformed = pipeline.named_steps["poly"].transform(X_holdout)
X_holdout_transformed = pipeline.named_steps["scaler"].transform(X_holdout_transformed)
test_probs = pipeline.named_steps["xgb"].predict_proba(X_holdout_transformed)[:, 1]

test_avg_prec = average_precision_score(y_holdout, test_probs)
test_rocauc = roc_auc_score(y_holdout, test_probs)
print(f"   AUC-PR: {test_avg_prec:.4f}")
print(f"   ROC-AUC: {test_rocauc:.4f}")
test_brier = brier_score_loss(y_holdout, test_probs)
print(f"   Brier score: {test_brier:.4f} (lower is better)")

# 9) Find optimal threshold by maximizing F1 score
print("\n[9] Finding optimal classification threshold...")
prec, rec, thresholds = precision_recall_curve(y_holdout, test_probs)
f1_scores = 2 * prec * rec / (prec + rec + 1e-12)
best_idx = np.nanargmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
print(f"   Best threshold: {best_threshold:.4f}")
print(f"   Best F1 score: {np.max(f1_scores):.4f}")

# 10) Classification report with optimal threshold
print("\n[10] Classification Report (with optimal threshold):")
test_preds = (test_probs >= best_threshold).astype(int)
print(classification_report(y_holdout, test_preds, target_names=["Legitimate", "Phishing"]))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_holdout, test_preds)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# Additional scalar metrics at best threshold
acc = accuracy_score(y_holdout, test_preds)
prec_th = precision_score(y_holdout, test_preds, zero_division=0)
rec_th = recall_score(y_holdout, test_preds, zero_division=0)
mcc = matthews_corrcoef(y_holdout, test_preds)
fpr = cm[0,1] / max(1, (cm[0,1] + cm[0,0]))
print(f"\nAccuracy: {acc:.4f}")
print(f"Precision: {prec_th:.4f}")
print(f"Recall: {rec_th:.4f}")
print(f"Matthews CorrCoef: {mcc:.4f}")
print(f"False Positive Rate: {fpr:.4f}")

# 11) Feature importance
print("\n[11] Feature Importance Analysis:")
feature_importance = pipeline.named_steps["xgb"].feature_importances_
print(f"   Total features (after polynomial interactions): {len(feature_importance)}")

# Get polynomial feature names
poly_feature_names = pipeline.named_steps["poly"].get_feature_names_out(X.columns)

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': poly_feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n   Top 20 Most Important Features:")
for idx, row in importance_df.head(20).iterrows():
    print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")

# Also calculate aggregated importance for original features
print("\n   Top 10 Original Features (aggregated importance):")
original_importance = {}
for feat_name, importance in zip(poly_feature_names, feature_importance):
    # Extract original feature names from polynomial features
    # e.g., "subject_length url_count" -> both features get credit
    for orig_feat in X.columns:
        if orig_feat in feat_name:
            original_importance[orig_feat] = original_importance.get(orig_feat, 0) + importance

# Sort and display
sorted_orig = sorted(original_importance.items(), key=lambda x: x[1], reverse=True)
for i, (feat_name, importance) in enumerate(sorted_orig[:10], 1):
    print(f"   {i}. {feat_name}: {importance:.4f}")

# 12) Save model

print("\n[12] Saving model...")
model_data = {
    "pipeline": pipeline,
    "threshold": float(best_threshold),
    "feature_names": list(X.columns),
    "metrics": {
        "test_auc_pr": float(test_avg_prec),
        "test_roc_auc": float(test_rocauc),
        "best_f1": float(np.max(f1_scores)),
        "brier_score": float(test_brier),
        "accuracy": float(acc),
        "precision": float(prec_th),
        "recall": float(rec_th),
        "mcc": float(mcc),
        "false_positive_rate": float(fpr),
        "confusion_matrix": cm.tolist(),
        "train_time_seconds": float(train_time_s)
    }
}

joblib.dump(model_data, "phishing_text_model.joblib")
print("   Model saved to: phishing_text_model.joblib")

# Save raw XGBoost model for compatibility
booster = pipeline.named_steps["xgb"].get_booster()
booster.save_model("phishing_text_model.xgb")
print("   Raw XGBoost model saved to: phishing_text_model.xgb")

# Also write a standalone JSON report for convenient consumption
report = {
    "dataset": "Enron.csv",
    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
    "n_test": int(len(X_holdout)),
    "threshold": float(best_threshold),
    "metrics": model_data["metrics"],
}
try:
    # Attach model file size if available
    if os.path.exists("phishing_text_model.joblib"):
        report["model_size_bytes"] = os.path.getsize("phishing_text_model.joblib")
    if os.path.exists("phishing_text_model.xgb"):
        report["xgb_model_size_bytes"] = os.path.getsize("phishing_text_model.xgb")
    with open("metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("   Metrics report written to: metrics_report.json")
except Exception:
    pass

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
