import argparse
import os
import pandas as pd
import joblib
import psutil
from feature_extraction_text import features_from_dataframe
from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        average_precision_score, confusion_matrix, matthews_corrcoef, brier_score_loss
    )
 


def evaluate_model(csv_path, model_path="phishing_text_model.joblib"):
    """
    Evaluate model on labeled CSV and print metrics
    Args:
        csv_path: Path to CSV with 'subject', 'body', and 'label' columns
        model_path: Path to saved model
    """
    df = pd.read_csv(csv_path)
    required_cols = {"subject", "body", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain 'subject', 'body', and 'label' columns")

    # Only keep rows with valid labels (0 or 1)
    valid_rows = df[df["label"].isin([0, 1])]
    parsing_success_rate = len(valid_rows) / max(1, len(df))

    # Load model
    model_data = joblib.load(model_path)
    pipeline = model_data["pipeline"]
    threshold = model_data.get("threshold", 0.5)

    # --- Memory usage measurement for evaluation ---
    process = psutil.Process(os.getpid())
    mem_before_eval = process.memory_info().rss

    # Extract features
    X = features_from_dataframe(valid_rows[["subject", "body"]])
    y_true = valid_rows["label"].values
    probas = pipeline.predict_proba(X)[:, 1]
    y_pred = (probas >= threshold).astype(int)

    mem_after_eval = process.memory_info().rss
    eval_mem_mb = (mem_after_eval - mem_before_eval) / (1024 * 1024)
    # --- End measurement ---

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_pr = average_precision_score(y_true, probas)
    roc_auc = roc_auc_score(y_true, probas)
    cm = confusion_matrix(y_true, y_pred)
    matthews = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, probas)
    # Confusion matrix: [[TN, FP], [FN, TP]]
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    total_predictions = len(y_true)
    correct_predictions = TN + TP
    total_predicted_positive = TP + FP
    total_actual_positive = TP + FN
    print(f"\nEvaluating on {len(y_true)} samples from '{os.path.basename(csv_path)}':")
    print(f"  - Legitimate (0): {(y_true == 0).sum()} samples")
    print(f"  - Phishing (1):   {(y_true == 1).sum()} samples")

    print("\nModel Evaluation Metrics:")
    print(f"  Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions} correct)")
    print(f"  Precision: {precision:.2%} ({TP} TP / {total_predicted_positive} predicted positive)")
    print(f"  Recall: {recall:.2%} ({TP} TP / {total_actual_positive} actual positive)")
    print(f"  F1-Score: {f1:.2%}")
    print(f"  AUC-PR: {auc_pr:.2%}")
    print(f"  ROC-AUC: {roc_auc:.2%}")
    print(f"  Confusion Matrix: [TN={TN}, FP={FP}], [FN={FN}, TP={TP}]")
    print(f"  Matthews Correlation: {matthews:.3f}") # This is a coefficient, not a percentage
    print(f"  Brier Score: {brier:.4f}") # This is a score, lower is better, not a percentage
    print(f"  Parsing Success Rate: {parsing_success_rate:.2%} ({len(valid_rows)}/{len(df)})")
    print(f"  Evaluation RAM usage: {eval_mem_mb:.2f} MB")


def predict_email(subject, body, model_path="phishing_text_model.joblib"):
    """
    Predict if an email is phishing based on subject and body
    
    Args:
        subject: Email subject line
        body: Email body text
        model_path: Path to saved model
    
    Returns:
        dict with prediction, probability, and confidence
    """
    # Load model
    model_data = joblib.load(model_path)
    pipeline = model_data["pipeline"]
    threshold = model_data.get("threshold", 0.5)
    
    # Create DataFrame 
    df = pd.DataFrame([{"subject": subject, "body": body}])
    
    # Extract features
    X = features_from_dataframe(df)
    
    # Get prediction
    proba = pipeline.predict_proba(X)[0, 1]
    prediction = 1 if proba >= threshold else 0
    
    return {
        "is_phishing": bool(prediction),
        "phishing_probability": float(proba),
        "confidence": float(abs(proba - 0.5) * 2),  # 0 to 1 scale
        "label": "Phishing" if prediction == 1 else "Legitimate"
    }

def predict_batch(csv_path, output_path="predictions.csv", model_path="phishing_text_model.joblib"):
    """
    Predict phishing for batch of emails from CSV
    
    Args:
        csv_path: Path to CSV with 'subject' and 'body' columns
        output_path: Path to save predictions
        model_path: Path to saved model
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    if "subject" not in df.columns or "body" not in df.columns:
        raise ValueError("CSV must contain 'subject' and 'body' columns")
    
    # Load model
    model_data = joblib.load(model_path)
    pipeline = model_data["pipeline"]
    threshold = model_data.get("threshold", 0.5)
    
    # Extract features
    X = features_from_dataframe(df[["subject", "body"]])
    
    # Get predictions
    probas = pipeline.predict_proba(X)[:, 1]
    predictions = (probas >= threshold).astype(int)
    
    # Add to dataframe
    df["phishing_probability"] = probas
    df["prediction"] = predictions
    df["prediction_label"] = df["prediction"].map({0: "Legitimate", 1: "Phishing"})
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Total emails: {len(df)}")
    print(f"  Predicted phishing: {predictions.sum()}")
    print(f"  Predicted legitimate: {len(predictions) - predictions.sum()}")
    
    return df

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing Email Prediction and Model Evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Batch prediction option
    batch_parser = subparsers.add_parser("batch", help="Run batch prediction on emails CSV")
    batch_parser.add_argument("--input", required=True, help="Input CSV file with 'subject' and 'body' columns")
    batch_parser.add_argument("--output", required=True, help="Output CSV file for predictions")
    batch_parser.add_argument("--model", default="phishing_text_model.joblib", help="Path to trained model file")

    # Model evaluation option
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on labeled emails CSV")
    eval_parser.add_argument("--input", required=True, help="Input CSV file with 'subject', 'body', and 'label' columns")
    eval_parser.add_argument("--model", default="phishing_text_model.joblib", help="Path to trained model file")

    args = parser.parse_args()

    if args.command == "batch":
        predict_batch(args.input, args.output, args.model) 
    elif args.command == "evaluate":
        evaluate_model(args.input, args.model)