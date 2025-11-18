# Predict phishing emails using trained text-based model

import pandas as pd
import joblib
from feature_extraction_text import features_from_dataframe

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
    # Example 1: Phishing email prediction
    print("=" * 60)
    print("EXAMPLE 1: Phishing Email Detection")
    print("=" * 60)
    
    test_subject_phishing = "URGENT: Verify your account immediately"
    test_body_phishing = """
    Dear Customer,
    
    Your account has been suspended due to unusual activity.
    Click here to verify your identity immediately:
    http://suspicious-bank-verify.tk/login
    
    Failure to act within 24 hours will result in permanent account closure.
    """
    
    result_phishing = predict_email(test_subject_phishing, test_body_phishing)
    print(f"\nSubject: {test_subject_phishing}")
    print(f"Prediction: {result_phishing['label']}")
    print(f"Probability: {result_phishing['phishing_probability']:.2%}")
    print(f"Confidence: {result_phishing['confidence']:.2%}")
    
    # Example 2: Legitimate email prediction (order confirmation)
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Legitimate Email Detection (Order Confirmation)")
    print("=" * 60)
    
    test_subject_legitimate = "Your order #12345 has been shipped"
    test_body_legitimate = """
    Hi John,
    
    Great news! Your order #12345 has been shipped and is on its way.
    
    Order Details:
    - Product: Laptop Computer
    - Tracking Number: 1Z999AA10123456784
    - Expected Delivery: November 10, 2025
    
    You can track your package at our official website.
    
    Thank you for shopping with us!
    
    Best regards,
    Customer Service Team
    """
    
    result_legitimate = predict_email(test_subject_legitimate, test_body_legitimate)
    print(f"\nSubject: {test_subject_legitimate}")
    print(f"Prediction: {result_legitimate['label']}")
    print(f"Probability: {result_legitimate['phishing_probability']:.2%}")
    print(f"Confidence: {result_legitimate['confidence']:.2%}")
    
    # Example 3: Legitimate email prediction (meeting notes - high confidence)
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Legitimate Email Detection (Meeting Notes)")
    print("=" * 60)
    
    test_subject_legitimate2 = "Meeting notes from today's standup"
    test_body_legitimate2 = """
    Hi team,
    
    Here are the key points from today's standup meeting:
    
    - Sprint progress is on track, 8 out of 10 stories completed
    - Design review scheduled for Thursday at 2pm
    - John will demo the new feature tomorrow
    - No major blockers reported
    
    Next standup is tomorrow at 9am as usual.
    
    Please review the attached documents before tomorrow's design meeting.
    
    Thanks,
    Sarah
    """
    
    result_legitimate2 = predict_email(test_subject_legitimate2, test_body_legitimate2)
    print(f"\nSubject: {test_subject_legitimate2}")
    print(f"Prediction: {result_legitimate2['label']}")
    print(f"Probability: {result_legitimate2['phishing_probability']:.2%}")
    print(f"Confidence: {result_legitimate2['confidence']:.2%}")
    
    # Example 4: Batch prediction
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Batch Prediction")
    print("=" * 60)
    print("\nTo predict on a CSV file, use:")
    print('  predict_batch("your_emails.csv", "predictions.csv")')