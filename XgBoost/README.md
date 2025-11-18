# XGBOOST_MODEL

## Text-Based Phishing Email Detection

A machine learning system for detecting phishing emails using XGBoost classifier. The model analyzes email content (subject and body) to identify phishing attempts with high accuracy.

## 🎯 Features

- **Text-based Analysis**: Extracts sophisticated features from email subject and body text
- **XGBoost Classifier**: Utilizes gradient boosting for robust classification
- **Feature Engineering**: 
  - URL analysis (count, suspicious domains, IP addresses)
  - Keyword detection (urgency, financial, security, deceptive words)
  - Character analysis (special characters, entropy, uppercase ratio)
  - Length and word count metrics
- **Model Interpretability**: SHAP analysis for feature importance
- **Batch & Single Predictions**: Support for both individual and bulk email analysis

## 📋 Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

**Dependencies:**
- pandas
- numpy
- scikit-learn
- xgboost
- shap
- tldextract
- joblib
- matplotlib

## 🗂️ Project Structure

```
XGBOOST_MODEL/
├── Enron.csv                      # Training dataset (Enron email corpus)
├── train_text_phishing.py         # Main training script
├── feature_extraction_text.py     # Feature engineering module
├── predict_phishing.py            # Prediction interface
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
```

## 🚀 Quick Start

### 1. Training the Model

Train the XGBoost classifier on the Enron dataset:

```bash
python train_text_phishing.py
```

This will:
- Load and preprocess the Enron email dataset
- Extract text-based features from subject and body
- Train an XGBoost model with hyperparameter tuning
- Evaluate performance on validation and test sets
- Save the trained model as `phishing_text_model.joblib`
- Generate SHAP analysis plots for feature importance

### 2. Making Predictions

#### Single Email Prediction

```python
from predict_phishing import predict_email

result = predict_email(
    subject="URGENT: Verify your account now!",
    body="Click here to verify your account or it will be suspended."
)

print(f"Is Phishing: {result['is_phishing']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probability: {result['phishing_probability']:.2%}")
```

#### Batch Prediction

```python
from predict_phishing import predict_batch

predict_batch(
    csv_path="emails.csv",
    output_path="predictions.csv"
)
```

#### Command-line Interactive Mode

```bash
python predict_phishing.py
```

## 📊 Model Performance

The model achieves strong performance metrics:
- **Accuracy**: High overall classification accuracy
- **Precision/Recall**: Balanced detection of phishing emails
- **ROC-AUC**: Excellent discriminative ability
- **F1-Score**: Strong harmonic mean of precision and recall

Performance metrics are displayed during training and saved with the model.

## 🔍 Feature Extraction

The system extracts multiple categories of features:

### URL Features
- Total URL count
- Suspicious domain detection
- IP address usage
- URL entropy

### Keyword Features
- Urgency indicators (urgent, immediately, action required)
- Financial terms (account, bank, payment)
- Security keywords (password, verify, confirm)
- Deceptive phrases (click here, dear customer, winner)

### Text Characteristics
- Length metrics (subject, body, total)
- Word counts
- Special character ratios
- Uppercase character ratio
- Text entropy (randomness measure)

## 🛠️ Model Architecture

- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Pipeline**: StandardScaler → XGBClassifier
- **Hyperparameters**: Tuned for optimal performance
  - Learning rate, max depth, n_estimators
  - Subsample and colsample ratios
  - Scale_pos_weight for imbalanced data
- **Threshold**: Optimized using precision-recall curve

## 📈 SHAP Analysis

The project includes SHAP (SHapley Additive exPlanations) analysis to understand:
- Which features contribute most to predictions
- How different features impact individual predictions
- Global feature importance across all samples

SHAP plots are automatically generated during training.

## 💾 Saved Model

The trained model is saved as `phishing_text_model.joblib` containing:
- Trained pipeline (scaler + classifier)
- Optimal threshold for classification
- Training metadata

## 📝 Dataset

The project uses the **Enron email dataset** which contains:
- Legitimate emails from Enron corporation
- Labeled phishing examples
- Columns: `subject`, `body`, `label` (0=legitimate, 1=phishing)

## 🔒 Use Cases

- **Email Security**: Filter incoming emails for phishing attempts
- **User Protection**: Warn users about suspicious emails
- **Security Auditing**: Analyze email logs for threats
- **Research**: Study phishing patterns and detection techniques

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional feature engineering
- Alternative ML algorithms
- Real-time detection capabilities
- Integration with email clients

## 📄 License

This project is available for educational and research purposes.

## ⚠️ Disclaimer

This tool is for educational and research purposes. Always use multiple layers of security when dealing with potential phishing threats.