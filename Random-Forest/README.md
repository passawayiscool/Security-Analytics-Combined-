# Random Forest Phishing Detection

Clean, script-centric workflow matching the XGBoost project structure.

## Core Scripts

| Script | Purpose |
|--------|---------|
| `train_rf_phishing.py` | Train Random Forest model from CSV |
| `predict_rf_phishing.py` | Single + batch email prediction |
| `evaluate_rf_benchmark.py` | Inference latency + throughput benchmark |
| `feature_extraction_rf.py` | Feature extraction utilities |
| `robustness_eval_rf.py` | Perturbation robustness evaluation |
| `api_server_fastapi.py` | FastAPI REST server |

## Quick Start

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Train Model
```powershell
python train_rf_phishing.py
```
This creates `checkpoints/phishing_detector/rf_phishing_detector.joblib`

### Step 3: Configure Splunk (Optional)
Edit `.env` file and add your Splunk credentials:
```bash
SPLUNK_HEC_URL=https://your-splunk.com:8088/services/collector
SPLUNK_TOKEN=your-hec-token-here
SPLUNK_INDEX=security
```

### Step 4: Start API Server
```powershell
python api_server_fastapi.py
```
- API Docs: http://localhost:8000/docs
- Alerts auto-send to Splunk when phishing detected

### Step 5: Test Prediction
```powershell
# PowerShell
$body = @{
    subject = "URGENT Account Suspended"
    body = "Click here to verify: http://phishing-site.tk"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -Body $body -ContentType "application/json"
```

### Optional: Benchmark & Robustness
```powershell
python evaluate_rf_benchmark.py
python robustness_eval_rf.py
```

## Requirements

```
pandas
numpy
scikit-learn
joblib
beautifulsoup4
tqdm
fastapi
uvicorn
```

Install with:
```powershell
pip install -r requirements.txt
```

## Model Output

- Model artifact: `checkpoints/phishing_detector/rf_phishing_detector.joblib`
- Contains: trained model, scaler, feature names, and metrics

## API Usage

**Predict Single Email:**
```powershell
$body = @{
    subject = "Account Alert"
    body = "Verify your account now"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -Body $body -ContentType "application/json"
```

**Batch Prediction:**
```powershell
$body = @{
    emails = @(
        @{subject="Alert"; body="Click here"},
        @{subject="Meeting"; body="Tomorrow at 2pm"}
    )
} | ConvertTo-Json -Depth 3

Invoke-RestMethod -Uri "http://localhost:8000/predict/batch" -Method POST -Body $body -ContentType "application/json"
```

## Splunk Integration

When configured, the API automatically sends alerts to Splunk for:
- Any email classified as phishing
- Any email with risk_score > 50

**What gets sent:**
- Alert ID, timestamp, severity
- Email subject, body preview, hash
- Phishing probability, confidence, risk score
- Recommended action (BLOCK/QUARANTINE/REVIEW/ALLOW)

**Splunk Query Examples:**
```spl
index=security sourcetype=phishing_detection
| stats count by event.detection.classification

index=security sourcetype=phishing_detection event.severity IN (CRITICAL, HIGH)
| table event.alert_id, event.email.subject, event.detection.risk_score
```

## Dataset

Place your dataset as `Enron.csv` with columns:
- `subject`: Email subject line
- `body`: Email body text  
- `label`: 0 for legitimate, 1 for phishing

## Notes

- Flat project structure for simplicity
- No nested folders required
- All dependencies self-contained
- Compatible with XGBoost project style
