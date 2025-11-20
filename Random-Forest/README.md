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

### Step 1: Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.bat
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Train Model
```powershell
python train_rf_phishing.py
```
This creates `checkpoints/phishing_detector/rf_phishing_detector.joblib`

### Step 4: Configure Splunk 
Edit `.env` file and add your Splunk credentials:
```bash
SPLUNK_HEC_URL=https://your-splunk.com:8088/services/collector
SPLUNK_TOKEN=your-hec-token-here
SPLUNK_INDEX=security
```

### Step 5: Start API Server
```powershell
python api_server_fastapi.py 8001
```
**Note:** Port 8001 is used because Splunk typically runs on port 8000. You can specify any available port as an argument.

- API Docs: http://localhost:8001/docs
- Alerts auto-send to Splunk when phishing detected

### Step 6: Test Prediction

#### Example: Clearly Phishing Email
```powershell
$body = @{
    subject = "URGENT: Your Account Will Be Closed!"
    body = "Dear Customer, Your account has been suspended due to unusual activity. Click here immediately to verify your identity and restore access: http://secure-banking-verify.tk/login.php?user=confirm. You have 24 hours before permanent deletion. Enter your password and SSN to continue."
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8001/predict" -Method POST -Body $body -ContentType "application/json"
```

#### Example: Clearly Legitimate Email
```powershell
$body = @{
    subject = "Team Meeting Notes - Q4 Planning"
    body = "Hi Team, Thanks for attending today's planning meeting. Here are the key takeaways: 1) Q4 goals approved, 2) New hire onboarding starts Monday, 3) Budget review next Friday at 2pm in Conference Room B. Please review the attached slides and send feedback by EOD Thursday. Best regards, Sarah"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8001/predict" -Method POST -Body $body -ContentType "application/json"
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

Invoke-RestMethod -Uri "http://localhost:8001/predict" -Method POST -Body $body -ContentType "application/json"
```

**Batch Prediction:**
```powershell
$body = @{
    emails = @(
        @{subject="Alert"; body="Click here"},
        @{subject="Meeting"; body="Tomorrow at 2pm"}
    )
} | ConvertTo-Json -Depth 3

Invoke-RestMethod -Uri "http://localhost:8001/predict/batch" -Method POST -Body $body -ContentType "application/json"
```

**CSV File Upload (Bulk Processing):**
```powershell
# Upload CSV file with 'subject' and 'body' columns
curl.exe -X POST "http://localhost:8001/predict/csv" -F "file=@test_emails.csv"
```

CSV file format:
```csv
subject,body
"URGENT: Verify Account","Click here immediately: http://phishing.tk"
"Team Meeting","Hi everyone, meeting at 2pm in Conference Room B"
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
