from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from pydantic import BaseModel
from predict_phishing import predict_email
import uvicorn
import requests
import json
from datetime import datetime
import os
import re
import hashlib
from urllib.parse import urlparse
import tldextract
import sys
import pandas as pd
import io

app = FastAPI(title="Phishing Detection API", version="1.0")

# Splunk HEC Configuration
SPLUNK_HEC_URL = os.getenv("SPLUNK_HEC_URL", "https://your-splunk-server:8088/services/collector")
SPLUNK_HEC_TOKEN = os.getenv("SPLUNK_HEC_TOKEN", "your-hec-token-here")
SPLUNK_INDEX = os.getenv("SPLUNK_INDEX", "phishing_detection")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0")

class EmailRequest(BaseModel):
    subject: str
    body: str

class PredictionResponse(BaseModel):
    is_phishing: bool
    phishing_probability: float
    confidence: float
    label: str

# ---------------- Enrichment helpers (privacy-preserving) ---------------- #
def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    from math import log2
    probs = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * log2(p) for p in probs if p > 0)

URL_REGEX = re.compile(r'(https?://[^\s\'"<>)]+|www\.[^\s\'"<>)]+)', re.IGNORECASE)
IP_HOST_REGEX = re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}$')
SHORTENERS = {
    "bit.ly", "t.co", "goo.gl", "ow.ly", "tinyurl.com", "is.gd",
    "buff.ly", "cutt.ly", "rb.gy", "soo.gd", "rebrand.ly", "bl.ink"
}
URGENT_TERMS = {"urgent", "immediately", "verify", "suspended", "locked", "action required", "limited time", "within 24 hours"}
CREDENTIAL_TERMS = {"password", "account", "login", "credentials", "confirm", "reset"}
FINANCIAL_TERMS = {"invoice", "payment", "bank", "wire", "transfer", "refund"}

def extract_indicators(subject: str, body: str) -> dict:
    text = f"{subject or ''}\n{body or ''}"
    urls = URL_REGEX.findall(text)
    domains = set()
    tlds = set()
    has_ip_url = False
    has_shortener = False

    for u in urls:
        u = u if u.lower().startswith(("http://", "https://")) else f"http://{u}"
        p = urlparse(u)
        host = p.netloc.split("@")[-1].split(":")[0]
        if IP_HOST_REGEX.match(host):
            has_ip_url = True
        ext = tldextract.extract(host)
        reg_domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        if reg_domain:
            rd = reg_domain.lower()
            domains.add(rd)
            if ext.suffix:
                tlds.add(ext.suffix.lower())
            if rd in SHORTENERS:
                has_shortener = True

    exclam = text.count("!")
    upper = sum(1 for c in text if c.isupper())
    letters = sum(1 for c in text if c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    non_ascii = any(ord(c) > 127 for c in text)

    lower_text = text.lower()
    urgent_count = sum(1 for t in URGENT_TERMS if t in lower_text)
    cred_count = sum(1 for t in CREDENTIAL_TERMS if t in lower_text)
    fin_count = sum(1 for t in FINANCIAL_TERMS if t in lower_text)

    indicators = {
        "subject_length": len(subject or ""),
        "body_length": len(body or ""),
        "num_urls": len(urls),
        "domains": sorted(list(domains))[:10],
        "tlds": sorted(list(tlds))[:10],
        "has_url_shortener": has_shortener,
        "has_ip_in_url": has_ip_url,
        "exclamation_count": exclam,
        "uppercase_ratio": round((upper / letters), 4) if letters else 0.0,
        "digit_ratio": round((digits / max(1, len(text))), 4),
        "body_entropy": round(shannon_entropy(body or ""), 4),
        "contains_non_ascii": non_ascii,
        "urgent_term_count": urgent_count,
        "credential_term_count": cred_count,
        "financial_term_count": fin_count,
        # Privacy-safe fingerprint of content
        "body_sha256": hashlib.sha256((body or "").encode("utf-8")).hexdigest(),
        "subject_sha256": hashlib.sha256((subject or "").encode("utf-8")).hexdigest(),
    }
    indicators["ioc_preview"] = {
        "sample_domains": indicators["domains"][:3],
        "sample_tlds": indicators["tlds"][:3],
    }
    return indicators

def send_to_splunk(event_data):
    """
    Send event data to Splunk via HTTP Event Collector (HEC)
    
    Args:
        event_data: Dictionary containing event information
    """
    try:
        # Prepare Splunk event
        splunk_event = {
            "time": datetime.now().timestamp(),
            "host": "phishing-detection-api",
            "source": "phishing_api",
            "sourcetype": "phishing_prediction",
            "index": SPLUNK_INDEX,
            "event": event_data
        }
        
        headers = {
            "Authorization": f"Splunk {SPLUNK_HEC_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Send to Splunk
        response = requests.post(
            SPLUNK_HEC_URL,
            headers=headers,
            data=json.dumps(splunk_event),
            verify=False  # Set to True in production with proper SSL cert
        )
        
        if response.status_code == 200:
            print(f"✓ Event sent to Splunk successfully")
        else:
            print(f"✗ Failed to send to Splunk: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"✗ Error sending to Splunk: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(email: EmailRequest, request: Request):
    """
    Predict if an email is phishing and log to Splunk
    """
    try:
        # Get prediction
        result = predict_email(email.subject, email.body)

        # Enrich with indicators (no raw body)
        indicators = extract_indicators(email.subject, email.body)
        client_ip = request.client.host if request and request.client else None

        # Prepare event data for Splunk
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "action": "phishing_prediction",
            "client_ip": client_ip,
            "model_version": MODEL_VERSION,
            "email_subject": email.subject,
            "email_body_length": len(email.body or ""),
            "prediction": result["label"],
            "is_phishing": result["is_phishing"],
            "phishing_probability": result["phishing_probability"],
            "confidence": result["confidence"],
            "severity": "HIGH" if result["is_phishing"] and result["confidence"] > 0.7 else ("MEDIUM" if result["is_phishing"] else "LOW"),
            "features": indicators,
        }

        # Send to Splunk
        send_to_splunk(event_data)
        
        return result
        
    except Exception as e:
        # Log error to Splunk
        error_event = {
            "timestamp": datetime.now().isoformat(),
            "action": "prediction_error",
            "error": str(e),
            "model_version": MODEL_VERSION,
            "email_subject": email.subject,
            "email_subject_sha256": hashlib.sha256((email.subject or "").encode("utf-8")).hexdigest()
        }
        send_to_splunk(error_event)
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload CSV file with 'subject' and 'body' columns for batch prediction
    Returns predictions for all emails and sends alerts to Splunk
    """
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
            
            # Get prediction
            result = predict_email(subject, body)
            
            # Enrich with indicators
            indicators = extract_indicators(subject, body)
            
            # Build result for this email
            email_result = {
                "row_index": int(idx),
                "subject": subject,
                "is_phishing": result["is_phishing"],
                "phishing_probability": result["phishing_probability"],
                "confidence": result["confidence"],
                "label": result["label"]
            }
            
            results.append(email_result)
            
            if result["is_phishing"]:
                phishing_count += 1
            
            # Prepare event data for Splunk
            event_data = {
                "timestamp": datetime.now().isoformat(),
                "action": "csv_batch_prediction",
                "row_index": int(idx),
                "model_version": MODEL_VERSION,
                "email_subject": subject,
                "email_body_length": len(body or ""),
                "prediction": result["label"],
                "is_phishing": result["is_phishing"],
                "phishing_probability": result["phishing_probability"],
                "confidence": result["confidence"],
                "severity": "HIGH" if result["is_phishing"] and result["confidence"] > 0.7 else ("MEDIUM" if result["is_phishing"] else "LOW"),
                "features": indicators,
            }
            
            # Send to Splunk
            send_to_splunk(event_data)
        
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

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}

if __name__ == "__main__":
    # Allow custom port via command line argument
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    print("Starting Phishing Detection API Server...")
    print(f"API available at: http://localhost:{port}")
    print(f"API Docs at: http://localhost:{port}/docs")
    print(f"Splunk HEC URL: {SPLUNK_HEC_URL}")
    print(f"Splunk Index: {SPLUNK_INDEX}")
    uvicorn.run(app, host="0.0.0.0", port=port)