# Text-based feature extraction for phishing email detection
# Analyzes subject and body content for suspicious patterns

import re
import math
import pandas as pd
import numpy as np
from urllib.parse import urlparse

# Suspicious keywords commonly found in phishing emails (expanded)
URGENCY_WORDS = {"urgent", "immediately", "action required", "verify", "suspend", "alert", "warning", "expire", 
                 "limited time", "act now", "expires", "expiration", "deadline", "asap", "right away", "hurry"}
FINANCIAL_WORDS = {"account", "bank", "credit", "payment", "paypal", "transaction", "refund", "invoice", 
                   "billing", "card", "debit", "visa", "mastercard", "wire", "transfer", "money", "cash", 
                   "reward", "bonus", "paycheck", "salary", "tax", "irs", "refund"}
SECURITY_WORDS = {"password", "security", "confirm", "verify", "update", "validate", "unauthorized", 
                  "breach", "compromised", "locked", "blocked", "suspended", "restricted", "credential",
                  "authentication", "access", "login", "signin", "sign-in"}
DECEPTIVE_WORDS = {"click here", "click below", "dear customer", "dear user", "congratulations", "winner", "prize",
                   "free", "gift", "claim", "won", "lucky", "selected", "chance", "offer", "promotion"}
BRAND_WORDS = {"paypal", "amazon", "ebay", "netflix", "apple", "microsoft", "google", "facebook", 
               "bank of america", "wells fargo", "chase", "citibank"}

def extract_urls_from_text(text):
    """Extract all URLs from text"""    
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, str(text))

def count_pattern_matches(text, pattern_set):
    """Count occurrences of patterns in text (case insensitive)"""
    text_lower = str(text).lower()
    return sum(1 for pattern in pattern_set if pattern in text_lower)

def entropy(s):
    """Calculate Shannon entropy of string"""
    if not s or len(s) == 0:
        return 0.0
    probs = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def extract_text_features(subject, body):
    """Extract features from email subject and body"""
    subject = str(subject) if pd.notna(subject) else ""
    body = str(body) if pd.notna(body) else ""
    combined_text = subject + " " + body
    
    feats = {}
    
    # Length features
    feats["subject_length"] = len(subject)
    feats["body_length"] = len(body)
    feats["total_length"] = len(combined_text)
    
    # Word count features
    feats["subject_word_count"] = len(subject.split())
    feats["body_word_count"] = len(body.split())
    
    # Character features
    feats["exclamation_count"] = combined_text.count("!")
    feats["question_count"] = combined_text.count("?")
    feats["caps_ratio"] = sum(1 for c in combined_text if c.isupper()) / max(1, len(combined_text))
    feats["digit_count"] = sum(1 for c in combined_text if c.isdigit())
    feats["special_char_count"] = len(re.findall(r'[^a-zA-Z0-9\s]', combined_text))
    
    # Entropy (randomness - phishing may have random strings)
    feats["subject_entropy"] = entropy(subject)
    feats["body_entropy"] = entropy(body)
    
    # Suspicious keyword counts
    feats["urgency_words"] = count_pattern_matches(combined_text, URGENCY_WORDS)
    feats["financial_words"] = count_pattern_matches(combined_text, FINANCIAL_WORDS)
    feats["security_words"] = count_pattern_matches(combined_text, SECURITY_WORDS)
    feats["deceptive_words"] = count_pattern_matches(combined_text, DECEPTIVE_WORDS)
    feats["brand_words"] = count_pattern_matches(combined_text, BRAND_WORDS)
    
    # Combined suspicious score
    feats["total_suspicious_words"] = (feats["urgency_words"] + feats["financial_words"] + 
                                       feats["security_words"] + feats["deceptive_words"])
    
    # Ratio features (normalized by length)
    feats["suspicious_ratio"] = feats["total_suspicious_words"] / max(1, feats["subject_word_count"] + feats["body_word_count"])
    feats["urgency_ratio"] = feats["urgency_words"] / max(1, feats["subject_word_count"])
    feats["financial_ratio"] = feats["financial_words"] / max(1, feats["body_word_count"])
    
    # URL analysis
    urls = extract_urls_from_text(combined_text)
    feats["url_count"] = len(urls)
    feats["has_url"] = 1 if urls else 0
    
    # Analyze URLs if present
    if urls:
        feats["avg_url_length"] = np.mean([len(url) for url in urls])
        feats["max_url_length"] = max([len(url) for url in urls])
        
        # Check for IP addresses in URLs
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        feats["url_has_ip"] = 1 if any(re.search(ip_pattern, url) for url in urls) else 0
        
        # Check for HTTPS
        feats["https_ratio"] = sum(1 for url in urls if url.startswith("https")) / len(urls)
        
        # Count suspicious TLDs
        suspicious_tlds = {".xyz", ".tk", ".ml", ".ga", ".cf", ".gq"}
        feats["suspicious_tld_count"] = sum(1 for url in urls if any(tld in url for tld in suspicious_tlds))
    else:
        feats["avg_url_length"] = 0
        feats["max_url_length"] = 0
        feats["url_has_ip"] = 0
        feats["https_ratio"] = 0
        feats["suspicious_tld_count"] = 0
    
    # Email-specific patterns (enhanced)
    feats["has_dear_pattern"] = 1 if re.search(r'\bdear\s+(customer|user|member|friend|valued)\b', combined_text.lower()) else 0
    feats["has_click_pattern"] = 1 if re.search(r'\bclick\s+(here|below|link|now|this)\b', combined_text.lower()) else 0
    feats["has_verify_pattern"] = 1 if re.search(r'\b(verify|confirm|update|validate)\s+(your|account|information|password|identity|details)\b', combined_text.lower()) else 0
    feats["has_suspend_pattern"] = 1 if re.search(r'\b(suspend|lock|block|restrict|disable|close)\s+(your|account|access)\b', combined_text.lower()) else 0
    feats["has_reset_password"] = 1 if re.search(r'\b(reset|change|update)\s+(your\s+)?(password|credentials)\b', combined_text.lower()) else 0
    
    # Urgency patterns in subject
    feats["subject_has_urgency"] = 1 if any(word in subject.lower() for word in ["urgent", "immediately", "asap", "expire", "alert"]) else 0
    feats["subject_has_re_fwd"] = 1 if re.search(r'^(re:|fwd?:|fw:)', subject.lower().strip()) else 0
    
    # Spelling/grammar indicators (expanded)
    common_misspellings = ["recieve", "untill", "occured", "seperate", "definately", "priviledge", "beleive"]
    feats["misspelling_count"] = sum(1 for word in common_misspellings if word in combined_text.lower())
    
    # HTML/formatting tags (phishing often uses HTML)
    feats["html_tag_count"] = len(re.findall(r'<[^>]+>', combined_text))
    feats["has_html"] = 1 if feats["html_tag_count"] > 0 else 0
    
    # Hyperlink patterns
    feats["has_hidden_link"] = 1 if re.search(r'<a\s+[^>]*href', combined_text.lower()) else 0
    
    # Phone number patterns (phishing often includes phone numbers)
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'
    feats["phone_count"] = len(re.findall(phone_pattern, combined_text))
    
    # Email address patterns (beyond sender)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    feats["email_address_count"] = len(re.findall(email_pattern, combined_text))
    
    # ALL CAPS words (shouting = suspicious)
    words = combined_text.split()
    feats["caps_word_count"] = sum(1 for word in words if len(word) > 2 and word.isupper())
    feats["caps_word_ratio"] = feats["caps_word_count"] / max(1, len(words))
    
    return feats

def features_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from DataFrame with 'subject' and 'body' columns
    
    Args:
        df: DataFrame with columns 'subject' and 'body'
    
    Returns:
        DataFrame with extracted features
    """
    # Extract features for each row
    features_list = []
    for idx, row in df.iterrows():
        subject = row.get('subject', '')
        body = row.get('body', '')
        features_list.append(extract_text_features(subject, body))
    
    return pd.DataFrame(features_list)
