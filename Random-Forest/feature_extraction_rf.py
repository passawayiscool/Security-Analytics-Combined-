"""
Feature extraction for Random Forest phishing detection.
Consolidated for cleaner project structure (no nested folders).
"""
import email
from email import policy
from email.parser import Parser
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd


class EmailFeatureExtractor:
    """Extract numerical features from email text for Random Forest classifier"""

    def __init__(self):
        self.parser = Parser(policy=policy.default)
        self.suspicious_tlds = {'.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.click', '.download', '.loan'}
        self.url_shorteners = {'bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly', 'buff.ly', 'is.gd'}
        self.urgency_keywords = [
            'urgent', 'immediate', 'action required', 'suspended', 'verify', 'confirm',
            'update', 'expire', 'limited time', 'act now', 'click here', 'within 24 hours'
        ]
        self.financial_keywords = [
            'bank', 'account', 'credit card', 'payment', 'paypal', 'invoice', 'refund',
            'transaction', 'wire transfer', 'tax', 'irs', 'security question', 'ssn'
        ]
        self.credential_keywords = [
            'password', 'username', 'login', 'signin', 'credentials', 'verify identity',
            'social security', 'personal information', 'account details'
        ]

    def parse_email(self, email_text: str) -> dict:
        """Parse email into structured components"""
        try:
            if email_text.startswith('From '):
                msg = self.parser.parsestr(email_text)
            else:
                msg = email.message.EmailMessage()
                msg.set_content(email_text)

            return {
                'subject': msg.get('Subject', ''),
                'sender': msg.get('From', ''),
                'body': self._extract_body(msg),
                'headers': dict(msg.items()),
                'attachments': self._check_attachments(msg)
            }
        except:
            return {
                'subject': '',
                'sender': '',
                'body': email_text,
                'headers': {},
                'attachments': []
            }

    def _extract_body(self, msg) -> str:
        """Extract email body"""
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except:
                        body += str(part.get_payload())
                elif content_type == "text/html":
                    try:
                        html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        body += BeautifulSoup(html_content, 'html.parser').get_text()
                    except:
                        body += str(part.get_payload())
        else:
            try:
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                body = str(msg.get_payload())
        return body.strip()

    def _check_attachments(self, msg) -> list:
        """Check for attachments"""
        attachments = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_disposition() == 'attachment':
                    filename = part.get_filename()
                    if filename:
                        attachments.append(filename)
        return attachments

    def extract_features(self, email_text: str) -> dict:
        """Extract all features"""
        parsed = self.parse_email(email_text)
        features = {}
        features.update(self._extract_url_features(parsed))
        features.update(self._extract_sender_features(parsed))
        features.update(self._extract_content_features(parsed))
        features.update(self._extract_structural_features(parsed))
        features.update(self._extract_linguistic_features(parsed))
        return features

    def _extract_url_features(self, parsed: dict) -> dict:
        body = parsed['body'].lower()
        features = {}
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, body + parsed['subject'].lower())

        features['url_count'] = len(urls)
        features['has_urls'] = 1.0 if len(urls) > 0 else 0.0

        suspicious_tld_count = sum(1 for url in urls for tld in self.suspicious_tlds if tld in url)
        features['suspicious_tld_count'] = suspicious_tld_count
        features['has_suspicious_tld'] = 1.0 if suspicious_tld_count > 0 else 0.0

        shortener_count = sum(1 for url in urls for shortener in self.url_shorteners if shortener in url)
        features['url_shortener_count'] = shortener_count
        features['has_url_shortener'] = 1.0 if shortener_count > 0 else 0.0

        ip_pattern = r'http[s]?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        ip_urls = re.findall(ip_pattern, body + parsed['subject'].lower())
        features['has_ip_address_url'] = 1.0 if len(ip_urls) > 0 else 0.0
        features['ip_address_url_count'] = len(ip_urls)

        at_in_url = sum(1 for url in urls if '@' in url)
        features['at_symbol_in_url'] = 1.0 if at_in_url > 0 else 0.0

        if len(urls) > 0:
            features['avg_url_length'] = np.mean([len(url) for url in urls])
            features['max_url_length'] = max(len(url) for url in urls)
        else:
            features['avg_url_length'] = 0.0
            features['max_url_length'] = 0.0

        href_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>'
        mismatches = sum(1 for match in re.finditer(href_pattern, body, re.IGNORECASE)
                        if 'http' in match.group(2) and match.group(1) != match.group(2))
        features['url_text_mismatch'] = mismatches
        return features

    def _extract_sender_features(self, parsed: dict) -> dict:
        features = {}
        sender = parsed['sender'].lower()
        features['has_sender'] = 1.0 if sender else 0.0

        domain_match = re.search(r'@([a-zA-Z0-9.-]+)', sender)
        if domain_match:
            domain = domain_match.group(1)
            features['sender_domain_length'] = len(domain)
            features['sender_has_numbers'] = 1.0 if re.search(r'\d', domain) else 0.0
            features['sender_has_hyphens'] = 1.0 if '-' in domain else 0.0
            features['sender_subdomain_count'] = domain.count('.') - 1
            legitimate_domains = {'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'aol.com'}
            features['sender_is_common_provider'] = 1.0 if domain in legitimate_domains else 0.0
        else:
            features['sender_domain_length'] = 0
            features['sender_has_numbers'] = 0.0
            features['sender_has_hyphens'] = 0.0
            features['sender_subdomain_count'] = 0
            features['sender_is_common_provider'] = 0.0

        reply_to = parsed['headers'].get('Reply-To', '').lower()
        features['reply_to_mismatch'] = 1.0 if (reply_to and sender and reply_to != sender) else 0.0
        return features

    def _extract_content_features(self, parsed: dict) -> dict:
        features = {}
        text = (parsed['subject'] + ' ' + parsed['body']).lower()

        urgency_count = sum(text.count(keyword) for keyword in self.urgency_keywords)
        features['urgency_keyword_count'] = urgency_count
        features['has_urgency_keywords'] = 1.0 if urgency_count > 0 else 0.0

        financial_count = sum(text.count(keyword) for keyword in self.financial_keywords)
        features['financial_keyword_count'] = financial_count
        features['has_financial_keywords'] = 1.0 if financial_count > 0 else 0.0

        credential_count = sum(text.count(keyword) for keyword in self.credential_keywords)
        features['credential_keyword_count'] = credential_count
        features['has_credential_keywords'] = 1.0 if credential_count > 0 else 0.0

        generic_greetings = ['dear customer', 'dear user', 'dear member', 'valued customer']
        features['has_generic_greeting'] = 1.0 if any(greeting in text for greeting in generic_greetings) else 0.0

        pii_requests = ['social security', 'ssn', 'date of birth', 'mother\'s maiden name', 'account number']
        features['requests_pii'] = 1.0 if any(request in text for request in pii_requests) else 0.0

        features['exclamation_count'] = text.count('!')
        features['has_excessive_exclamation'] = 1.0 if text.count('!') > 3 else 0.0

        words = text.split()
        if len(words) > 0:
            caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
            features['caps_word_ratio'] = caps_words / len(words)
        else:
            features['caps_word_ratio'] = 0.0
        return features

    def _extract_structural_features(self, parsed: dict) -> dict:
        features = {}
        body = parsed['body']
        features['body_length'] = len(body)
        features['subject_length'] = len(parsed['subject'])
        features['word_count'] = len(body.split())
        features['has_html'] = 1.0 if '<html' in body.lower() or '<body' in body.lower() else 0.0
        features['attachment_count'] = len(parsed['attachments'])
        features['has_attachments'] = 1.0 if len(parsed['attachments']) > 0 else 0.0

        suspicious_extensions = ['.exe', '.scr', '.bat', '.cmd', '.com', '.pif', '.vbs', '.js']
        suspicious_attachment_count = sum(1 for attachment in parsed['attachments']
                                         if any(attachment.lower().endswith(ext) for ext in suspicious_extensions))
        features['suspicious_attachment_count'] = suspicious_attachment_count
        features['has_subject'] = 1.0 if parsed['subject'] else 0.0
        features['has_html_form'] = 1.0 if '<form' in body.lower() else 0.0
        features['has_javascript'] = 1.0 if '<script' in body.lower() or 'javascript:' in body.lower() else 0.0
        return features

    def _extract_linguistic_features(self, parsed: dict) -> dict:
        features = {}
        text = parsed['subject'] + ' ' + parsed['body']

        features['repeated_char_sequences'] = len(re.findall(r'(.)\1{2,}', text))
        leet_speak_pattern = r'\b\w*[0-9]+[a-zA-Z]+[0-9]+\w*\b'
        features['leet_speak_count'] = len(re.findall(leet_speak_pattern, text))

        if len(text) > 0:
            punctuation_count = sum(text.count(p) for p in '!?.,;:')
            features['punctuation_density'] = punctuation_count / len(text)
        else:
            features['punctuation_density'] = 0.0

        words = text.split()
        features['avg_word_length'] = np.mean([len(word) for word in words]) if len(words) > 0 else 0.0
        features['has_non_ascii'] = 1.0 if any(ord(char) > 127 for char in text) else 0.0

        sentences = re.split(r'[.!?]+', text)
        features['sentence_count'] = len([s for s in sentences if len(s.strip()) > 0])
        return features


def features_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from DataFrame with 'subject' and 'body' columns"""
    extractor = EmailFeatureExtractor()
    all_features = []
    
    for _, row in df.iterrows():
        email_text = f"Subject: {row.get('subject', '')}\n\n{row.get('body', '')}"
        features = extractor.extract_features(email_text)
        all_features.append(features)
    
    return pd.DataFrame(all_features)


def features_from_text(email_text: str) -> dict:
    """Return feature dict for a single email."""
    extractor = EmailFeatureExtractor()
    return extractor.extract_features(email_text)


def features_from_list(emails):
    """Return list of feature dicts for multiple emails."""
    return [features_from_text(e) for e in emails]


if __name__ == "__main__":
    sample = "Subject: Account Alert\n\nURGENT: Your account will be suspended if you do not verify immediately at http://verify-login.xyz"
    feats = features_from_text(sample)
    print("Extracted feature count:", len(feats))
    for k in sorted(list(feats.keys()))[:15]:
        print(f"{k}: {feats[k]}")
