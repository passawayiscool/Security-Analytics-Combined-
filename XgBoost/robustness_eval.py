"""
Adversarial robustness evaluation for the phishing text model.
Applies simple evasion tactics to holdout emails and reports metric degradation.
"""

import json
import random
import re
from copy import deepcopy
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from feature_extraction_text import features_from_dataframe


def choose_indices(df, y):
    idx_train, idx_holdout = train_test_split(
        df.index, test_size=0.2, stratify=y, random_state=42
    )
    return idx_holdout


def to_df(subjects, bodies):
    return pd.DataFrame({"subject": subjects, "body": bodies})


# Simple homoglyph substitution for a subset of characters
HOMO_MAP = str.maketrans({
    "a": "а",  # Latin a -> Cyrillic a
    "e": "е",  # Latin e -> Cyrillic e
    "o": "о",  # Latin o -> Cyrillic o
    "i": "і",  # Latin i -> Ukrainian i
    "c": "с",  # Latin c -> Cyrillic s
})


def homoglyph(text, rate=0.2):
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch.lower() in "aeoic" and random.random() < rate:
            repl = chr(ord(ch.lower().translate(HOMO_MAP)))
            # preserve case approximately
            chars[i] = repl.upper() if ch.isupper() else repl
    return "".join(chars)


def inject_zwsp(text, rate=0.1):
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and random.random() < rate:
            out.append("\u200b")  # zero-width space
    return "".join(out)


def html_obfuscate(text):
    # break suspicious phrases with harmless tags
    text = re.sub(r"click here", "click<b></b> here", text, flags=re.IGNORECASE)
    text = re.sub(r"verify your", "ver<i></i>ify your", text, flags=re.IGNORECASE)
    return text


def multilingual_mix(text):
    suffix = "\nPor favor verifique su cuenta. Vérifiez votre compte."
    return text + suffix


PARAPHRASE = {
    "urgent": "time-sensitive",
    "immediately": "as soon as possible",
    "verify": "confirm",
    "password": "passcode",
    "account": "profile",
}


def paraphrase(text):
    result = text
    for k, v in PARAPHRASE.items():
        result = re.sub(rf"\b{k}\b", v, result, flags=re.IGNORECASE)
    return result


def apply_tactic(df):
    subjects = df["subject"].astype(str).tolist()
    bodies = df["body"].astype(str).tolist()

    tactics = {
        "homoglyph": lambda s, b: (homoglyph(s), homoglyph(b)),
        "zero_width": lambda s, b: (inject_zwsp(s), inject_zwsp(b)),
        "html_obfuscation": lambda s, b: (html_obfuscate(s), html_obfuscate(b)),
        "multilingual_mix": lambda s, b: (multilingual_mix(s), multilingual_mix(b)),
        "paraphrase": lambda s, b: (paraphrase(s), paraphrase(b)),
    }

    transformed = {}
    for name, fn in tactics.items():
        ts = []
        tb = []
        for s, b in zip(subjects, bodies):
            s2, b2 = fn(s, b)
            ts.append(s2)
            tb.append(b2)
        transformed[name] = to_df(ts, tb)
    return transformed


def evaluate_set(pipeline, threshold, df_subset, y_true):
    X_feats = features_from_dataframe(df_subset[["subject", "body"]])
    probs = pipeline.predict_proba(X_feats)[:, 1]

    # threshold-based metrics
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    mcc = matthews_corrcoef(y_true, preds)

    # probabilistic metrics
    auc_pr = average_precision_score(y_true, probs)
    auc_roc = roc_auc_score(y_true, probs)

    return {
        "auc_pr": float(auc_pr),
        "auc_roc": float(auc_roc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1s),
        "mcc": float(mcc),
    }


def main():
    random.seed(42)
    np.random.seed(42)

    df = pd.read_csv("Enron.csv").dropna(subset=["subject", "body", "label"]).copy()
    y = df["label"].astype(int)
    hold_idx = choose_indices(df, y)
    df_hold = df.loc[hold_idx]
    y_hold = y.loc[hold_idx].values

    model = joblib.load("phishing_text_model.joblib")
    pipeline = model["pipeline"]
    threshold = model.get("threshold", 0.5)

    # Baseline (no transformation)
    baseline = evaluate_set(pipeline, threshold, df_hold, y_hold)

    # Apply tactics
    transformed_sets = apply_tactic(df_hold)
    results = {"baseline": baseline}
    for name, df_t in transformed_sets.items():
        results[name] = evaluate_set(pipeline, threshold, df_t, y_hold)

    with open("robustness_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Robustness report written to robustness_report.json")


if __name__ == "__main__":
    main()
