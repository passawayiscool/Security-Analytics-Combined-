"""
Inference benchmarking for phishing text model.
Measures single-sample latency and batch throughput on the holdout set.
"""

import time
import json
import math
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from feature_extraction_text import features_from_dataframe


def percentile(arr, p):
    if len(arr) == 0:
        return None
    return float(np.percentile(arr, p))


def main():
    model_path = "phishing_text_model.joblib"
    df = pd.read_csv("Enron.csv").dropna(subset=["subject", "body", "label"]).copy()
    y = df["label"].astype(int)

    # Use same split parameters as training to approximate the holdout
    idx_train, idx_holdout = train_test_split(
        df.index, test_size=0.2, stratify=y, random_state=42
    )
    df_holdout = df.loc[idx_holdout]

    # Extract baseline features for holdout
    X_holdout = features_from_dataframe(df_holdout[["subject", "body"]])

    model_data = joblib.load(model_path)
    pipeline = model_data["pipeline"]

    # Warmup
    _ = pipeline.predict_proba(X_holdout.iloc[:32])

    # Single-sample latency (ms)
    single_latencies = []
    n_samples = min(200, len(X_holdout))
    for i in range(n_samples):
        x = X_holdout.iloc[[i]]
        t0 = time.perf_counter()
        _ = pipeline.predict_proba(x)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        single_latencies.append(dt_ms)

    # Throughput (emails/sec) across batch sizes
    batch_sizes = [1, 8, 32, 128]
    throughput = {}
    for bs in batch_sizes:
        total = 0
        t0 = time.perf_counter()
        for start in range(0, len(X_holdout), bs):
            batch = X_holdout.iloc[start : start + bs]
            _ = pipeline.predict_proba(batch)
            total += len(batch)
        dt = time.perf_counter() - t0
        eps = total / dt if dt > 0 else math.inf
        throughput[str(bs)] = float(eps)

    report = {
        "samples_benchmarked": int(n_samples),
        "latency_ms": {
            "mean": float(np.mean(single_latencies)) if single_latencies else None,
            "p50": percentile(single_latencies, 50),
            "p90": percentile(single_latencies, 90),
            "p95": percentile(single_latencies, 95),
            "p99": percentile(single_latencies, 99),
            "max": float(np.max(single_latencies)) if single_latencies else None,
        },
        "throughput_eps": throughput,
    }

    with open("benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Benchmark report written to benchmark_report.json")


if __name__ == "__main__":
    main()
