"""Benchmark inference performance for Random Forest phishing model."""
import time
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from feature_extraction_rf import features_from_dataframe

MODEL_PATH = "checkpoints/phishing_detector/rf_phishing_detector.joblib"


def percentile(arr, p):
    if len(arr) == 0:
        return None
    return float(np.percentile(arr, p))


def main():
    print("=" * 60)
    print("RANDOM FOREST INFERENCE BENCHMARK")
    print("=" * 60)

    # Load dataset and extract features
    df = pd.read_csv("Enron.csv").dropna(subset=["subject", "body", "label"]).copy()
    y = df["label"].astype(int)
    
    # Use same split to approximate test set
    _, idx_holdout = train_test_split(df.index, test_size=0.2, stratify=y, random_state=42)
    df_test = df.loc[idx_holdout]
    
    print("\n[1] Extracting features from test set...")
    X_test = features_from_dataframe(df_test[["subject", "body"]])
    
    # Load model
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_names = model_data["feature_names"]
    
    # Ensure feature order
    X_test_ordered = X_test[feature_names]
    X_test_scaled = scaler.transform(X_test_ordered)

    # Warmup
    _ = model.predict_proba(X_test_scaled[:32])

    # Single-sample latency (ms)
    print("\n[2] Measuring single-sample latency...")
    single_latencies = []
    n_samples = min(200, len(X_test_scaled))
    for i in range(n_samples):
        x = X_test_scaled[i:i+1]
        t0 = time.perf_counter()
        _ = model.predict_proba(x)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        single_latencies.append(dt_ms)

    # Throughput benchmarking
    print("\n[3] Measuring batch throughput...")
    batch_sizes = [1, 8, 32, 128]
    throughput = {}
    for bs in batch_sizes:
        total = 0
        t0 = time.perf_counter()
        for start in range(0, len(X_test_scaled), bs):
            batch = X_test_scaled[start:start+bs]
            _ = model.predict_proba(batch)
            total += len(batch)
        dt = time.perf_counter() - t0
        eps = total / dt if dt > 0 else float('inf')
        throughput[str(bs)] = float(eps)

    report = {
        "model_path": MODEL_PATH,
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

    with open("rf_benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\n[4] Benchmark report written to rf_benchmark_report.json")
    print(f"   Mean latency: {report['latency_ms']['mean']:.2f}ms")
    print(f"   P95 latency: {report['latency_ms']['p95']:.2f}ms")
    print(f"   Throughput (batch=32): {throughput['32']:.1f} emails/sec")


if __name__ == "__main__":
    main()
