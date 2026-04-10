"""
Baseline 2: Historical Mean — predict the mean PM2.5 over the lookback window.

For each station, the prediction for all horizons is the average PM2.5
across the 24-hour lookback window. No learned parameters.

This baseline answers: "Does the model beat predicting the recent average?"
It is slightly more sophisticated than persistence because it smooths out
short-term fluctuations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from baselines.shared import load_and_split, fit_and_scale, compute_metrics, print_metrics


def historical_mean_predict(X_raw, target_scaler, horizon=6):
    """
    Historical mean baseline: average PM2.5 over the lookback window.

    Args:
        X_raw: (samples, input_len, nodes, features) in original ug/m3 scale
        target_scaler: fitted MinMaxScaler for targets
        horizon: forecast steps

    Returns:
        preds: (samples, horizon, nodes) in scaled [0,1] space
    """
    # PM2.5 is feature index 0, average over all timesteps in window
    mean_pm25 = X_raw[:, :, :, 0].mean(axis=1)  # (samples, nodes)

    # Repeat for all horizons
    preds_raw = np.repeat(mean_pm25[:, np.newaxis, :], horizon, axis=1)  # (samples, horizon, nodes)

    # Scale into target space
    shape = preds_raw.shape
    preds_scaled = target_scaler.transform(preds_raw.reshape(-1, 1)).reshape(shape)

    return preds_scaled.astype(np.float32)


def main():
    print("=" * 60)
    print("Baseline 2: Historical Mean")
    print("  Predicts mean PM2.5 over 24h lookback for all horizons")
    print("=" * 60)

    # Load and split (raw scale)
    train, val, test, adj = load_and_split()

    # Fit scalers
    _, _, _, _, target_scaler = fit_and_scale(train, val, test)

    X_test_raw, Y_test_raw = test

    # Generate predictions
    Y_test_scaled = target_scaler.transform(Y_test_raw.reshape(-1, 1)).reshape(Y_test_raw.shape).astype(np.float32)
    preds_scaled = historical_mean_predict(X_test_raw, target_scaler, horizon=6)

    # Compute metrics
    metrics = compute_metrics(preds_scaled, Y_test_scaled, target_scaler)
    print_metrics(metrics, "Historical Mean Baseline")

    # Validation set
    X_val_raw, Y_val_raw = val
    Y_val_scaled = target_scaler.transform(Y_val_raw.reshape(-1, 1)).reshape(Y_val_raw.shape).astype(np.float32)
    preds_val_scaled = historical_mean_predict(X_val_raw, target_scaler, horizon=6)
    val_metrics = compute_metrics(preds_val_scaled, Y_val_scaled, target_scaler)
    print_metrics(val_metrics, "Historical Mean Baseline (Validation)")

    return metrics


if __name__ == "__main__":
    main()
