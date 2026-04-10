"""
Baseline 1: Persistence — predict last observed PM2.5 for all horizons.

This is the simplest possible baseline. For each station, the prediction
for +1h through +6h is the PM2.5 value at t=0 (the last timestep in the
lookback window). No learned parameters.

This baseline answers: "Does the model beat simply assuming PM2.5 stays
the same as the most recent observation?"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from baselines.shared import load_and_split, fit_and_scale, compute_metrics, print_metrics


def persistence_predict(X_scaled, target_scaler, horizon=6):
    """
    Persistence baseline: repeat the last observed PM2.5 value.

    The model was trained on scaled targets, so we need to produce predictions
    in scaled space for fair metric computation.

    X contains raw features. PM2.5 is feature index 0.
    We extract the last timestep's PM2.5, then scale it through the target scaler
    to produce predictions in the same space as scaled Y.

    Args:
        X_scaled: (samples, input_len, nodes, features) — scaled features
        target_scaler: fitted MinMaxScaler for PM2.5 targets
        horizon: number of forecast steps

    Returns:
        preds: (samples, horizon, nodes) in scaled space
    """
    # Last timestep PM2.5 in scaled feature space (index 0)
    # But feature scaling and target scaling may differ —
    # we need to go through raw -> target_scaler to match Y.
    #
    # Actually, since both X features and Y targets come from the same raw PM2.5
    # values, and the feature_scaler was fit on all features while target_scaler
    # was fit on Y only, we need to:
    # 1. Get the last PM2.5 from X in raw scale
    # 2. Scale it through target_scaler
    #
    # But X_scaled[:, -1, :, 0] is already in feature-scaler space, not target-scaler space.
    # The safest approach: inverse the feature scaling on PM2.5, then apply target scaling.
    # However, since we don't easily have access to the feature scaler's PM2.5 column here,
    # we take a simpler correct approach: work from raw X.

    # We'll receive raw X separately. See main() below.
    raise NotImplementedError("Use persistence_predict_raw instead")


def persistence_predict_raw(X_raw, target_scaler, horizon=6):
    """
    Persistence from raw (unscaled) X.

    Args:
        X_raw: (samples, input_len, nodes, features) in original ug/m3 scale
        target_scaler: fitted MinMaxScaler for targets
        horizon: forecast steps

    Returns:
        preds: (samples, horizon, nodes) in scaled [0,1] space
    """
    # PM2.5 is feature index 0, last timestep
    last_pm25 = X_raw[:, -1, :, 0]  # (samples, nodes)

    # Repeat for all horizons
    preds_raw = np.repeat(last_pm25[:, np.newaxis, :], horizon, axis=1)  # (samples, horizon, nodes)

    # Scale into target space
    shape = preds_raw.shape
    preds_scaled = target_scaler.transform(preds_raw.reshape(-1, 1)).reshape(shape)

    return preds_scaled.astype(np.float32)


def main():
    print("=" * 60)
    print("Baseline 1: Persistence")
    print("  Predicts last observed PM2.5 for all future horizons")
    print("=" * 60)

    # Load and split (raw scale)
    train, val, test, adj = load_and_split()

    # Fit scalers (we need target_scaler for metric computation)
    _, _, _, _, target_scaler = fit_and_scale(train, val, test)

    X_test_raw, Y_test_raw = test

    # Generate predictions in scaled space
    # We need Y_test in scaled space too for consistent metric computation
    Y_test_scaled = target_scaler.transform(Y_test_raw.reshape(-1, 1)).reshape(Y_test_raw.shape).astype(np.float32)
    preds_scaled = persistence_predict_raw(X_test_raw, target_scaler, horizon=6)

    # Compute metrics (inverse-transforms internally)
    metrics = compute_metrics(preds_scaled, Y_test_scaled, target_scaler)
    print_metrics(metrics, "Persistence Baseline")

    # Also evaluate on val set for completeness
    X_val_raw, Y_val_raw = val
    Y_val_scaled = target_scaler.transform(Y_val_raw.reshape(-1, 1)).reshape(Y_val_raw.shape).astype(np.float32)
    preds_val_scaled = persistence_predict_raw(X_val_raw, target_scaler, horizon=6)
    val_metrics = compute_metrics(preds_val_scaled, Y_val_scaled, target_scaler)
    print_metrics(val_metrics, "Persistence Baseline (Validation)")

    return metrics


if __name__ == "__main__":
    main()
