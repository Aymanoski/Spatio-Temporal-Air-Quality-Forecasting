"""
Shared data loading and evaluation for baseline models.

Shared data loading and evaluation for baseline models.

NOTE: This module uses MinMaxScaler (no log1p), which does NOT match the current
best model's preprocessing (StandardScaler + log1p on indices 0-5). Baselines
trained here are not directly comparable to the GraphTransformer results on
normalized metrics; compare on inverse-transformed µg/m³ (MAE/RMSE) only.

- Same chronological train/val/test split (70/15/15)
- MinMaxScaler fitted on training data only
- Same MAPE threshold (5 µg/m³)
"""

import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler


# ============================================================================
# Configuration (matches train.py exactly)
# ============================================================================

SHARED_CONFIG = {
    'data_path': 'data/processed/',
    'input_len': 24,
    'horizon': 6,
    'input_dim': 33,
    'num_nodes': 12,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'wind_dir_start_idx': 17,
    'seed': 42,
}


# ============================================================================
# Data Loading
# ============================================================================

def load_and_split(config=None):
    """
    Load data and split chronologically (identical to train.py).

    Returns:
        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), adj
        All in raw scale (not yet scaled).
    """
    if config is None:
        config = SHARED_CONFIG
    data_path = config['data_path']

    X = np.load(os.path.join(data_path, 'X.npy'))
    Y = np.load(os.path.join(data_path, 'Y.npy'))
    adj = np.load(os.path.join(data_path, 'adjacency.npy'))

    n = len(X)
    train_end = int(n * config['train_ratio'])
    val_end = int(n * (config['train_ratio'] + config['val_ratio']))

    train = (X[:train_end], Y[:train_end])
    val = (X[train_end:val_end], Y[train_end:val_end])
    test = (X[val_end:], Y[val_end:])

    print(f"Data loaded: X={X.shape}, Y={Y.shape}")
    print(f"  Train: {train[0].shape[0]}  Val: {val[0].shape[0]}  Test: {test[0].shape[0]}")

    return train, val, test, adj


def fit_and_scale(train, val, test, config=None):
    """
    Fit scalers on training data, transform all splits.
    Mirrors train.py fit_scalers_on_train + scale_data exactly.

    Returns:
        (X_train_s, Y_train_s), (X_val_s, Y_val_s), (X_test_s, Y_test_s),
        feature_scaler, target_scaler
    """
    if config is None:
        config = SHARED_CONFIG

    X_train, Y_train = train
    wind_start = config.get('wind_dir_start_idx', 17)
    n_features = X_train.shape[-1]

    # Feature scaler (non-wind features only)
    feature_scaler = MinMaxScaler()
    X_flat = X_train.reshape(-1, n_features)
    feature_scaler.fit(X_flat[:, :wind_start])

    # Target scaler (PM2.5)
    target_scaler = MinMaxScaler()
    target_scaler.fit(Y_train.reshape(-1, 1))

    def scale_split(X, Y):
        ns, sl, nn, nf = X.shape
        Xf = X.reshape(-1, nf)
        Xs = np.concatenate([
            feature_scaler.transform(Xf[:, :wind_start]),
            Xf[:, wind_start:]
        ], axis=1).reshape(ns, sl, nn, nf).astype(np.float32)

        Ys = target_scaler.transform(Y.reshape(-1, 1)).reshape(Y.shape).astype(np.float32)
        return Xs, Ys

    train_s = scale_split(X_train, Y_train)
    val_s = scale_split(*val)
    test_s = scale_split(*test)

    print(f"  Target scaler range: [{target_scaler.data_min_[0]:.1f}, {target_scaler.data_max_[0]:.1f}] ug/m3")

    return train_s, val_s, test_s, feature_scaler, target_scaler


# ============================================================================
# Metrics (matches train.py compute_metrics exactly)
# ============================================================================

def compute_metrics(preds, targets, target_scaler=None):
    """
    Compute RMSE, MAE, MAPE, R2 on original scale.

    Args:
        preds: (samples, horizon, nodes) in scaled space
        targets: (samples, horizon, nodes) in scaled space
        target_scaler: if provided, inverse-transforms both before computing metrics

    Returns:
        dict with RMSE, MAE, MAPE, R2 and per-horizon MAE
    """
    if target_scaler is not None:
        shape = preds.shape
        preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(shape)
        targets = target_scaler.inverse_transform(targets.reshape(-1, 1)).reshape(shape)

    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae = float(np.mean(np.abs(preds - targets)))

    # R2
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    # MAPE with threshold=5 (matches train.py)
    mask = targets > 5.0
    if mask.any():
        mape = float(np.mean(np.abs((preds[mask] - targets[mask]) / targets[mask])) * 100)
    else:
        mape = float('nan')

    # Per-horizon MAE
    horizon = preds.shape[1]
    horizon_mae = []
    for h in range(horizon):
        horizon_mae.append(float(np.mean(np.abs(preds[:, h, :] - targets[:, h, :]))))

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'horizon_mae': horizon_mae,
    }


def print_metrics(metrics, name):
    """Print metrics in a format consistent with the main training script."""
    print(f"\n{'=' * 60}")
    print(f"  {name} — Test Set Results (ug/m3)")
    print(f"{'=' * 60}")
    print(f"  RMSE:  {metrics['RMSE']:.2f}")
    print(f"  MAE:   {metrics['MAE']:.2f}")
    print(f"  MAPE:  {metrics['MAPE']:.1f}%")
    print(f"  R2:    {metrics['R2']:.3f}")
    print()
    print(f"  Per-horizon MAE:")
    for h, m in enumerate(metrics['horizon_mae']):
        print(f"    +{h+1}h: {m:.2f}")
    print(f"{'=' * 60}")
