"""
Evaluation script for trained GCN-LSTM model.
Loads best checkpoint and computes metrics on test set.
"""

import torch
import numpy as np
import os
import sys
import joblib

# Add parent directory to path for model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GCNLSTMModel
from graph import build_wind_aware_adjacency_batch, WIND_CATEGORIES


# ============================================================================
# Configuration
# ============================================================================

DATA_PATH = "../data/processed/"
MODEL_PATH = "../models/checkpoints/best_model.pt"


# ============================================================================
# Metrics
# ============================================================================

def compute_rmse(pred, target):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((pred - target) ** 2))


def compute_mae(pred, target):
    """Mean Absolute Error."""
    return np.mean(np.abs(pred - target))


def compute_mape(pred, target, epsilon=1e-8):
    """Mean Absolute Percentage Error."""
    mask = np.abs(target) > epsilon
    return np.mean(np.abs((pred[mask] - target[mask]) / target[mask])) * 100


def compute_r2(pred, target):
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return 1 - (ss_res / ss_tot)


# ============================================================================
# Evaluation Functions
# ============================================================================

def load_model(model_path):
    """Load trained model from checkpoint."""
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    config = checkpoint['config']
    
    model = GCNLSTMModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        num_nodes=config['num_nodes'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        horizon=config.get('horizon', 6),
        use_direct_decoding=config.get('use_direct_decoding', False)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Loaded from epoch {checkpoint['epoch'] + 1}")
    print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
    print(f"  Parameters: {model.get_num_params():,}")
    
    return model, config


def load_data(data_path, config):
    """Load test data."""
    print(f"\nLoading data from {data_path}...")
    
    X = np.load(os.path.join(data_path, "X.npy"))
    Y = np.load(os.path.join(data_path, "Y.npy"))
    adj = np.load(os.path.join(data_path, "adjacency.npy"))
    
    # Split to get test set (same split as training)
    n_samples = len(X)
    train_end = int(n_samples * config['train_ratio'])
    val_end = int(n_samples * (config['train_ratio'] + config['val_ratio']))
    
    X_test = X[val_end:]
    Y_test = Y[val_end:]
    
    print(f"  Test samples: {len(X_test)}")
    print(f"  X shape: {X_test.shape}")
    print(f"  Y shape: {Y_test.shape}")
    
    return X_test, Y_test, adj


def extract_wind_features(X_batch, config):
    """
    Extract wind speed and direction from input batch.

    Args:
        X_batch: (batch, timesteps, num_nodes, features) tensor
        config: configuration dict

    Returns:
        wind_speeds: (batch, timesteps, num_nodes) tensor
        wind_directions: (batch, timesteps, num_nodes, num_wind_categories) tensor
    """
    wind_speed_idx = config.get('wind_speed_idx', 10)
    wind_dir_start = config.get('wind_dir_start_idx', 17)
    wind_dir_end = config.get('wind_dir_end_idx', 33)

    # Extract wind speed (single feature)
    wind_speeds = X_batch[:, :, :, wind_speed_idx]  # (batch, timesteps, nodes)

    # Extract wind direction one-hot (multiple features)
    wind_directions = X_batch[:, :, :, wind_dir_start:wind_dir_end]  # (batch, timesteps, nodes, categories)

    return wind_speeds, wind_directions


def build_dynamic_adjacency(X_batch, config, device):
    """
    Build dynamic wind-aware adjacency matrix for a batch.

    Args:
        X_batch: (batch, timesteps, num_nodes, features) tensor
        config: configuration dict
        device: torch device

    Returns:
        adj_batch: (batch, num_nodes, num_nodes) tensor
    """
    wind_speeds, wind_directions = extract_wind_features(X_batch, config)

    # Build wind-aware adjacency
    adj_batch = build_wind_aware_adjacency_batch(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        wind_categories=WIND_CATEGORIES,
        alpha=config.get('wind_alpha', 0.6),
        distance_sigma=config.get('distance_sigma', 100),
        aggregation_mode=config.get('wind_aggregation_mode', 'recent_weighted'),
        recency_beta=config.get('wind_recency_beta', 3.0),
        direction_method=config.get('wind_direction_method', 'circular'),
        normalization=config.get('wind_normalization', 'row'),
        calm_speed_threshold=config.get('wind_calm_speed_threshold', 0.1)
    )

    # Convert to tensor
    adj_batch = torch.FloatTensor(adj_batch).to(device)

    return adj_batch


def evaluate(model, X_test, Y_test, adj, config):
    """Run evaluation on test set."""
    print("\nRunning evaluation...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    adj_tensor = torch.FloatTensor(adj).to(device)

    use_wind_adj = config.get('use_wind_adjacency', False)

    # Batch prediction
    batch_size = config.get('batch_size', 32)
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_x = torch.FloatTensor(X_test[i:i+batch_size]).to(device)

            # Build dynamic adjacency if enabled
            if use_wind_adj:
                adj_batch = build_dynamic_adjacency(batch_x, config, device)
            else:
                adj_batch = adj_tensor

            pred = model.predict(batch_x, adj_batch, horizon=config['horizon'])
            all_preds.append(pred.cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)

    return predictions


def inverse_transform(predictions, targets, data_path):
    """Inverse transform predictions to original scale."""
    scaler_path = os.path.join(data_path, "target_scaler.save")
    
    if os.path.exists(scaler_path):
        print("\nInverse transforming to original scale...")
        scaler = joblib.load(scaler_path)
        
        orig_shape = predictions.shape
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(orig_shape)
        targets = scaler.inverse_transform(targets.reshape(-1, 1)).reshape(orig_shape)
        
        return predictions, targets, True
    else:
        print("\nNo scaler found, using normalized values...")
        return predictions, targets, False


def print_metrics(predictions, targets, is_original_scale):
    """Compute and print all metrics."""
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    
    scale_label = "(Original Scale - µg/m³)" if is_original_scale else "(Normalized)"
    print(f"\nOverall Metrics {scale_label}:")
    print("-" * 40)
    
    rmse = compute_rmse(predictions, targets)
    mae = compute_mae(predictions, targets)
    mape = compute_mape(predictions, targets)
    r2 = compute_r2(predictions, targets)
    
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  MAE:   {mae:.4f}")
    print(f"  MAPE:  {mape:.2f}%")
    print(f"  R²:    {r2:.4f}")
    
    # Per-horizon metrics
    print(f"\nPer-Horizon Metrics {scale_label}:")
    print("-" * 40)
    print(f"{'Hour':<6} {'RMSE':<10} {'MAE':<10} {'MAPE(%)':<10}")
    print("-" * 40)
    
    horizon = predictions.shape[1]
    for h in range(horizon):
        h_rmse = compute_rmse(predictions[:, h, :], targets[:, h, :])
        h_mae = compute_mae(predictions[:, h, :], targets[:, h, :])
        h_mape = compute_mape(predictions[:, h, :], targets[:, h, :])
        print(f"  +{h+1:<4} {h_rmse:<10.4f} {h_mae:<10.4f} {h_mape:<10.2f}")
    
    # Per-station metrics
    print(f"\nPer-Station Metrics {scale_label}:")
    print("-" * 40)
    
    station_names = [
        "Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan",
        "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan",
        "Wanliu", "Wanshouxigong"
    ]
    
    print(f"{'Station':<15} {'RMSE':<10} {'MAE':<10} {'MAPE(%)':<10}")
    print("-" * 40)
    
    for s, name in enumerate(station_names):
        s_rmse = compute_rmse(predictions[:, :, s], targets[:, :, s])
        s_mae = compute_mae(predictions[:, :, s], targets[:, :, s])
        s_mape = compute_mape(predictions[:, :, s], targets[:, :, s])
        print(f"  {name:<13} {s_rmse:<10.4f} {s_mae:<10.4f} {s_mape:<10.2f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }


def save_predictions(predictions, targets, save_path="../data/processed/"):
    """Save predictions for further analysis."""
    np.save(os.path.join(save_path, "test_predictions.npy"), predictions)
    np.save(os.path.join(save_path, "test_targets.npy"), targets)
    print(f"\nPredictions saved to {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("GCN-LSTM Model Evaluation")
    print("=" * 60)
    
    # Load model
    model, config = load_model(MODEL_PATH)
    
    # Load data
    X_test, Y_test, adj = load_data(DATA_PATH, config)
    
    # Run predictions
    predictions = evaluate(model, X_test, Y_test, adj, config)
    
    print(f"\nPrediction shape: {predictions.shape}")
    print(f"Target shape: {Y_test.shape}")
    
    # Inverse transform
    predictions, Y_test, is_original = inverse_transform(predictions, Y_test, DATA_PATH)
    
    # Compute metrics
    metrics = print_metrics(predictions, Y_test, is_original)
    
    # Save predictions
    save_predictions(predictions, Y_test)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    main()
