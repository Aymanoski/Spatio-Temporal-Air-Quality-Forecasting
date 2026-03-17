"""
Comprehensive evaluation comparison for GCN-LSTM models.
Compares multiple models on the same test set with standardized metrics.
"""

import torch
import numpy as np
import os
import sys
import joblib
from pathlib import Path

# Add parent directory to path for model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GCNLSTMModel
from graph import build_wind_aware_adjacency_batch, WIND_CATEGORIES


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
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((pred[mask] - target[mask]) / target[mask])) * 100


def compute_r2(pred, target):
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return 1 - (ss_res / ss_tot)


def compute_extreme_value_metrics(pred, target, quantile=0.9):
    """
    Compute metrics specifically for extreme values (high pollution events).

    Args:
        pred: predictions
        target: ground truth
        quantile: threshold quantile for extreme values (default 0.9 = 90th percentile)

    Returns:
        dict with extreme value metrics
    """
    threshold = np.quantile(target, quantile)
    mask = target >= threshold

    if np.sum(mask) == 0:
        return {
            'threshold': threshold,
            'count': 0,
            'rmse': 0,
            'mae': 0,
            'mape': 0,
            'r2': 0
        }

    return {
        'threshold': threshold,
        'count': np.sum(mask),
        'rmse': compute_rmse(pred[mask], target[mask]),
        'mae': compute_mae(pred[mask], target[mask]),
        'mape': compute_mape(pred[mask], target[mask]),
        'r2': compute_r2(pred[mask], target[mask])
    }


# ============================================================================
# Model Loading & Evaluation
# ============================================================================

def load_model_from_checkpoint(checkpoint_path):
    """Load model from checkpoint."""
    print(f"\nLoading: {Path(checkpoint_path).name}")

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
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

    arch_info = checkpoint.get('architecture', {})
    print(f"  Architecture: {arch_info.get('name', 'unknown')}")
    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  Direct Decoding: {config.get('use_direct_decoding', False)}")
    print(f"  Wind Adjacency: {config.get('use_wind_adjacency', False)}")
    print(f"  Loss Type: {config.get('loss_type', 'standard')}")
    print(f"  Epoch: {checkpoint['epoch']}")

    return model, config, checkpoint


def extract_wind_features(X_batch, config):
    """Extract wind speed and direction from input batch."""
    wind_speed_idx = config.get('wind_speed_idx', 10)
    wind_dir_start = config.get('wind_dir_start_idx', 17)
    wind_dir_end = config.get('wind_dir_end_idx', 33)

    wind_speeds = X_batch[:, :, :, wind_speed_idx]
    wind_directions = X_batch[:, :, :, wind_dir_start:wind_dir_end]

    return wind_speeds, wind_directions


def build_dynamic_adjacency(X_batch, config, device):
    """Build dynamic wind-aware adjacency matrix for a batch."""
    wind_speeds, wind_directions = extract_wind_features(X_batch, config)

    adj_batch = build_wind_aware_adjacency_batch(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        wind_categories=WIND_CATEGORIES,
        alpha=config.get('wind_alpha', 0.6),
        distance_sigma=config.get('distance_sigma', 100)
    )

    adj_batch = torch.FloatTensor(adj_batch).to(device)
    return adj_batch


def evaluate_model(model, X_test, Y_test, adj, config, device='cpu'):
    """Run model predictions on test set."""
    model = model.to(device)
    adj_tensor = torch.FloatTensor(adj).to(device)

    use_wind_adj = config.get('use_wind_adjacency', False)
    batch_size = config.get('batch_size', 32)
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_x = torch.FloatTensor(X_test[i:i+batch_size]).to(device)

            if use_wind_adj:
                adj_batch = build_dynamic_adjacency(batch_x, config, device)
            else:
                adj_batch = adj_tensor

            pred = model.predict(batch_x, adj_batch, horizon=config['horizon'])
            all_preds.append(pred.cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)
    return predictions


def load_test_data(data_path, config):
    """Load test data."""
    X = np.load(os.path.join(data_path, "X.npy"))
    Y = np.load(os.path.join(data_path, "Y.npy"))
    adj = np.load(os.path.join(data_path, "adjacency.npy"))

    # Split to get test set
    n_samples = len(X)
    train_end = int(n_samples * config['train_ratio'])
    val_end = int(n_samples * (config['train_ratio'] + config['val_ratio']))

    X_test = X[val_end:]
    Y_test = Y[val_end:]

    return X_test, Y_test, adj


def inverse_transform(data, scaler_path):
    """Inverse transform to original scale."""
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        orig_shape = data.shape
        data = scaler.inverse_transform(data.reshape(-1, 1)).reshape(orig_shape)
        return data, True
    return data, False


# ============================================================================
# Comparison & Reporting
# ============================================================================

def compute_all_metrics(predictions, targets):
    """Compute all evaluation metrics."""
    metrics = {
        'overall': {
            'rmse': compute_rmse(predictions, targets),
            'mae': compute_mae(predictions, targets),
            'mape': compute_mape(predictions, targets),
            'r2': compute_r2(predictions, targets)
        },
        'extreme_90': compute_extreme_value_metrics(predictions, targets, 0.90),
        'extreme_95': compute_extreme_value_metrics(predictions, targets, 0.95),
        'per_horizon': []
    }

    # Per-horizon metrics
    horizon = predictions.shape[1]
    for h in range(horizon):
        metrics['per_horizon'].append({
            'hour': h + 1,
            'rmse': compute_rmse(predictions[:, h, :], targets[:, h, :]),
            'mae': compute_mae(predictions[:, h, :], targets[:, h, :]),
            'mape': compute_mape(predictions[:, h, :], targets[:, h, :])
        })

    return metrics


def print_comparison_table(headers, rows):
    """Print comparison table."""
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    header_str = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * len(header_str))

    for row in rows:
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))


def print_comparison_report(model_results):
    """Print comprehensive comparison report."""
    print("\n" + "=" * 80)
    print("MODEL EVALUATION COMPARISON - TEST SET RESULTS")
    print("=" * 80)

    model_names = list(model_results.keys())

    # 1. Overall Performance
    print("\n1. OVERALL PERFORMANCE (Original Scale - µg/m³)")
    print("-" * 80)

    headers = ["Metric"] + model_names + ["Best", "Improvement"]
    rows = []

    metrics_list = ['rmse', 'mae', 'mape', 'r2']
    metrics_labels = ['RMSE', 'MAE', 'MAPE (%)', 'R²']

    for metric, label in zip(metrics_list, metrics_labels):
        values = [model_results[name]['metrics']['overall'][metric] for name in model_names]

        if metric == 'r2':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)

        best_name = model_names[best_idx]

        # Calculate improvement
        if len(values) == 2:
            if metric == 'r2':
                improvement = ((values[1] - values[0]) / abs(values[0])) * 100
                improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
            else:
                improvement = ((values[0] - values[1]) / values[0]) * 100
                improvement_str = f"-{improvement:.2f}%" if improvement > 0 else f"+{abs(improvement):.2f}%"
        else:
            improvement_str = "N/A"

        formatted_values = [f"{v:.4f}" if metric != 'mape' else f"{v:.2f}" for v in values]
        rows.append([label] + formatted_values + [best_name, improvement_str])

    print_comparison_table(headers, rows)

    # 2. Extreme Value Performance
    print("\n2. EXTREME VALUE PERFORMANCE (90th Percentile)")
    print("-" * 80)

    rows = []
    for metric, label in zip(metrics_list, metrics_labels):
        values = [model_results[name]['metrics']['extreme_90'][metric] for name in model_names]

        if metric == 'r2':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)

        best_name = model_names[best_idx]

        if len(values) == 2:
            if metric == 'r2':
                improvement = ((values[1] - values[0]) / abs(values[0])) * 100
                improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
            else:
                improvement = ((values[0] - values[1]) / values[0]) * 100
                improvement_str = f"-{improvement:.2f}%" if improvement > 0 else f"+{abs(improvement):.2f}%"
        else:
            improvement_str = "N/A"

        formatted_values = [f"{v:.4f}" if metric != 'mape' else f"{v:.2f}" for v in values]
        rows.append([label] + formatted_values + [best_name, improvement_str])

    # Add threshold info
    threshold_values = [model_results[name]['metrics']['extreme_90']['threshold'] for name in model_names]
    count_values = [model_results[name]['metrics']['extreme_90']['count'] for name in model_names]
    rows.insert(0, ["Threshold"] + [f"{v:.2f}" for v in threshold_values] + ["-", "-"])
    rows.insert(1, ["Count"] + [str(int(v)) for v in count_values] + ["-", "-"])

    print_comparison_table(headers, rows)

    # 3. Per-Horizon Comparison
    print("\n3. PER-HORIZON RMSE COMPARISON")
    print("-" * 80)

    headers = ["Hour"] + model_names + ["Best"]
    rows = []

    horizon = len(model_results[model_names[0]]['metrics']['per_horizon'])
    for h in range(horizon):
        hour = h + 1
        values = [model_results[name]['metrics']['per_horizon'][h]['rmse'] for name in model_names]
        best_idx = np.argmin(values)
        best_name = model_names[best_idx]

        formatted_values = [f"{v:.4f}" for v in values]
        rows.append([f"+{hour}h"] + formatted_values + [best_name])

    print_comparison_table(headers, rows)

    # 4. Summary & Analysis
    print("\n4. SUMMARY & KEY INSIGHTS")
    print("-" * 80)

    if len(model_names) == 2:
        # Overall winner
        overall_rmse = [model_results[name]['metrics']['overall']['rmse'] for name in model_names]
        overall_winner = model_names[np.argmin(overall_rmse)]

        # Extreme value winner
        extreme_rmse = [model_results[name]['metrics']['extreme_90']['rmse'] for name in model_names]
        extreme_winner = model_names[np.argmin(extreme_rmse)]

        print(f"\nOverall Best Model: {overall_winner}")
        print(f"  RMSE: {model_results[overall_winner]['metrics']['overall']['rmse']:.4f} µg/m³")
        print(f"  MAE: {model_results[overall_winner]['metrics']['overall']['mae']:.4f} µg/m³")
        print(f"  R²: {model_results[overall_winner]['metrics']['overall']['r2']:.4f}")

        print(f"\nExtreme Event Best Model: {extreme_winner}")
        print(f"  RMSE: {model_results[extreme_winner]['metrics']['extreme_90']['rmse']:.4f} µg/m³")
        print(f"  MAE: {model_results[extreme_winner]['metrics']['extreme_90']['mae']:.4f} µg/m³")

        # Architecture comparison
        print("\nArchitecture Differences:")
        for name in model_names:
            config = model_results[name]['config']
            print(f"\n{name}:")
            print(f"  Direct Decoding: {config.get('use_direct_decoding', False)}")
            print(f"  Wind Adjacency: {config.get('use_wind_adjacency', False)}")
            print(f"  Loss Type: {config.get('loss_type', 'standard')}")
            print(f"  Parameters: {model_results[name]['num_params']:,}")

    print("\n" + "=" * 80)


# ============================================================================
# Main
# ============================================================================

def compare_models(*checkpoint_paths, data_path="data/processed/"):
    """
    Compare multiple models on the same test set.

    Args:
        *checkpoint_paths: Paths to model checkpoints
        data_path: Path to test data
    """
    if len(checkpoint_paths) < 2:
        raise ValueError("Need at least 2 checkpoints to compare")

    print("=" * 80)
    print("LOADING MODELS")
    print("=" * 80)

    models_info = {}

    # Load all models
    for i, ckpt_path in enumerate(checkpoint_paths):
        model, config, checkpoint = load_model_from_checkpoint(ckpt_path)
        arch_name = checkpoint.get('architecture', {}).get('name', f'model_{i+1}')

        models_info[arch_name] = {
            'model': model,
            'config': config,
            'checkpoint': checkpoint,
            'num_params': model.get_num_params()
        }

    # Load test data (use first model's config for data split)
    print("\n" + "=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)

    first_config = list(models_info.values())[0]['config']
    X_test, Y_test, adj = load_test_data(data_path, first_config)

    print(f"\nTest samples: {len(X_test)}")
    print(f"X shape: {X_test.shape}")
    print(f"Y shape: {Y_test.shape}")

    # Load scaler
    scaler_path = os.path.join(data_path, "target_scaler.save")

    # Evaluate all models
    print("\n" + "=" * 80)
    print("EVALUATING MODELS")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    model_results = {}

    for name, info in models_info.items():
        print(f"\nEvaluating {name}...")

        predictions = evaluate_model(
            info['model'],
            X_test,
            Y_test,
            adj,
            info['config'],
            device
        )

        # Inverse transform
        predictions_orig, _ = inverse_transform(predictions, scaler_path)
        targets_orig, is_scaled = inverse_transform(Y_test, scaler_path)

        # Compute metrics
        metrics = compute_all_metrics(predictions_orig, targets_orig)

        model_results[name] = {
            'predictions': predictions_orig,
            'metrics': metrics,
            'config': info['config'],
            'num_params': info['num_params']
        }

        print(f"  RMSE: {metrics['overall']['rmse']:.4f}")
        print(f"  MAE: {metrics['overall']['mae']:.4f}")

    # Print comparison report
    print_comparison_report(model_results)

    return model_results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate_comparison.py <checkpoint1> <checkpoint2> [checkpoint3...]")
        print("\nExample:")
        print("  python evaluate_comparison.py models/checkpoints/gcn_lstm_v1_best.pt models/checkpoints/gcn_lstm_v2_best.pt")
        sys.exit(1)

    checkpoint_paths = sys.argv[1:]
    compare_models(*checkpoint_paths)
