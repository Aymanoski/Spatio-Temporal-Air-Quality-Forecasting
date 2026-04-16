"""
Training script for GCN-LSTM Encoder-Decoder model.
Air quality (PM2.5) forecasting using spatio-temporal graph neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
import random
import joblib
from sklearn.preprocessing import MinMaxScaler
from models import GCNLSTMModel, GraphTransformerModel
from utils.graph import (
    build_wind_aware_adjacency_batch,
    build_dynamic_adjacency_gpu,
    WIND_CATEGORIES
)

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Data
    'data_path': 'data/processed/',
    'input_len': 24,        # Lookback window (hours)
    'horizon': 6,           # Prediction horizon (hours)
    
    # Model architecture
    'input_dim': 33,        # Number of input features per node
    'hidden_dim': 64,       # Hidden dimension
    'output_dim': 1,        # Output dimension (PM2.5 only)
    'num_nodes': 12,        # Number of monitoring stations
    'num_layers': 2,        # Number of Graph LSTM layers
    'num_heads': 4,         # Attention heads
    'dropout': 0.1,
    'use_direct_decoding': True,  # Direct multi-horizon decoding (no autoregression)
    'use_attention': False,        # MHA tested and removed — zero measurable effect (2026-04-14)

    # Model type switch
    # 'gcn_lstm'         — original GraphLSTM encoder-decoder (recurrent baseline)
    # 'graph_transformer'— GCN per-timestep + Transformer encoder + direct head (new)
    'model_type': 'graph_transformer',
    'num_tf_layers': 2,  # Transformer encoder layers (graph_transformer only)

    # Graph convolution type ('gcn' | 'gat')
    # 'gcn' — original GraphConvolution with normalized adjacency (default, backward-compatible)
    # 'gat' — GraphAttentionLayer: learned attention + wind-aware adjacency as additive bias
    'graph_conv': 'gat',
    'num_gat_layers': 1,   # Number of stacked GAT layers (1=1-hop, 2=2-hop neighbourhood)
    'gat_version': 'v1',  # 'v1' = standard GAT, 'v2' = GATv2 (dynamic attention)
    'use_post_temporal_gat': False,  # Post-temporal GAT: FAILED (Test MAE 21.022, destabilized training)

    # Horizon-conditioned temporal attention head.
    # Replaces last-token pooling with per-horizon soft attention over all T timesteps.
    # Hypothesis: H4-H6 degradation is caused by collapsing T=24 → 1 vector.
    # Each horizon h learns which timesteps to attend to (trend, slope, periodicity).
    # Compare against DirectHorizonHead baseline: Val 18.834 / Test 20.624 / RMSE 37.729.
    'use_temporal_attention_head': True,

    # Residual prediction: model outputs delta from last-observed PM2.5 (persistence baseline).
    # final_prediction = model_output + last_observed_PM2.5
    # Forces model to learn corrections to a sensible prior instead of absolute values.
    # Particularly helps longer horizons where absolute prediction is harder.
    # PM2.5 feature index in X must match config['target_feature_idx'].
    'use_persistence_residual': True,
    'target_feature_idx': 0,  # Index of PM2.5 in the feature dimension of X

    # Training
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'epochs': 100,
    'patience': 15,          # Early stopping patience (back to original)
    'teacher_forcing_start': 1.0,  # Initial teacher forcing ratio
    'teacher_forcing_end': 0.0,    # Final teacher forcing ratio

    # Loss
    'loss_type': 'evt_hybrid',     # Options: 'mse' or 'evt_hybrid'
    'evt_lambda': 0.05,            # Base weight of EVT tail component (used if no schedule)
    'evt_tail_quantile': 0.90,     # Threshold quantile for extremes
    'evt_xi': 0.10,                # GPD shape parameter
    'evt_threshold': None,   
    'evt_threshold_mode': 'global',      # Populated from training targets

    # EVT Improvements: DISABLED (fixed λ=0.05 validation)
    'evt_asymmetric_penalty': False,        # Disabled for baseline comparison
    'evt_under_penalty_multiplier': 2.0,    # Not used when asymmetric_penalty=False
    'evt_use_lambda_schedule': False,       # Disabled - using fixed λ=0.05
    'evt_lambda_schedule': {
        'initial': 0.05,     # Epochs 1-25: Learn general patterns (extended warmup)
        'mid': 0.12,         # Epochs 26-50: Gentle increase in extreme focus
        'final': 0.25,       # Epochs 51+: Moderate extreme emphasis (less aggressive)
        'warmup_epochs': 25, # Extended warmup period
        'mid_epochs': 50,    # Slower transition to final
        'transition': 'smooth'  # 'smooth' for gradual, 'step' for abrupt changes
    },

    # Wind-aware adjacency
    'use_wind_adjacency': True,    # Use dynamic wind-aware adjacency
    'wind_alpha': 0.6,             # Wind influence weight (0=distance-only, 1=wind-only)
    'use_learnable_alpha_gate': True,  # Learn alpha instead of keeping wind_alpha fixed
    'use_node_embeddings': True,        # Learnable per-station identity embeddings (post-LN injection)
    'distance_sigma': 1800,        # Distance decay parameter (calibrated for Beijing ~35km mean)
    'wind_aggregation_mode': 'recent_weighted',  # 'recent_weighted' | 'last' | 'mean'
    'wind_recency_beta': 3.0,      # Recency emphasis for recent_weighted aggregation
    'wind_direction_method': 'circular',  # 'circular' | 'argmax_mean'
    'wind_normalization': 'row',   # 'row' (directed) | 'symmetric'
    'wind_calm_speed_threshold': 0.1,
    'wind_speed_idx': 10,          # Index of wind speed feature (wspm)
    'wind_dir_start_idx': 17,      # Start index of wind direction one-hot
    'wind_dir_end_idx': 33,        # End index of wind direction one-hot

    # Data split
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Paths
    'model_save_path': 'models/checkpoints/',
    'best_model_name': 'best_model.pt',

    # Checkpoint naming (for comparing different runs)
    'architecture_name': 'graph_transformer_gat_v1_residual_temporalpool',  # GraphTransformer + GATv1 + persistence residual + horizon-conditioned temporal attention head
    'hardware_tag': 'T4',       # Options: 'integrated_gpu', 'T4', 'rtx3090', etc.
    'use_versioned_checkpoint': True,       # If True, saves as <arch>_<hardware>_best.pt

    # Resume training
    'resume': False,        # Set to True to resume from checkpoint

    # Reproducibility
    'seed': 42,
    'deterministic': False,

    # Runtime controls (useful for hyperparameter tuning)
    'save_checkpoints': True,
    'evaluate_test': True,
    'save_history': True
}


class EVTHybridLoss(nn.Module):
    """
    Hybrid loss: standard MSE + EVT tail-aware penalty.

    Penalizes errors more heavily for extreme values above the threshold.
    Uses a power-law penalty inspired by GPD to emphasize extreme value errors.

    Improvements:
    - Asymmetric penalty: under-predictions penalized more than over-predictions
    - Adaptive lambda: can be adjusted during training via set_lambda()
    """

    def __init__(self, threshold, lambda_tail=0.05, xi=0.10, eps=1e-6,
                 asymmetric_penalty=False, under_penalty_multiplier=2.0):
        super().__init__()
        # threshold can be a float (global) or a 1D array-like (per node).
        thr = torch.as_tensor(threshold, dtype=torch.float32)
        self.register_buffer("threshold", thr)
        self.lambda_tail = float(lambda_tail)
        self.xi = float(xi)
        self.eps = float(eps)
        self.asymmetric_penalty = asymmetric_penalty
        self.under_penalty_multiplier = float(under_penalty_multiplier)

    def set_lambda(self, new_lambda):
        """Update lambda weight (for adaptive scheduling)."""
        self.lambda_tail = float(new_lambda)

    def forward(self, predictions, targets):
        # Base MSE loss for all predictions
        mse = F.mse_loss(predictions, targets)

        # Identify extreme values (above threshold)
        thr = self.threshold.to(device=targets.device, dtype=targets.dtype)
        if thr.ndim == 0:
            thr_view = thr
        elif thr.ndim == 1:
            # Per-node thresholds; predictions/targets are (batch, horizon, nodes)
            thr_view = thr.view(1, 1, -1)
        else:
            raise ValueError(f"Unsupported threshold shape: {tuple(thr.shape)}")

        mask = targets > thr_view

        if mask.any():
            # Get predictions and targets for extreme values
            extreme_preds = predictions[mask]
            extreme_targets = targets[mask]

            # Select matching thresholds for masked elements (handles global or per-node thresholds)
            thr_selected = thr_view.expand_as(targets)[mask]
            target_excess = extreme_targets - thr_selected

            # Weighted squared error directly on predictions (keeps gradient for under-prediction)
            err = extreme_preds - extreme_targets

            # Weight increases with magnitude of excess (stabilize denominator)
            mean_excess = target_excess.mean().detach()
            excess_weight = 1.0 + self.xi * target_excess / (mean_excess + self.eps)
            excess_weight = torch.clamp(excess_weight, min=1.0)

            # Asymmetric penalty: penalize under-prediction more
            if self.asymmetric_penalty:
                under_mask = err < 0
                excess_weight = excess_weight * torch.where(
                    under_mask,
                    err.new_tensor(self.under_penalty_multiplier),
                    err.new_tensor(1.0)
                )

            tail_loss = (excess_weight * (err ** 2)).mean()
        else:
            tail_loss = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)

        # Combine MSE + weighted tail penalty
        return mse + self.lambda_tail * tail_loss


def get_evt_lambda_for_epoch(epoch, schedule_config, total_epochs):
    """
    Calculate EVT lambda for current epoch based on schedule.

    Args:
        epoch: Current epoch (0-indexed)
        schedule_config: Dictionary with schedule parameters
        total_epochs: Total number of training epochs

    Returns:
        lambda value for current epoch
    """
    initial = schedule_config['initial']
    mid = schedule_config['mid']
    final = schedule_config['final']
    warmup_epochs = schedule_config.get('warmup_epochs', 20)
    mid_epochs = schedule_config.get('mid_epochs', 40)
    transition = schedule_config.get('transition', 'smooth')

    if epoch < warmup_epochs:
        # Warmup phase: stay at initial lambda
        return initial
    elif epoch < mid_epochs:
        # Transition from initial to mid
        if transition == 'smooth':
            # Linear interpolation
            progress = (epoch - warmup_epochs) / (mid_epochs - warmup_epochs)
            return initial + (mid - initial) * progress
        else:  # step
            return mid
    else:
        # Transition from mid to final
        if transition == 'smooth':
            # Linear interpolation
            progress = (epoch - mid_epochs) / max(total_epochs - mid_epochs, 1)
            progress = min(progress, 1.0)  # Cap at 1.0
            return mid + (final - mid) * progress
        else:  # step
            return final


def set_global_seed(seed, deterministic=False):
    """Set random seeds for reproducible training runs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Optional deterministic mode (slower, but more reproducible)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Data Loading
# ============================================================================

def load_data(config):
    """Load preprocessed data tensors.

    Looks for X_{input_len}.npy first (named files generated by utils/window.py
    --input_len flag). Falls back to X.npy for backward compatibility.
    """
    data_path = config['data_path']
    input_len = config['input_len']

    # Prefer named files (e.g. X_48.npy) so multiple window sizes can coexist.
    # Fall back to X.npy if the named file doesn't exist (original behaviour).
    x_named = os.path.join(data_path, f'X_{input_len}.npy')
    y_named = os.path.join(data_path, f'Y_{input_len}.npy')
    x_default = os.path.join(data_path, 'X.npy')
    y_default = os.path.join(data_path, 'Y.npy')

    if os.path.exists(x_named):
        X = np.load(x_named)
        Y = np.load(y_named)
        print(f"Loaded windowed data (input_len={input_len}): {x_named}")
    else:
        print(f"[Warning] {x_named} not found — falling back to X.npy. "
              f"Run: python utils/window.py --input_len {input_len}")
        X = np.load(x_default)
        Y = np.load(y_default)

    adj = np.load(os.path.join(data_path, 'adjacency.npy'))

    print(f"Loaded X: {X.shape}")  # (samples, input_len, num_nodes, features)
    print(f"Loaded Y: {Y.shape}")  # (samples, horizon, num_nodes)
    print(f"Loaded adjacency: {adj.shape}")  # (num_nodes, num_nodes)

    return X, Y, adj


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
    wind_speed_idx = config['wind_speed_idx']
    wind_dir_start = config['wind_dir_start_idx']
    wind_dir_end = config['wind_dir_end_idx']

    # Extract wind speed (single feature)
    wind_speeds = X_batch[:, :, :, wind_speed_idx]  # (batch, timesteps, nodes)

    # Extract wind direction one-hot (multiple features)
    wind_directions = X_batch[:, :, :, wind_dir_start:wind_dir_end]  # (batch, timesteps, nodes, categories)

    return wind_speeds, wind_directions


def build_dynamic_adjacency(X_batch, config, device, alpha_override=None, sigma_override=None):
    """
    Build dynamic wind-aware adjacency matrix for a batch.
    Uses GPU-optimized computation when possible.

    Args:
        X_batch: (batch, timesteps, num_nodes, features) tensor
        config: configuration dict
        device: torch device
        alpha_override: learnable alpha tensor from model.get_wind_alpha() — keeps grad flow
        sigma_override: learnable sigma tensor from model.get_distance_sigma() — keeps grad flow

    Returns:
        adj_batch: (batch, num_nodes, num_nodes) tensor
    """
    # Use the torch implementation whenever either parameter is learnable so
    # gradients can flow from the loss back through the adjacency construction.
    if X_batch.is_cuda or alpha_override is not None or sigma_override is not None:
        return build_dynamic_adjacency_gpu(
            X_batch, config,
            alpha_override=alpha_override,
            sigma_override=sigma_override
        )

    # Fallback to CPU version for non-GPU tensors
    wind_speeds, wind_directions = extract_wind_features(X_batch, config)

    # Build wind-aware adjacency
    adj_batch = build_wind_aware_adjacency_batch(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        wind_categories=WIND_CATEGORIES,
        alpha=config['wind_alpha'] if alpha_override is None else float(alpha_override),
        distance_sigma=config['distance_sigma'],
        aggregation_mode=config.get('wind_aggregation_mode', 'recent_weighted'),
        recency_beta=config.get('wind_recency_beta', 3.0),
        direction_method=config.get('wind_direction_method', 'circular'),
        normalization=config.get('wind_normalization', 'row'),
        calm_speed_threshold=config.get('wind_calm_speed_threshold', 0.1)
    )

    # Convert to tensor
    adj_batch = torch.FloatTensor(adj_batch).to(device)

    return adj_batch


def split_data(X, Y, config):
    """Split data into train/val/test sets (chronological split)."""
    n_samples = len(X)

    train_end = int(n_samples * config['train_ratio'])
    val_end = int(n_samples * (config['train_ratio'] + config['val_ratio']))

    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def fit_scalers_on_train(X_train, Y_train, config):
    """
    Fit scalers on training data only to prevent data leakage.

    Args:
        X_train: Training input data (samples, seq_len, nodes, features)
        Y_train: Training target data (samples, horizon, nodes)
        config: Configuration dict

    Returns:
        feature_scaler: Fitted scaler for input features (excluding wind one-hot)
        target_scaler: Fitted scaler for PM2.5 targets
        already_scaled: Boolean indicating if data appears pre-scaled
    """
    # Get dimensions
    n_samples, seq_len, n_nodes, n_features = X_train.shape

    # Find wind direction feature indices (these are one-hot, don't scale)
    wind_start_idx = config.get('wind_dir_start_idx', 17)

    # Check if data appears already scaled (values mostly in 0-1 range)
    # PM2.5 (index 0) in original scale is typically 0-500+ µg/m³
    pm25_max = X_train[:, :, :, 0].max()
    pm25_min = X_train[:, :, :, 0].min()
    y_max = Y_train.max()
    y_min = Y_train.min()

    # If PM2.5 values are in 0-1 range, data is likely pre-scaled
    already_scaled = pm25_max <= 1.5 and pm25_min >= -0.5 and y_max <= 1.5 and y_min >= -0.5

    if already_scaled:
        print("  [WARNING] Data appears to be pre-scaled (values in 0-1 range).")
        print("  Skipping additional scaling to avoid double-scaling.")
        print("  For proper data leakage prevention, regenerate data:")
        print("    1. python preproccess.py")
        print("    2. cd utils && python window.py")
        return None, None, True

    # Flatten X for fitting: (samples * seq_len * nodes, features)
    X_flat = X_train.reshape(-1, n_features)

    # Fit feature scaler on non-wind features (indices 0 to wind_start_idx)
    # This includes PM2.5 (idx 0), other pollutants, meteo, and temporal features
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(X_flat[:, :wind_start_idx])

    # Fit target scaler on Y values (PM2.5 only)
    Y_flat = Y_train.reshape(-1, 1)
    target_scaler = MinMaxScaler()
    target_scaler.fit(Y_flat)

    return feature_scaler, target_scaler, False


def scale_data(X, Y, feature_scaler, target_scaler, config):
    """
    Scale input features and targets using pre-fitted scalers.

    Args:
        X: Input data (samples, seq_len, nodes, features)
        Y: Target data (samples, horizon, nodes)
        feature_scaler: Pre-fitted scaler for features
        target_scaler: Pre-fitted scaler for targets
        config: Configuration dict

    Returns:
        X_scaled, Y_scaled: Scaled arrays
    """
    n_samples, seq_len, n_nodes, n_features = X.shape
    wind_start_idx = config.get('wind_dir_start_idx', 17)

    # Scale features
    X_flat = X.reshape(-1, n_features)
    X_scaled_features = feature_scaler.transform(X_flat[:, :wind_start_idx])

    # Reconstruct X with scaled features + unscaled wind one-hot
    X_scaled_flat = np.concatenate([
        X_scaled_features,
        X_flat[:, wind_start_idx:]  # Wind one-hot stays unchanged
    ], axis=1)
    X_scaled = X_scaled_flat.reshape(n_samples, seq_len, n_nodes, n_features)

    # Scale targets
    Y_flat = Y.reshape(-1, 1)
    Y_scaled_flat = target_scaler.transform(Y_flat)
    Y_scaled = Y_scaled_flat.reshape(Y.shape)

    return X_scaled.astype(np.float32), Y_scaled.astype(np.float32)


def create_dataloaders(train_data, val_data, test_data, config):
    """Create PyTorch DataLoaders."""
    X_train, Y_train = train_data
    X_val, Y_val = val_data
    X_test, Y_test = test_data
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(Y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(Y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(Y_test)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion, adj, config, teacher_forcing_ratio):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    device = config['device']
    use_wind_adj = config.get('use_wind_adjacency', False)

    for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()

        # Build dynamic adjacency if enabled
        if use_wind_adj:
            alpha_override = model.get_wind_alpha()
            adj_batch = build_dynamic_adjacency(X_batch, config, device, alpha_override=alpha_override)
        else:
            adj_batch = adj

        # Forward pass
        predictions, _ = model(
            x=X_batch,
            adj=adj_batch,
            target=Y_batch,
            horizon=config['horizon'],
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        # predictions: (batch, horizon, num_nodes, 1)
        # Y_batch: (batch, horizon, num_nodes)
        predictions = predictions.squeeze(-1)

        # Persistence residual: model outputs delta from last observed PM2.5.
        # Add the baseline so loss is computed on final predictions vs actual targets.
        if config.get('use_persistence_residual', False):
            feat_idx = config.get('target_feature_idx', 0)
            # last observed PM2.5 (scaled): (B, N)
            y_last = X_batch[:, -1, :, feat_idx]
            # expand to (B, horizon, N) and add to model delta output
            predictions = predictions + y_last.unsqueeze(1).expand_as(predictions)

        # Compute loss
        loss = criterion(predictions, Y_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, adj, config, target_scaler=None):
    """
    Validate model.

    Returns:
        val_loss: Mean EVT/MSE loss on normalized predictions (used for LR scheduler).
        val_mae:  Mean absolute error on original scale if target_scaler is provided,
                  otherwise on normalized scale. Used for early stopping and checkpoint
                  selection — decoupled from the loss function so EVT lambda changes
                  do not artificially trigger early stopping.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    device = config['device']
    use_wind_adj = config.get('use_wind_adjacency', False)

    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Build dynamic adjacency if enabled
            if use_wind_adj:
                alpha_override = model.get_wind_alpha()
                adj_batch = build_dynamic_adjacency(X_batch, config, device, alpha_override=alpha_override)
            else:
                adj_batch = adj

            # Forward pass (no teacher forcing)
            predictions, _ = model(
                x=X_batch,
                adj=adj_batch,
                target=None,
                horizon=config['horizon'],
                teacher_forcing_ratio=0.0
            )

            predictions = predictions.squeeze(-1)

            # Persistence residual: add last observed PM2.5 to model delta output
            if config.get('use_persistence_residual', False):
                feat_idx = config.get('target_feature_idx', 0)
                y_last = X_batch[:, -1, :, feat_idx]
                predictions = predictions + y_last.unsqueeze(1).expand_as(predictions)

            loss = criterion(predictions, Y_batch)
            total_loss += loss.item()

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(Y_batch.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute MAE in original scale for early stopping (loss-function-independent)
    if target_scaler is not None:
        orig_shape = preds.shape
        preds_inv = target_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(orig_shape)
        targets_inv = target_scaler.inverse_transform(targets.reshape(-1, 1)).reshape(orig_shape)
        val_mae = float(np.mean(np.abs(preds_inv - targets_inv)))
    else:
        val_mae = float(np.mean(np.abs(preds - targets)))

    return total_loss / len(val_loader), val_mae


def compute_metrics(model, test_loader, adj, config, target_scaler=None):
    """Compute evaluation metrics on test set."""
    model.eval()
    device = config['device']
    use_wind_adj = config.get('use_wind_adjacency', False)

    all_preds = []
    all_targets = []

    use_residual = config.get('use_persistence_residual', False)
    feat_idx = config.get('target_feature_idx', 0)

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)

            # Build dynamic adjacency if enabled
            if use_wind_adj:
                alpha_override = model.get_wind_alpha()
                adj_batch = build_dynamic_adjacency(X_batch, config, device, alpha_override=alpha_override)
            else:
                adj_batch = adj

            predictions = model.predict(X_batch, adj_batch, horizon=config['horizon'])

            # Persistence residual: add last observed PM2.5 to model delta output
            if use_residual:
                y_last = X_batch[:, -1, :, feat_idx]                   # (B, N)
                predictions = predictions + y_last.unsqueeze(1).expand_as(predictions)

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(Y_batch.numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Inverse transform if scaler provided
    if target_scaler is not None:
        # Reshape for inverse transform
        orig_shape = preds.shape
        preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(orig_shape)
        targets = target_scaler.inverse_transform(targets.reshape(-1, 1)).reshape(orig_shape)
    
    # Compute metrics
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))
    
    # MAPE (use threshold to avoid division by small values)
    # PM2.5 values below 5 µg/m³ are very low and can cause unstable MAPE
    mape_threshold = 5.0
    mask = targets > mape_threshold
    if mask.any():
        mape = np.mean(np.abs((preds[mask] - targets[mask]) / targets[mask])) * 100
    else:
        mape = float('nan')

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


# ============================================================================
# Main Training Loop
# ============================================================================

def train(config, trial=None):
    """Main training function."""
    print("=" * 60)
    print("GCN-LSTM Training")
    print("=" * 60)

    if config.get('seed') is not None:
        set_global_seed(config['seed'], deterministic=config.get('deterministic', False))
        print(f"Seed: {config['seed']} (deterministic={config.get('deterministic', False)})")

    device = config['device']
    print(f"\nDevice: {device}")

    # Load data
    print("\n[1/6] Loading data...")
    X, Y, adj = load_data(config)
    adj = torch.FloatTensor(adj).to(device)

    # Update config with actual dimensions
    config['input_dim'] = X.shape[-1]
    config['num_nodes'] = X.shape[2]

    # Split data BEFORE scaling to prevent data leakage
    print("\n[2/6] Splitting data...")
    train_data, val_data, test_data = split_data(X, Y, config)
    X_train, Y_train = train_data
    X_val, Y_val = val_data
    X_test, Y_test = test_data

    # Fit scalers on training data only (prevents data leakage)
    print("\n[3/6] Fitting scalers on training data only...")
    feature_scaler, target_scaler, already_scaled = fit_scalers_on_train(X_train, Y_train, config)

    if already_scaled:
        # Data is already scaled, use as-is
        X_train_scaled, Y_train_scaled = X_train.astype(np.float32), Y_train.astype(np.float32)
        X_val_scaled, Y_val_scaled = X_val.astype(np.float32), Y_val.astype(np.float32)
        X_test_scaled, Y_test_scaled = X_test.astype(np.float32), Y_test.astype(np.float32)

        # Try to load existing target scaler for metrics inverse transform
        scaler_path = os.path.join(config['data_path'], 'target_scaler.save')
        if os.path.exists(scaler_path):
            target_scaler = joblib.load(scaler_path)
            print(f"  Loaded existing target scaler from {scaler_path}")
    else:
        # Scale all splits using training-fitted scalers
        X_train_scaled, Y_train_scaled = scale_data(X_train, Y_train, feature_scaler, target_scaler, config)
        X_val_scaled, Y_val_scaled = scale_data(X_val, Y_val, feature_scaler, target_scaler, config)
        X_test_scaled, Y_test_scaled = scale_data(X_test, Y_test, feature_scaler, target_scaler, config)

        print(f"  Feature scaler range: {feature_scaler.data_min_[:3]}... to {feature_scaler.data_max_[:3]}...")
        print(f"  Target scaler range: [{target_scaler.data_min_[0]:.2f}, {target_scaler.data_max_[0]:.2f}]")

        # Save scalers for inference
        os.makedirs(config['data_path'], exist_ok=True)
        joblib.dump(target_scaler, os.path.join(config['data_path'], 'target_scaler.save'))
        joblib.dump(feature_scaler, os.path.join(config['data_path'], 'feature_scaler.save'))

    # Derive EVT threshold from SCALED training targets to match loss computation
    if config.get('evt_threshold') is None:
        q = config['evt_tail_quantile']
        mode = config.get('evt_threshold_mode', 'global')
        if mode == 'global':
            config['evt_threshold'] = float(np.quantile(Y_train_scaled, q))
        elif mode == 'per_node':
            # Y_train_scaled: (samples, horizon, nodes) -> per-node threshold over (samples, horizon)
            config['evt_threshold'] = np.quantile(Y_train_scaled, q, axis=(0, 1)).astype(np.float32)
        else:
            raise ValueError(f"Unknown evt_threshold_mode: {mode}")

    # Create dataloaders with scaled data
    train_loader, val_loader, test_loader = create_dataloaders(
        (X_train_scaled, Y_train_scaled),
        (X_val_scaled, Y_val_scaled),
        (X_test_scaled, Y_test_scaled),
        config
    )

    # Create model
    print("\n[4/6] Creating model...")
    model_type = config.get('model_type', 'gcn_lstm')
    if model_type == 'graph_transformer':
        model = GraphTransformerModel(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            num_nodes=config['num_nodes'],
            num_tf_layers=config.get('num_tf_layers', 2),
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            horizon=config['horizon'],
            use_node_embeddings=config.get('use_node_embeddings', True),
            use_learnable_alpha_gate=config.get('use_learnable_alpha_gate', False),
            initial_wind_alpha=config.get('wind_alpha', 0.6),
            graph_conv=config.get('graph_conv', 'gcn'),
            num_gat_layers=config.get('num_gat_layers', 1),
            gat_version=config.get('gat_version', 'v1'),
            use_post_temporal_gat=config.get('use_post_temporal_gat', False),
            use_temporal_attention_head=config.get('use_temporal_attention_head', False),
        ).to(device)
        print(f"  Model type: GraphTransformerModel  graph_conv={config.get('graph_conv', 'gcn')}  gat_version={config.get('gat_version', 'v1')}  num_gat_layers={config.get('num_gat_layers', 1)}  post_gat={config.get('use_post_temporal_gat', False)}  temporal_attn_head={config.get('use_temporal_attention_head', False)}")
    else:
        model = GCNLSTMModel(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            num_nodes=config['num_nodes'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            horizon=config['horizon'],
            use_direct_decoding=config.get('use_direct_decoding', False),
            use_learnable_alpha_gate=config.get('use_learnable_alpha_gate', False),
            initial_wind_alpha=config.get('wind_alpha', 0.6),
            graph_conv=config.get('graph_conv', 'gcn'),
        ).to(device)
        print(f"  Model type: GCNLSTMModel  graph_conv={config.get('graph_conv', 'gcn')}")

    print(f"  Model parameters: {model.get_num_params():,}")
    if config.get('use_wind_adjacency', False) and config.get('use_learnable_alpha_gate', False):
        print(f"  Learnable Alpha Gate: ON (initial alpha={config.get('wind_alpha', 0.6):.3f})")
    
    # Loss and optimizer
    if config.get('loss_type', 'mse') == 'evt_hybrid':
        # Determine initial lambda (may be adjusted by schedule)
        if config.get('evt_use_lambda_schedule', False):
            initial_lambda = config['evt_lambda_schedule']['initial']
        else:
            initial_lambda = config['evt_lambda']

        criterion = EVTHybridLoss(
            threshold=config['evt_threshold'],
            lambda_tail=initial_lambda,
            xi=config['evt_xi'],
            asymmetric_penalty=config.get('evt_asymmetric_penalty', False),
            under_penalty_multiplier=config.get('evt_under_penalty_multiplier', 2.0)
        )

        # Print configuration
        thr = config['evt_threshold']
        if isinstance(thr, (list, tuple, np.ndarray)):
            thr_mean = float(np.mean(thr))
            thr_str = f"per_node(mean={thr_mean:.4f})"
        else:
            thr_str = f"{float(thr):.4f}"
        print(
            f"  Loss: EVT Hybrid (lambda={initial_lambda}, "
            f"q={config['evt_tail_quantile']}, xi={config['evt_xi']}, "
            f"threshold={thr_str})"
        )

        if config.get('evt_asymmetric_penalty', False):
            print(
                f"  Asymmetric Penalty: ON (under-prediction multiplier={config.get('evt_under_penalty_multiplier', 2.0)}x)"
            )

        if config.get('evt_use_lambda_schedule', False):
            sched = config['evt_lambda_schedule']
            print(
                f"  Adaptive Lambda Schedule: {sched['initial']} -> {sched['mid']} -> {sched['final']} "
                f"(warmup={sched['warmup_epochs']}, mid={sched['mid_epochs']}, {sched['transition']})"
            )
    else:
        criterion = nn.MSELoss()
        print("  Loss: MSE")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("\n[5/6] Training...")
    print("-" * 60)
    
    best_val_loss = float('inf')   # used only for LR scheduler
    best_val_mae = float('inf')    # used for early stopping and checkpoint selection
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    start_epoch = 0
    
    save_checkpoints = config.get('save_checkpoints', True)
    evaluate_test = config.get('evaluate_test', True)
    save_history = config.get('save_history', True)
    best_state_dict = None

    # Create checkpoint directory if needed
    if save_checkpoints:
        os.makedirs(config['model_save_path'], exist_ok=True)

    # Determine checkpoint filename
    if config.get('use_versioned_checkpoint', False):
        arch_name = config.get('architecture_name', 'model')
        hw_tag = config.get('hardware_tag', 'unknown')
        checkpoint_filename = f"{arch_name}_{hw_tag}_best.pt"
    else:
        checkpoint_filename = config['best_model_name']

    checkpoint_path = os.path.join(config['model_save_path'], checkpoint_filename)
    if save_checkpoints:
        print(f"\nCheckpoint path: {checkpoint_path}")
    else:
        print(f"\nCheckpoint path: {checkpoint_path} (disabled during this run)")

    # Resume from checkpoint if specified
    if save_checkpoints and config.get('resume', False) and os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        best_val_mae = checkpoint.get('val_mae', float('inf'))
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best val_loss: {best_val_loss:.6f}, best val_mae: {best_val_mae:.4f}")
    
    for epoch in range(start_epoch, config['epochs']):
        start_time = time.time()

        # Update EVT lambda if using adaptive schedule
        if config.get('loss_type', 'mse') == 'evt_hybrid' and config.get('evt_use_lambda_schedule', False):
            new_lambda = get_evt_lambda_for_epoch(
                epoch,
                config['evt_lambda_schedule'],
                config['epochs']
            )
            criterion.set_lambda(new_lambda)

            # Log lambda changes (only when it actually changes)
            if epoch == 0 or epoch == config['evt_lambda_schedule'].get('warmup_epochs', 20) or \
               epoch == config['evt_lambda_schedule'].get('mid_epochs', 40):
                print(f"  -> EVT lambda updated to {new_lambda:.4f} at epoch {epoch+1}")

        # Calculate teacher forcing ratio (linear decay)
        tf_ratio = config['teacher_forcing_start'] - \
                   (config['teacher_forcing_start'] - config['teacher_forcing_end']) * \
                   (epoch / config['epochs'])

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, adj, config, tf_ratio
        )

        # Validate — get both loss (for LR scheduler) and MAE (for early stopping)
        val_loss, val_mae = validate(model, val_loader, criterion, adj, config, target_scaler)

        # LR scheduler tracks val_loss.
        # NOTE: if evt_use_lambda_schedule is ever re-enabled, consider switching
        # this to val_mae — lambda changes alter val_loss scale independently of
        # model quality and can trigger spurious LR reductions.
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        elapsed = time.time() - start_time
        alpha_str = ""
        if config.get('use_wind_adjacency', False) and config.get('use_learnable_alpha_gate', False):
            alpha_str = f"Alpha: {float(model.get_wind_alpha().detach().cpu()):.3f} | "

        # Display current lambda if using schedule
        lambda_str = ""
        if config.get('loss_type', 'mse') == 'evt_hybrid' and config.get('evt_use_lambda_schedule', False):
            lambda_str = f"λ: {criterion.lambda_tail:.3f} | "

        scale_label = "µg/m³" if target_scaler is not None else "norm"
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val MAE: {val_mae:.4f} ({scale_label}) | "
              f"{alpha_str}"
              f"{lambda_str}"
              f"TF: {tf_ratio:.2f} | "
              f"Time: {elapsed:.1f}s")

        # Early stopping and checkpoint selection: use val_mae (loss-function-independent)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_loss = val_loss
            patience_counter = 0

            # Keep best weights in memory (needed when checkpoint saving is disabled)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if save_checkpoints:
                # Save best model with metadata
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'train_loss': train_loss,
                    'config': config,
                    'hardware': {
                        'device': str(device),
                        'tag': config.get('hardware_tag', 'unknown'),
                        'cuda_available': torch.cuda.is_available(),
                        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                    },
                    'architecture': {
                        'name': config.get('architecture_name', 'unknown'),
                        'num_params': model.get_num_params(),
                        'use_direct_decoding': config.get('use_direct_decoding', False),
                        'use_wind_adjacency': config.get('use_wind_adjacency', False),
                        'use_learnable_alpha_gate': config.get('use_learnable_alpha_gate', False),
                        'learned_wind_alpha': (
                            float(model.get_wind_alpha().detach().cpu())
                            if model.get_wind_alpha() is not None else None
                        )
                    },
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }, checkpoint_path)
                scale_label = "µg/m³" if target_scaler is not None else "norm"
                print(f"  -> Saved best model (val_mae: {val_mae:.4f} {scale_label})")
            else:
                scale_label = "µg/m³" if target_scaler is not None else "norm"
                print(f"  -> New best model in memory (val_mae: {val_mae:.4f} {scale_label})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Report intermediate value for Optuna and prune bad trials early
        # Use val_mae so pruning is loss-function-independent
        if trial is not None:
            trial.report(val_mae, step=epoch)
            if trial.should_prune():
                raise RuntimeError("TRIAL_PRUNED")
    
    # Restore best model weights
    if save_checkpoints and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    metrics = {'best_val_loss': best_val_loss, 'best_val_mae': best_val_mae}
    if evaluate_test:
        print("\n[6/6] Evaluating best model...")
        print("-" * 60)

        # Target scaler was already saved earlier and is in scope
        test_metrics = compute_metrics(model, test_loader, adj, config, target_scaler)
        metrics.update(test_metrics)

        print("\nTest Set Metrics:")
        for name, value in test_metrics.items():
            print(f"  {name}: {value:.4f}")
    else:
        print("\n[6/6] Skipping test evaluation (evaluate_test=False)")

    # Save training history
    if save_history:
        np.save(
            os.path.join(config['model_save_path'], 'history.npy'),
            history
        )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    if save_checkpoints:
        print(f"Best model saved to: {checkpoint_path}")
    else:
        print("Best model checkpoint saving was disabled for this run.")
    print(f"Architecture: {config.get('architecture_name', 'N/A')}")
    print(f"Hardware: {config.get('hardware_tag', 'N/A')}")
    print("=" * 60)

    return model, history, metrics


if __name__ == "__main__":
    model, history, metrics = train(CONFIG)
