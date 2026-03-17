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
import joblib
from models import GCNLSTMModel
from utils.graph import build_wind_aware_adjacency_batch, WIND_CATEGORIES

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
    
    # Training
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'epochs': 100,
    'patience': 25,          # Early stopping patience (increased for lambda adaptation)
    'teacher_forcing_start': 1.0,  # Initial teacher forcing ratio
    'teacher_forcing_end': 0.0,    # Final teacher forcing ratio

    # Loss
    'loss_type': 'evt_hybrid',     # Options: 'mse' or 'evt_hybrid'
    'evt_lambda': 0.05,            # Base weight of EVT tail component (used if no schedule)
    'evt_tail_quantile': 0.90,     # Threshold quantile for extremes
    'evt_xi': 0.10,                # GPD shape parameter
    'evt_threshold': None,         # Populated from training targets

    # EVT Improvements: Asymmetric Penalty + Adaptive Lambda
    'evt_asymmetric_penalty': True,         # Penalize under-prediction more than over-prediction
    'evt_under_penalty_multiplier': 2.0,    # Multiplier for under-prediction errors
    'evt_use_lambda_schedule': True,        # Use adaptive lambda scheduling
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
    'distance_sigma': 100,         # Distance decay parameter
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
    'architecture_name': 'gcn_lstm_v3',     # v3: v2 + asymmetric penalty + adaptive lambda
    'hardware_tag': 'T4',       # Options: 'integrated_gpu', 'T4', 'rtx3090', etc.
    'use_versioned_checkpoint': True,       # If True, saves as <arch>_<hardware>_best.pt

    # Resume training
    'resume': False         # Set to True to resume from checkpoint
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
        self.threshold = float(threshold)
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
        threshold = targets.new_tensor(self.threshold)
        mask = targets > threshold

        if mask.any():
            # Get predictions and targets for extreme values
            extreme_preds = predictions[mask]
            extreme_targets = targets[mask]

            # Compute excesses over threshold
            target_excess = extreme_targets - threshold
            pred_excess = torch.clamp(extreme_preds - threshold, min=0.0)  # Ensure non-negative

            # Weighted MSE on excesses
            # For extreme values, errors are weighted more heavily
            # Weight increases with the magnitude of the excess
            excess_weight = 1.0 + self.xi * target_excess / (target_excess.mean() + self.eps)

            # Asymmetric penalty: penalize under-prediction more
            if self.asymmetric_penalty:
                # Under-prediction: predicted excess < target excess
                under_mask = pred_excess < target_excess
                excess_weight = excess_weight.clone()  # Avoid in-place modification
                excess_weight[under_mask] *= self.under_penalty_multiplier

            weighted_excess_error = excess_weight * (pred_excess - target_excess) ** 2
            tail_loss = weighted_excess_error.mean()
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


# ============================================================================
# Data Loading
# ============================================================================

def load_data(config):
    """Load preprocessed data tensors."""
    data_path = config['data_path']

    # Load feature tensor and adjacency matrix
    X = np.load(os.path.join(data_path, 'X.npy'))
    Y = np.load(os.path.join(data_path, 'Y.npy'))
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
        alpha=config['wind_alpha'],
        distance_sigma=config['distance_sigma']
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
            adj_batch = build_dynamic_adjacency(X_batch, config, device)
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
        
        # Compute loss
        loss = criterion(predictions, Y_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, adj, config):
    """Validate model."""
    model.eval()
    total_loss = 0
    device = config['device']
    use_wind_adj = config.get('use_wind_adjacency', False)

    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Build dynamic adjacency if enabled
            if use_wind_adj:
                adj_batch = build_dynamic_adjacency(X_batch, config, device)
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
            loss = criterion(predictions, Y_batch)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def compute_metrics(model, test_loader, adj, config, target_scaler=None):
    """Compute evaluation metrics on test set."""
    model.eval()
    device = config['device']
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            
            predictions = model.predict(X_batch, adj, horizon=config['horizon'])
            
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
    
    # MAPE (avoid division by zero)
    mask = targets > 0
    mape = np.mean(np.abs((preds[mask] - targets[mask]) / targets[mask])) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


# ============================================================================
# Main Training Loop
# ============================================================================

def train(config):
    """Main training function."""
    print("=" * 60)
    print("GCN-LSTM Training")
    print("=" * 60)
    
    device = config['device']
    print(f"\nDevice: {device}")
    
    # Load data
    print("\n[1/5] Loading data...")
    X, Y, adj = load_data(config)
    adj = torch.FloatTensor(adj).to(device)
    
    # Update config with actual dimensions
    config['input_dim'] = X.shape[-1]
    config['num_nodes'] = X.shape[2]
    
    # Split data
    print("\n[2/5] Splitting data...")
    train_data, val_data, test_data = split_data(X, Y, config)

    # Derive EVT threshold from training targets only to avoid leakage
    if config.get('evt_threshold') is None:
        _, Y_train = train_data
        config['evt_threshold'] = float(np.quantile(Y_train, config['evt_tail_quantile']))

    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, config
    )
    
    # Create model
    print("\n[3/5] Creating model...")
    model = GCNLSTMModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        num_nodes=config['num_nodes'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        horizon=config['horizon'],
        use_direct_decoding=config.get('use_direct_decoding', False)
    ).to(device)
    
    print(f"  Model parameters: {model.get_num_params():,}")
    
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
        print(
            f"  Loss: EVT Hybrid (lambda={initial_lambda}, "
            f"q={config['evt_tail_quantile']}, xi={config['evt_xi']}, "
            f"threshold={config['evt_threshold']:.4f})"
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
    print("\n[4/5] Training...")
    print("-" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    start_epoch = 0
    
    # Create checkpoint directory
    os.makedirs(config['model_save_path'], exist_ok=True)

    # Determine checkpoint filename
    if config.get('use_versioned_checkpoint', False):
        arch_name = config.get('architecture_name', 'model')
        hw_tag = config.get('hardware_tag', 'unknown')
        checkpoint_filename = f"{arch_name}_{hw_tag}_best.pt"
    else:
        checkpoint_filename = config['best_model_name']

    checkpoint_path = os.path.join(config['model_save_path'], checkpoint_filename)
    print(f"\nCheckpoint path: {checkpoint_path}")

    # Resume from checkpoint if specified
    if config.get('resume', False) and os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best validation loss: {best_val_loss:.6f}")
    
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

        # Validate
        val_loss = validate(model, val_loader, criterion, adj, config)

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        elapsed = time.time() - start_time

        # Display current lambda if using schedule
        lambda_str = ""
        if config.get('loss_type', 'mse') == 'evt_hybrid' and config.get('evt_use_lambda_schedule', False):
            lambda_str = f"λ: {criterion.lambda_tail:.3f} | "

        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"{lambda_str}"
              f"TF: {tf_ratio:.2f} | "
              f"Time: {elapsed:.1f}s")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model with metadata
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
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
                    'use_wind_adjacency': config.get('use_wind_adjacency', False)
                },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, checkpoint_path)
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model for evaluation
    print("\n[5/5] Evaluating best model...")
    print("-" * 60)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load target scaler for inverse transform
    target_scaler = None
    scaler_path = os.path.join(config['data_path'], 'target_scaler.save')
    if os.path.exists(scaler_path):
        target_scaler = joblib.load(scaler_path)
    
    # Compute metrics
    metrics = compute_metrics(model, test_loader, adj, config, target_scaler)
    
    print("\nTest Set Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Save training history
    np.save(
        os.path.join(config['model_save_path'], 'history.npy'),
        history
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model saved to: {checkpoint_path}")
    print(f"Architecture: {config.get('architecture_name', 'N/A')}")
    print(f"Hardware: {config.get('hardware_tag', 'N/A')}")
    print("=" * 60)

    return model, history, metrics


if __name__ == "__main__":
    model, history, metrics = train(CONFIG)
