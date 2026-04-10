"""
Baseline 3: MLP — simple feedforward network, no graph, no temporal modeling.

Flattens the entire input (24 timesteps x 12 nodes x 33 features) into a
single vector and predicts all horizons and nodes jointly.

This baseline answers: "How much does explicit spatial (GCN) and temporal
(LSTM) structure actually contribute? Can a plain MLP match the GCN-LSTM
just by having enough capacity to learn implicit patterns?"

Architecture:
    Input:  flatten(24 * 12 * 33) = 9504
    -> Linear(9504, 512) + LeakyReLU + Dropout
    -> Linear(512, 256) + LeakyReLU + Dropout
    -> Linear(256, 6 * 12) = 72
    -> Reshape to (horizon=6, nodes=12)

Training uses the same pipeline: chronological split, train-only scaling,
MSE loss, early stopping on val MAE in ug/m3.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from baselines.shared import (
    load_and_split, fit_and_scale, compute_metrics, print_metrics, SHARED_CONFIG
)


# ============================================================================
# Model
# ============================================================================

class MLPBaseline(nn.Module):
    """Simple feedforward network: flatten all input -> predict all outputs."""

    def __init__(self, input_len, num_nodes, input_dim, horizon, hidden1=512, hidden2=256, dropout=0.1):
        super().__init__()
        flat_in = input_len * num_nodes * input_dim
        flat_out = horizon * num_nodes

        self.net = nn.Sequential(
            nn.Linear(flat_in, hidden1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden2, flat_out),
        )
        self.horizon = horizon
        self.num_nodes = num_nodes

    def forward(self, x):
        """
        Args:
            x: (batch, input_len, num_nodes, input_dim)
        Returns:
            (batch, horizon, num_nodes)
        """
        batch = x.size(0)
        out = self.net(x.reshape(batch, -1))
        return out.reshape(batch, self.horizon, self.num_nodes)


# ============================================================================
# Training
# ============================================================================

CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'epochs': 100,
    'patience': 15,
    'hidden1': 512,
    'hidden2': 256,
    'dropout': 0.1,
    'seed': 42,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("=" * 60)
    print("Baseline 3: MLP (no graph, no temporal structure)")
    print("=" * 60)

    set_seed(CONFIG['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load and split
    train_raw, val_raw, test_raw, adj = load_and_split()

    # Fit scalers on train, transform all
    train_s, val_s, test_s, feature_scaler, target_scaler = fit_and_scale(
        train_raw, val_raw, test_raw
    )

    X_train, Y_train = train_s
    X_val, Y_val = val_s
    X_test, Y_test = test_s

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
        batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val)),
        batch_size=CONFIG['batch_size'], shuffle=False
    )

    # Model
    input_len = SHARED_CONFIG['input_len']
    num_nodes = SHARED_CONFIG['num_nodes']
    input_dim = X_train.shape[-1]
    horizon = SHARED_CONFIG['horizon']

    model = MLPBaseline(
        input_len=input_len,
        num_nodes=num_nodes,
        input_dim=input_dim,
        horizon=horizon,
        hidden1=CONFIG['hidden1'],
        hidden2=CONFIG['hidden2'],
        dropout=CONFIG['dropout'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_mae = float('inf')
    patience_counter = 0
    best_state = None

    print(f"\nTraining for up to {CONFIG['epochs']} epochs (patience={CONFIG['patience']})...")
    print("-" * 60)

    for epoch in range(CONFIG['epochs']):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        all_preds, all_targets = [], []
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                preds = model(X_batch)
                val_loss += criterion(preds, Y_batch).item()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(Y_batch.cpu().numpy())
        val_loss /= len(val_loader)

        # Val MAE in original scale
        preds_cat = np.concatenate(all_preds)
        targets_cat = np.concatenate(all_targets)
        preds_inv = target_scaler.inverse_transform(preds_cat.reshape(-1, 1)).reshape(preds_cat.shape)
        targets_inv = target_scaler.inverse_transform(targets_cat.reshape(-1, 1)).reshape(targets_cat.shape)
        val_mae = float(np.mean(np.abs(preds_inv - targets_inv)))

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"Val MAE: {val_mae:.2f} ug/m3 | {elapsed:.1f}s")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> New best (val MAE: {val_mae:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), CONFIG['batch_size']):
            batch = torch.FloatTensor(X_test[i:i+CONFIG['batch_size']]).to(device)
            preds = model(batch)
            all_preds.append(preds.cpu().numpy())

    preds_test = np.concatenate(all_preds)
    metrics = compute_metrics(preds_test, Y_test, target_scaler)
    print_metrics(metrics, "MLP Baseline")

    # Save checkpoint
    os.makedirs('models/checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {**SHARED_CONFIG, **CONFIG},
        'val_mae': best_val_mae,
        'metrics': metrics,
    }, 'models/checkpoints/baseline_mlp_best.pt')
    print("Checkpoint saved to models/checkpoints/baseline_mlp_best.pt")

    return metrics


if __name__ == "__main__":
    main()
