"""
Baseline 4: LSTM — standard encoder-decoder LSTM, no graph structure.

Each station is processed independently through a shared LSTM encoder-decoder.
This is architecturally identical to the GCN-LSTM but with the GCN removed:
the LSTM gates receive the raw projected input instead of GCN-aggregated input.

This baseline directly quantifies the contribution of graph (spatial) structure.
If the GCN-LSTM models only marginally beat this, then the graph is not helping
much for this dataset. If there is a large gap, graph structure is justified.

Architecture:
    Encoder: 2-layer LSTM (hidden_dim=64) over 24 timesteps, per node
    Decoder: Direct multi-horizon — one MLP head per horizon step
             (same idea as DirectMultiHorizonDecoder but without GCN)

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

class LSTMBaseline(nn.Module):
    """
    LSTM encoder-decoder without graph convolution.

    Processes each node independently through a shared LSTM encoder,
    then predicts all horizons via a direct (non-autoregressive) decoder.
    Nodes share parameters but do not exchange information spatially.

    Input:  (batch, input_len, num_nodes, input_dim)
    Output: (batch, horizon, num_nodes)
    """

    def __init__(self, input_dim, hidden_dim, num_nodes, num_layers, horizon, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.horizon = horizon

        # Input projection (same as GCN-LSTM encoder)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Standard LSTM encoder (shared across all nodes)
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Direct multi-horizon decoder: one head per horizon step
        # Each head takes the final encoder hidden state and predicts one scalar per node
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(horizon)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, input_len, num_nodes, input_dim)
        Returns:
            (batch, horizon, num_nodes)
        """
        batch, seq_len, num_nodes, feat = x.shape

        # Reshape: treat each node as a separate sequence in the batch
        # (batch * num_nodes, seq_len, input_dim)
        x_flat = x.permute(0, 2, 1, 3).reshape(batch * num_nodes, seq_len, feat)

        # Project input
        x_proj = self.input_proj(x_flat)  # (batch * nodes, seq_len, hidden_dim)

        # Encode
        encoder_out, (h_n, c_n) = self.encoder(x_proj)
        # h_n: (num_layers, batch * nodes, hidden_dim)

        # Use top-layer hidden state as the representation
        h_final = h_n[-1]  # (batch * nodes, hidden_dim)
        h_final = self.dropout(h_final)

        # Decode each horizon step independently
        outputs = []
        for head in self.horizon_heads:
            out = head(h_final)  # (batch * nodes, 1)
            outputs.append(out)

        # Stack: (batch * nodes, horizon, 1) -> (batch, nodes, horizon) -> (batch, horizon, nodes)
        out = torch.cat(outputs, dim=-1)  # (batch * nodes, horizon)
        out = out.reshape(batch, num_nodes, self.horizon)  # (batch, nodes, horizon)
        out = out.permute(0, 2, 1)  # (batch, horizon, nodes)

        return out


# ============================================================================
# Training
# ============================================================================

CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'epochs': 100,
    'patience': 15,
    'hidden_dim': 64,
    'num_layers': 2,
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
    print("Baseline 4: LSTM (no graph structure)")
    print("  Same LSTM architecture but without GCN spatial aggregation")
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
    input_dim = X_train.shape[-1]
    horizon = SHARED_CONFIG['horizon']
    num_nodes = SHARED_CONFIG['num_nodes']

    model = LSTMBaseline(
        input_dim=input_dim,
        hidden_dim=CONFIG['hidden_dim'],
        num_nodes=num_nodes,
        num_layers=CONFIG['num_layers'],
        horizon=horizon,
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
    print_metrics(metrics, "LSTM Baseline (no graph)")

    # Save checkpoint
    os.makedirs('models/checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {**SHARED_CONFIG, **CONFIG},
        'val_mae': best_val_mae,
        'metrics': metrics,
    }, 'models/checkpoints/baseline_lstm_best.pt')
    print("Checkpoint saved to models/checkpoints/baseline_lstm_best.pt")

    return metrics


if __name__ == "__main__":
    main()
