"""
MAE Pretraining for SpatioTemporalTransformerEncoder.

Trains the Transformer encoder using masked autoencoding before supervised fine-tuning.
75% of the 24 input timesteps are randomly masked (zeroed) per sample; the encoder
reconstructs the original input features at masked positions.

Only the training split (first 70%) is used — no val/test data leakage.
The reconstruction head is discarded after pretraining; only encoder weights are saved.

Saved checkpoint: models/checkpoints/pretrained_encoder_mae.pt

Usage:
    python pretrain_mae.py
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.transformer_model import SpatioTemporalTransformerEncoder

# ============================================================================
# Config — architecture params must match the supervised Seg-MoE config
# ============================================================================

PRETRAIN_CONFIG = {
    # Data
    'data_path': 'data/processed/',
    'input_len': 24,
    'train_ratio': 0.7,
    'log_transform_indices': [0, 1, 2, 3, 4, 5],  # must match train.py
    'wind_dir_start_idx': 17,                       # must match train.py

    # Architecture — must match supervised Seg-MoE config exactly
    'input_dim': 33,
    'hidden_dim': 64,
    'num_nodes': 12,
    'num_tf_layers': 2,
    'num_heads': 4,
    'dropout': 0.1,
    'use_node_embeddings': True,
    'graph_conv': 'gat',
    'num_gat_layers': 1,
    'gat_version': 'v1',
    'use_temporal_first': True,
    'use_seg_moe': True,

    # Pretraining hyperparams
    'mask_ratio': 0.75,       # fraction of timesteps to mask per sample (MAE standard)
    'batch_size': 64,
    'learning_rate': 1e-3,
    'epochs': 20,

    # Output
    'save_path': 'models/checkpoints/pretrained_encoder_mae.pt',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
}


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_scale_train(cfg: dict) -> np.ndarray:
    """
    Load X_24.npy, apply log1p on skewed features, fit StandardScaler on train split,
    return scaled training X as float32 array of shape (n_train, T, N, F).

    Mirrors the train.py preprocessing pipeline exactly:
      - log1p applied to indices 0-5
      - StandardScaler fitted and applied to features 0:17
      - Wind one-hot features (17:33) left unscaled
    """
    x_path = os.path.join(cfg['data_path'], f"X_{cfg['input_len']}.npy")
    X = np.load(x_path)               # (samples, T, N, F)
    print(f"Loaded X: {X.shape}")

    n_train = int(len(X) * cfg['train_ratio'])
    X_train = X[:n_train]
    print(f"Training split: {n_train} samples ({100*cfg['train_ratio']:.0f}%)")

    wind_start = cfg['wind_dir_start_idx']  # 17

    # Apply log1p — same indices as train.py
    X_log = X_train.copy()
    for idx in cfg['log_transform_indices']:
        X_log[:, :, :, idx] = np.log1p(X_train[:, :, :, idx])

    # Fit StandardScaler on non-wind features (0:17) using training data only
    n_s, T, N, F = X_log.shape
    X_flat = X_log.reshape(-1, F)                          # (n_train*T*N, F)
    scaler = StandardScaler()
    scaler.fit(X_flat[:, :wind_start])

    # Apply scaler; leave wind one-hot unchanged
    X_scaled_nonwind = scaler.transform(X_flat[:, :wind_start])
    X_scaled = np.concatenate([X_scaled_nonwind, X_flat[:, wind_start:]], axis=-1)
    return X_scaled.reshape(n_s, T, N, F).astype(np.float32)


def apply_random_mask(x: torch.Tensor, mask_ratio: float):
    """
    Randomly mask mask_ratio fraction of timesteps per sample by zeroing them.

    Args:
        x: (B, T, N, F) input tensor
        mask_ratio: fraction of timesteps to mask
    Returns:
        x_masked: (B, T, N, F) with masked timesteps zeroed
        mask:     (B, T) bool — True = masked
    """
    B, T, N, F = x.shape
    num_masked = int(T * mask_ratio)

    # Random timestep ordering per sample; take first num_masked as masked
    noise = torch.rand(B, T, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    mask = torch.zeros(B, T, device=x.device)
    mask.scatter_(1, ids_shuffle[:, :num_masked], 1.0)
    mask = mask.bool()

    x_masked = x.clone()
    x_masked[mask.unsqueeze(-1).unsqueeze(-1).expand_as(x)] = 0.0
    return x_masked, mask


# ============================================================================
# Main
# ============================================================================

def pretrain(cfg: dict):
    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])
    print(f"Device: {device}\n")

    # --- Data ---
    X_train = load_and_scale_train(cfg)
    dataset = TensorDataset(torch.from_numpy(X_train))
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=False)

    # Static distance-based adjacency (no wind dynamics — keeps pretraining simple)
    adj_np = np.load(os.path.join(cfg['data_path'], 'adjacency.npy')).astype(np.float32)
    adj = torch.from_numpy(adj_np).to(device)      # (N, N)

    # --- Encoder ---
    # return_full_sequence=True → forward returns (B, N, T, H) instead of (B, N, H).
    # The flag is a Python attribute, not a parameter, so state_dict is identical
    # to the supervised encoder (return_full_sequence=False) — loading is safe.
    encoder = SpatioTemporalTransformerEncoder(
        input_dim=cfg['input_dim'],
        hidden_dim=cfg['hidden_dim'],
        num_nodes=cfg['num_nodes'],
        num_tf_layers=cfg['num_tf_layers'],
        num_heads=cfg['num_heads'],
        dropout=cfg['dropout'],
        use_node_embeddings=cfg['use_node_embeddings'],
        graph_conv=cfg['graph_conv'],
        num_gat_layers=cfg['num_gat_layers'],
        gat_version=cfg['gat_version'],
        use_temporal_first=cfg['use_temporal_first'],
        use_seg_moe=cfg['use_seg_moe'],
        return_full_sequence=True,   # needed to get full sequence for reconstruction
        input_len=cfg['input_len'],
    ).to(device)

    # Reconstruction head: H → F.  Zero-init → pure encoder signal at epoch 0.
    # This head is NOT saved — it is discarded after pretraining.
    recon_head = nn.Linear(cfg['hidden_dim'], cfg['input_dim']).to(device)
    nn.init.zeros_(recon_head.weight)
    nn.init.zeros_(recon_head.bias)

    n_enc_params = sum(p.numel() for p in encoder.parameters())
    n_head_params = sum(p.numel() for p in recon_head.parameters())
    print(f"Encoder params: {n_enc_params:,}")
    print(f"Reconstruction head params: {n_head_params:,}  (discarded after pretraining)")
    print(f"\nPretraining: {cfg['epochs']} epochs, mask_ratio={cfg['mask_ratio']}, "
          f"batch_size={cfg['batch_size']}, lr={cfg['learning_rate']}\n")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(recon_head.parameters()),
        lr=cfg['learning_rate'],
    )

    for epoch in range(1, cfg['epochs'] + 1):
        encoder.train()
        recon_head.train()
        total_loss = 0.0
        n_batches = 0

        for (x_batch,) in loader:
            x_batch = x_batch.to(device)                       # (B, T, N, F)
            x_masked, mask = apply_random_mask(x_batch, cfg['mask_ratio'])

            # Encoder forward: (B, T, N, F) → (B, N, T, H) when return_full_sequence=True
            enc_seq = encoder(x_masked, adj)                   # (B, N, T, H)
            enc_seq = enc_seq.permute(0, 2, 1, 3)              # (B, T, N, H)
            recon = recon_head(enc_seq)                        # (B, T, N, F)

            # MSE only on masked timesteps
            mask_exp = mask.unsqueeze(-1).unsqueeze(-1).expand_as(recon)
            loss = F.mse_loss(recon[mask_exp], x_batch[mask_exp])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(recon_head.parameters()), 1.0
            )
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch:3d}/{cfg['epochs']}  recon_loss={total_loss / n_batches:.6f}")

    # Save encoder weights only — reconstruction head is discarded
    os.makedirs(os.path.dirname(cfg['save_path']), exist_ok=True)
    torch.save(encoder.state_dict(), cfg['save_path'])
    print(f"\nPretrained encoder saved → {cfg['save_path']}")


if __name__ == '__main__':
    pretrain(PRETRAIN_CONFIG)
