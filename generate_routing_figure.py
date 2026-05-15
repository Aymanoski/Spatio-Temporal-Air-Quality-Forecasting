"""
generate_routing_figure.py
Generates the Seg-MoE expert routing visualization for the thesis.

Loads the confirmed best checkpoint, hooks the SegMoE router on the test set,
and plots the high-PM2.5 expert weight (g2) against the window-mean PM2.5.

Run:  python generate_routing_figure.py
Output: Figures/thesis/fig_segmoe_routing.pdf  (+ .png)
"""

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from train import (
    CONFIG as TRAIN_CONFIG,
    build_dynamic_adjacency,
    fit_scalers_on_train,
    scale_data,
    split_data,
)
from utils.window import create_windows
from utils.tester import build_model_from_config

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("data/processed")
CKPT_PATH  = Path("models/checkpoints/transformer/"
                  "graph_transformer_gat_v1_residual_log1p_all_std_stationbias"
                  "_temporal_first_SEgmoe_T4_best.pt")
OUT_DIR    = Path("Figures/thesis")

# ── Style (match generate_thesis_figures.py) ──────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'legend.fontsize':  10,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top':  False,
    'axes.spines.right': False,
})
PRIMARY = '#2C6FAC'


def build_config(ckpt: dict) -> dict:
    """Merge checkpoint config into TRAIN_CONFIG, setting required flags."""
    cfg = {**TRAIN_CONFIG, **ckpt.get("config", {})}
    cfg["device"] = "cpu"
    # Flags the tester detects from state-dict keys
    sd = ckpt["model_state_dict"]
    cfg["use_seg_moe"]            = any(k.startswith("encoder.transformer.layers.0.moe_ffn.") for k in sd)
    cfg["use_temporal_first"]     = bool(cfg.get("use_temporal_first", False))
    cfg["use_future_met"]         = "head.future_met_proj.weight" in sd
    cfg["future_met_dim"]         = 0
    cfg["use_station_horizon_bias"] = "station_horizon_bias" in sd
    cfg["use_persistence_residual"] = bool(cfg.get("use_persistence_residual", True))
    cfg["use_wind_adjacency"]     = bool(cfg.get("use_wind_adjacency", True))
    cfg["use_per_station_norm"]   = False
    cfg["use_revin"]              = False
    cfg["use_trend_residual"]     = False
    return cfg


def main():
    device = torch.device("cpu")

    print("Loading checkpoint …")
    ckpt   = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    config = build_config(ckpt)

    print("Building model …")
    model = build_model_from_config(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    print("Loading and splitting data …")
    data_tensor = np.load(DATA_PATH / "data_tensor.npy")
    X, Y = create_windows(data_tensor,
                           input_len=int(config["input_len"]),
                           horizon=int(config["horizon"]))
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    (X_train, Y_train), _, (X_test, Y_test) = split_data(X, Y, config)

    print("Fitting scalers on training split …")
    feature_scaler, target_scaler, _ = fit_scalers_on_train(X_train, Y_train, config)

    X_test_sc, _ = scale_data(X_test, Y_test, feature_scaler, target_scaler, config)

    # PM2.5 scaler stats (feature index 0, in log1p space)
    pm25_mu    = float(feature_scaler.mean_[0])
    pm25_sigma = float(feature_scaler.scale_[0])

    # ── Register forward hook on the SegMoE router ────────────────────────────
    # encoder.transformer.layers[0].moe_ffn.router captures pm25_mean_flat input
    # and raw logits output; we softmax after collection.
    router_inputs  = []
    router_logits  = []

    router_module = model.encoder.transformer.layers[0].moe_ffn.router

    def _hook(module, inp, out):
        router_inputs.append(inp[0].detach().cpu())
        router_logits.append(out.detach().cpu())

    handle = router_module.register_forward_hook(_hook)

    print("Running test-set inference …")
    batch_size = 64
    X_tensor   = torch.from_numpy(X_test_sc)

    with torch.no_grad():
        for start in range(0, len(X_test_sc), batch_size):
            batch_x = X_tensor[start : start + batch_size].to(device)
            alpha_ov  = model.get_wind_alpha()  if hasattr(model, "get_wind_alpha")  else None
            sigma_ov  = model.get_distance_sigma() if hasattr(model, "get_distance_sigma") else None
            static_ov = model.get_static_adj()  if hasattr(model, "get_static_adj")  else None
            adj_b = build_dynamic_adjacency(
                batch_x, config, device,
                alpha_override=alpha_ov,
                sigma_override=sigma_ov,
                static_adj_override=static_ov,
            )
            # model.predict handles persistence residual internally
            model.predict(batch_x, adj_b,
                          horizon=int(config["horizon"]),
                          future_met=None)

    handle.remove()

    # ── Reconstruct routing weights and PM2.5 values ──────────────────────────
    all_inputs  = torch.cat(router_inputs, dim=0).squeeze(-1).numpy()   # (total,)
    all_logits  = torch.cat(router_logits, dim=0).numpy()               # (total, 2)
    r_weights   = torch.softmax(torch.from_numpy(all_logits), dim=-1).numpy()
    r_high      = r_weights[:, 1]  # weight for the high-PM2.5 expert (g2)

    # Inverse transform: scaled -> log1p -> μg/m³
    pm25_log1p  = all_inputs * pm25_sigma + pm25_mu
    pm25_ug     = np.clip(np.expm1(pm25_log1p), 0, None)

    print(f"  Total router calls: {len(r_high):,}")
    print(f"  PM2.5 range: [{pm25_ug.min():.1f}, {pm25_ug.max():.1f}] μg/m³")
    print(f"  r_high range: [{r_high.min():.3f}, {r_high.max():.3f}]")

    # ── Compute binned mean for trend line ────────────────────────────────────
    bin_edges  = np.percentile(pm25_ug, np.linspace(0, 100, 31))
    bin_edges  = np.unique(bin_edges)
    bin_idx    = np.digitize(pm25_ug, bin_edges) - 1
    bin_idx    = np.clip(bin_idx, 0, len(bin_edges) - 2)
    bin_centers, bin_means = [], []
    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        if mask.sum() > 20:
            bin_centers.append(0.5 * (bin_edges[b] + bin_edges[b + 1]))
            bin_means.append(r_high[mask].mean())

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))

    hb = ax.hexbin(pm25_ug, r_high,
                   gridsize=55, mincnt=5,
                   cmap='Blues', linewidths=0.15)
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label('Sample count', fontsize=10)

    if bin_centers:
        ax.plot(bin_centers, bin_means,
                color='#E07B39', linewidth=2.0,
                label='Binned mean $g_2$', zorder=5)
        ax.legend(fontsize=10)

    ax.axhline(0.5, color='#7B7B7B', linestyle='--',
               linewidth=0.9, alpha=0.8, label='_nolegend_')
    ax.set_xlabel(r'Window-mean PM$_{2.5}$ ($\mu$g m$^{-3}$)', fontsize=11)
    ax.set_ylabel(r'Router weight — high-regime expert ($g_2$)', fontsize=11)
    ax.set_title('Seg-MoE routing behaviour across pollution regimes', fontsize=12)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_segmoe_routing.{ext}", dpi=300)
    plt.close()
    print("Saved  Figures/thesis/fig_segmoe_routing.pdf  and  .png")


if __name__ == "__main__":
    main()
