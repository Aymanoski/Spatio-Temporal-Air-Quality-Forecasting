"""
Inference timing benchmark for the best Seg-MoE checkpoint.

Measures wall-clock time for the full inference loop over the test set,
including dynamic wind adjacency construction (which is part of deployment cost).

Usage:
    python utils/time_inference.py
    python utils/time_inference.py --runs 3
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tester import (
    DATA_PATH,
    evaluate_checkpoint,
    fit_scalers_on_train,
    load_model_for_checkpoint,
    load_raw_data_for_config,
    prepare_checkpoint_config,
    run_model_predictions,
    align_feature_tensor,
    get_current_feature_cols,
    infer_variant_feature,
    infer_checkpoint_feature_cols,
    refresh_feature_config,
    set_per_station_residual_tensors,
)
from train import split_data, scale_data

CHECKPOINT = Path(
    "models/checkpoints/transformer/"
    "graph_transformer_gat_v1_residual_log1p_all_std_stationbias_temporal_first_SEgmoe_T4_best.pt"
)


def time_inference(checkpoint_path: Path, data_path: Path, device: str, runs: int = 3) -> dict:
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    config = prepare_checkpoint_config(checkpoint, device=device)

    raw_X, raw_Y, adj, raw_Z = load_raw_data_for_config(data_path, config)
    variant_feature = infer_variant_feature(config)
    current_feature_cols = get_current_feature_cols(
        data_path,
        observed_feature_dim=int(raw_X.shape[-1]),
        variant_feature=variant_feature,
    )
    checkpoint_feature_cols = infer_checkpoint_feature_cols(config, current_feature_cols)
    refresh_feature_config(config, checkpoint_feature_cols)
    aligned_X = align_feature_tensor(raw_X, current_feature_cols, checkpoint_feature_cols)

    train_data, val_data, test_data = split_data(aligned_X, raw_Y, config)
    X_train, Y_train = train_data
    X_test, Y_test = test_data

    feature_scaler, target_scaler, already_scaled = fit_scalers_on_train(X_train, Y_train, config)
    set_per_station_residual_tensors(config, feature_scaler, target_scaler, device)

    if already_scaled:
        X_test_scaled = X_test.astype(np.float32)
    else:
        from train import scale_data as _scale
        X_test_scaled, _ = _scale(X_test, Y_test, feature_scaler, target_scaler, config)

    model, _ = load_model_for_checkpoint(checkpoint_path, config, device)
    model.eval()

    n_samples = len(X_test_scaled)
    batch_size = int(config.get("batch_size", 32))
    n_batches = (n_samples + batch_size - 1) // batch_size

    # Warm-up pass (not timed) to initialise any lazy state
    _ = run_model_predictions(model, X_test_scaled[:batch_size], adj, config, device=device)

    elapsed_times = []
    for run_idx in range(runs):
        t0 = time.perf_counter()
        _ = run_model_predictions(model, X_test_scaled, adj, config, device=device)
        t1 = time.perf_counter()
        elapsed_times.append(t1 - t0)
        print(f"  Run {run_idx + 1}/{runs}: {t1 - t0:.3f}s")

    mean_s   = float(np.mean(elapsed_times))
    min_s    = float(np.min(elapsed_times))
    ms_per_sample = mean_s / n_samples * 1000.0
    ms_per_batch  = mean_s / n_batches * 1000.0

    return {
        "n_samples": n_samples,
        "n_batches": n_batches,
        "batch_size": batch_size,
        "device": device,
        "runs": runs,
        "mean_total_s": mean_s,
        "min_total_s": min_s,
        "ms_per_sample": ms_per_sample,
        "ms_per_batch": ms_per_batch,
        "elapsed_times": elapsed_times,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs (default 3)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    data_path = DATA_PATH

    print("=" * 60)
    print("Inference Timing Benchmark")
    print("=" * 60)
    print(f"Checkpoint : {checkpoint_path.name}")
    print(f"Device     : {device}")
    print(f"Runs       : {args.runs}")
    print()

    r = time_inference(checkpoint_path, data_path, device, runs=args.runs)

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Test samples      : {r['n_samples']}")
    print(f"Batches           : {r['n_batches']}  (batch_size={r['batch_size']})")
    print(f"Mean total time   : {r['mean_total_s']:.3f} s")
    print(f"Best total time   : {r['min_total_s']:.3f} s")
    print(f"Mean per sample   : {r['ms_per_sample']:.3f} ms")
    print(f"Mean per batch    : {r['ms_per_batch']:.3f} ms")
    print()
    print("NOTE: STGATN reports 37.0s inference on GPU (batch_size=32).")
    print("      This benchmark runs on", device, "— hardware differs.")


if __name__ == "__main__":
    main()
