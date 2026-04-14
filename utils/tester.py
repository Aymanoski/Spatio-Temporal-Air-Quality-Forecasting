"""
Flexible evaluation script for checkpointed GCN-LSTM models.

This tester is designed to:
1. Evaluate any checkpoint in ``models/checkpoints`` whose architecture is still
   representable by the current codebase.
2. Reconstruct the checkpoint's expected feature layout from metadata/config.
3. Refit scalers on the training split only, matching the training pipeline.
4. Tolerate backward-compatible architecture changes such as optional alpha
   gates, learnable sigma, node embeddings, and future config-driven upgrades.

Examples:
    python utils/tester.py
    python utils/tester.py --checkpoint models/checkpoints/alpha__best.pt
    python utils/tester.py --all
    python utils/tester.py --all --output-json models/checkpoints/eval_all.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

# Add parent directory to path for model/training imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_model
from train import (
    CONFIG as TRAIN_CONFIG,
    build_dynamic_adjacency,
    fit_scalers_on_train,
    scale_data,
    split_data,
)
from utils.graph import WIND_DIRECTION_MAP


DATA_PATH = Path("data/processed")
CHECKPOINTS_DIR = Path("models/checkpoints")
DEFAULT_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pt"
MAPE_THRESHOLD = 5.0

STATION_ORDER = [
    "Aotizhongxin",
    "Changping",
    "Dingling",
    "Dongsi",
    "Guanyuan",
    "Gucheng",
    "Huairou",
    "Nongzhanguan",
    "Shunyi",
    "Tiantan",
    "Wanliu",
    "Wanshouxigong",
]

BASE_33_FEATURE_COLS = [
    "pm2.5",
    "pm10",
    "so2",
    "no2",
    "co",
    "o3",
    "temp",
    "pres",
    "dewp",
    "rain",
    "wspm",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
] + [f"wd_{name}" for name in sorted(WIND_DIRECTION_MAP.keys())]

ANGLE_FEATURE_COLS = ["wind_dir_sin", "wind_dir_cos"]
LEGACY_35_FEATURE_COLS = BASE_33_FEATURE_COLS[:17] + ANGLE_FEATURE_COLS + BASE_33_FEATURE_COLS[17:]


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def compute_mape(pred: np.ndarray, target: np.ndarray, threshold: float = MAPE_THRESHOLD) -> float:
    mask = target > threshold
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((pred[mask] - target[mask]) / target[mask])) * 100.0)


def compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def load_processed_metadata(data_path: Path) -> dict[str, Any]:
    metadata_path = data_path / "metadata.save"
    if metadata_path.exists():
        return joblib.load(metadata_path)
    return {}


def normalize_angle_feature_names(feature_cols: list[str]) -> list[str]:
    # Older metadata may use wd_sin/wd_cos for the same semantic features.
    rename_map = {"wd_sin": "wind_dir_sin", "wd_cos": "wind_dir_cos"}
    return [rename_map.get(col, col) for col in feature_cols]


def get_current_feature_cols(data_path: Path, observed_feature_dim: int | None = None) -> list[str]:
    metadata = load_processed_metadata(data_path)
    feature_cols = metadata.get("feature_cols")

    if isinstance(feature_cols, list) and feature_cols:
        normalized = normalize_angle_feature_names(list(feature_cols))
        if observed_feature_dim is None or len(normalized) == observed_feature_dim:
            return normalized

    if observed_feature_dim is None:
        return BASE_33_FEATURE_COLS

    if observed_feature_dim == len(BASE_33_FEATURE_COLS):
        return list(BASE_33_FEATURE_COLS)
    if observed_feature_dim == len(LEGACY_35_FEATURE_COLS):
        return list(LEGACY_35_FEATURE_COLS)
    if observed_feature_dim == 19:
        return BASE_33_FEATURE_COLS[:17] + ANGLE_FEATURE_COLS

    raise ValueError(
        f"Cannot infer feature column layout for observed tensor dimension={observed_feature_dim}. "
        "Please provide consistent metadata.feature_cols for this dataset."
    )


def infer_checkpoint_feature_cols(config: dict[str, Any], current_feature_cols: list[str]) -> list[str]:
    # Best case: future checkpoints save their exact feature layout.
    for key in ("feature_cols", "used_feature_cols"):
        cols = config.get(key)
        if isinstance(cols, list) and cols:
            return cols

    input_dim = int(config.get("input_dim", len(current_feature_cols)))

    if len(current_feature_cols) == input_dim:
        return list(current_feature_cols)
    if input_dim == 33:
        return list(BASE_33_FEATURE_COLS)
    if input_dim == 35:
        return list(LEGACY_35_FEATURE_COLS)

    raise ValueError(
        f"Cannot infer feature layout for checkpoint input_dim={input_dim}. "
        "Save feature_cols inside future checkpoints to make evaluation unambiguous."
    )


def add_derived_angle_features(X: np.ndarray, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    if all(col in feature_cols for col in ANGLE_FEATURE_COLS):
        return X, feature_cols

    wd_cols = [f"wd_{name}" for name in sorted(WIND_DIRECTION_MAP.keys())]
    missing_wd = [col for col in wd_cols if col not in feature_cols]
    if missing_wd:
        raise ValueError(
            "Cannot derive wind angle features because one-hot wind columns are missing: "
            f"{missing_wd}"
        )

    wd_idx = [feature_cols.index(col) for col in wd_cols]
    wind_one_hot = X[..., wd_idx]

    category_angles = np.array([WIND_DIRECTION_MAP[name] for name in sorted(WIND_DIRECTION_MAP.keys())], dtype=np.float32)
    wind_angle_deg = np.sum(wind_one_hot * category_angles.reshape((1, 1, 1, -1)), axis=-1)
    wind_angle_rad = np.deg2rad(wind_angle_deg)

    wind_dir_sin = np.sin(wind_angle_rad).astype(np.float32)[..., None]
    wind_dir_cos = np.cos(wind_angle_rad).astype(np.float32)[..., None]

    X_aug = np.concatenate([X, wind_dir_sin, wind_dir_cos], axis=-1)
    feature_cols_aug = list(feature_cols) + ANGLE_FEATURE_COLS
    return X_aug, feature_cols_aug


def align_feature_tensor(
    X: np.ndarray,
    current_feature_cols: list[str],
    target_feature_cols: list[str]
) -> np.ndarray:
    working_X = X
    working_cols = list(current_feature_cols)

    missing = [col for col in target_feature_cols if col not in working_cols]
    if missing and any(col in ANGLE_FEATURE_COLS for col in missing):
        working_X, working_cols = add_derived_angle_features(working_X, working_cols)
        missing = [col for col in target_feature_cols if col not in working_cols]

    if missing:
        raise ValueError(
            "Current processed tensor is missing features required by the checkpoint: "
            f"{missing}"
        )

    index_map = {col: idx for idx, col in enumerate(working_cols)}
    aligned_idx = [index_map[col] for col in target_feature_cols]
    return working_X[..., aligned_idx].astype(np.float32)


def refresh_feature_config(config: dict[str, Any], feature_cols: list[str]) -> None:
    config["input_dim"] = len(feature_cols)
    config["feature_cols"] = list(feature_cols)
    config["num_nodes"] = int(config.get("num_nodes", len(STATION_ORDER)))

    if "wspm" in feature_cols:
        config["wind_speed_idx"] = feature_cols.index("wspm")

    wind_one_hot_cols = [idx for idx, col in enumerate(feature_cols) if col.startswith("wd_")]
    if wind_one_hot_cols:
        config["wind_dir_start_idx"] = wind_one_hot_cols[0]
        config["wind_dir_end_idx"] = wind_one_hot_cols[-1] + 1


def prepare_checkpoint_config(checkpoint: dict[str, Any], device: str) -> dict[str, Any]:
    config = dict(TRAIN_CONFIG)
    config.update(checkpoint.get("config", {}))

    state_dict = checkpoint["model_state_dict"]
    config["use_direct_decoding"] = "decoder.step_queries" in state_dict
    config["use_attention"] = any(key.startswith("decoder.attention.") for key in state_dict)
    config["use_learnable_alpha_gate"] = "alpha_logit" in state_dict
    config["use_learnable_sigma"] = "log_sigma" in state_dict
    config["use_node_embeddings"] = any(key.startswith("encoder.node_embed.") for key in state_dict)

    config["device"] = device
    config.setdefault("train_ratio", TRAIN_CONFIG.get("train_ratio", 0.7))
    config.setdefault("val_ratio", TRAIN_CONFIG.get("val_ratio", 0.15))
    config.setdefault("test_ratio", TRAIN_CONFIG.get("test_ratio", 0.15))
    config.setdefault("horizon", TRAIN_CONFIG.get("horizon", 6))
    config.setdefault("batch_size", TRAIN_CONFIG.get("batch_size", 32))
    config.setdefault("output_dim", TRAIN_CONFIG.get("output_dim", 1))
    config.setdefault("num_nodes", TRAIN_CONFIG.get("num_nodes", len(STATION_ORDER)))

    return config


def load_model_for_checkpoint(
    checkpoint_path: Path,
    config: dict[str, Any],
    device: str,
    allow_partial_load: bool = True
) -> tuple[torch.nn.Module, dict[str, list[str]]]:
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    model = create_model(config).to(device)

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        load_info = {"missing_keys": [], "unexpected_keys": [], "partial_load": False}
    except RuntimeError as exc:
        if not allow_partial_load:
            raise RuntimeError(f"Strict checkpoint load failed for {checkpoint_path.name}: {exc}") from exc
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        load_info = {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "partial_load": True,
        }

    model.eval()
    return model, load_info


def load_raw_data(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.load(data_path / "X.npy")
    Y = np.load(data_path / "Y.npy")
    adj = np.load(data_path / "adjacency.npy")
    return X, Y, adj


def evaluate_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    station_order: list[str]
) -> dict[str, Any]:
    overall = {
        "RMSE": compute_rmse(predictions, targets),
        "MAE": compute_mae(predictions, targets),
        "MAPE": compute_mape(predictions, targets),
        "R2": compute_r2(predictions, targets),
        "horizon_mae": [compute_mae(predictions[:, h, :], targets[:, h, :]) for h in range(predictions.shape[1])],
    }

    horizon_metrics = []
    for h in range(predictions.shape[1]):
        horizon_metrics.append(
            {
                "horizon": h + 1,
                "RMSE": compute_rmse(predictions[:, h, :], targets[:, h, :]),
                "MAE": compute_mae(predictions[:, h, :], targets[:, h, :]),
                "MAPE": compute_mape(predictions[:, h, :], targets[:, h, :]),
            }
        )

    station_metrics = []
    for idx, station in enumerate(station_order):
        station_metrics.append(
            {
                "station": station,
                "RMSE": compute_rmse(predictions[:, :, idx], targets[:, :, idx]),
                "MAE": compute_mae(predictions[:, :, idx], targets[:, :, idx]),
                "MAPE": compute_mape(predictions[:, :, idx], targets[:, :, idx]),
            }
        )

    station_metrics_sorted = sorted(station_metrics, key=lambda item: item["MAE"])
    best3 = station_metrics_sorted[:3]
    worst3 = list(reversed(station_metrics_sorted[-3:]))

    return {
        "overall": overall,
        "horizon_metrics": horizon_metrics,
        "station_metrics_best3": best3,
        "station_metrics_worst3": worst3,
    }


def run_model_predictions(
    model: torch.nn.Module,
    X_test: np.ndarray,
    adj: np.ndarray,
    config: dict[str, Any],
    device: str
) -> np.ndarray:
    use_wind_adj = config.get("use_wind_adjacency", False)
    batch_size = int(config.get("batch_size", 32))
    adj_tensor = torch.as_tensor(adj, dtype=torch.float32, device=device)
    all_preds = []

    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            batch_x = torch.as_tensor(X_test[start:start + batch_size], dtype=torch.float32, device=device)

            if use_wind_adj:
                alpha_override = model.get_wind_alpha() if hasattr(model, "get_wind_alpha") else None
                sigma_override = model.get_distance_sigma() if hasattr(model, "get_distance_sigma") else None
                adj_batch = build_dynamic_adjacency(
                    batch_x,
                    config,
                    device,
                    alpha_override=alpha_override,
                    sigma_override=sigma_override,
                )
            else:
                adj_batch = adj_tensor

            pred = model.predict(batch_x, adj_batch, horizon=config["horizon"])
            all_preds.append(pred.detach().cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def inverse_transform_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_scaler: Any | None
) -> tuple[np.ndarray, np.ndarray, bool]:
    if target_scaler is None:
        return predictions, targets, False

    orig_shape = predictions.shape
    predictions_inv = target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(orig_shape)
    targets_inv = target_scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)
    return predictions_inv, targets_inv, True


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_path: Path,
    device: str,
    save_predictions: bool = False,
    allow_partial_load: bool = True
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    config = prepare_checkpoint_config(checkpoint, device=device)

    raw_X, raw_Y, adj = load_raw_data(data_path)
    current_feature_cols = get_current_feature_cols(data_path, observed_feature_dim=int(raw_X.shape[-1]))
    checkpoint_feature_cols = infer_checkpoint_feature_cols(config, current_feature_cols)
    refresh_feature_config(config, checkpoint_feature_cols)

    aligned_X = align_feature_tensor(raw_X, current_feature_cols, checkpoint_feature_cols)

    train_data, val_data, test_data = split_data(aligned_X, raw_Y, config)
    X_train, Y_train = train_data
    X_val, Y_val = val_data
    X_test, Y_test = test_data

    feature_scaler, target_scaler, already_scaled = fit_scalers_on_train(X_train, Y_train, config)

    if already_scaled:
        X_test_scaled = X_test.astype(np.float32)
        Y_test_scaled = Y_test.astype(np.float32)
    else:
        X_test_scaled, Y_test_scaled = scale_data(X_test, Y_test, feature_scaler, target_scaler, config)

    model, load_info = load_model_for_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
        allow_partial_load=allow_partial_load,
    )

    predictions_scaled = run_model_predictions(model, X_test_scaled, adj, config, device=device)
    predictions, targets, is_original_scale = inverse_transform_predictions(
        predictions_scaled,
        Y_test_scaled,
        target_scaler,
    )

    metadata = load_processed_metadata(data_path)
    station_order = metadata.get("station_order", STATION_ORDER)
    metrics = evaluate_predictions(predictions, targets, station_order)

    if save_predictions:
        out_dir = checkpoint_path.parent / "eval_predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = checkpoint_path.stem
        np.save(out_dir / f"{stem}_predictions.npy", predictions)
        np.save(out_dir / f"{stem}_targets.npy", targets)

    learned_alpha = None
    if hasattr(model, "get_wind_alpha"):
        alpha = model.get_wind_alpha()
        if alpha is not None:
            learned_alpha = float(alpha.detach().cpu())

    learned_sigma = None
    if hasattr(model, "get_distance_sigma"):
        sigma = model.get_distance_sigma()
        if sigma is not None:
            learned_sigma = float(sigma.detach().cpu())

    return {
        "checkpoint": str(checkpoint_path).replace("\\", "/"),
        "file_name": checkpoint_path.name,
        "architecture_name": config.get("architecture_name", checkpoint.get("architecture", {}).get("name")),
        "epoch": checkpoint.get("epoch", None),
        "val_loss": checkpoint.get("val_loss", None),
        "val_mae": checkpoint.get("val_mae", None),
        "device_used": device,
        "input_dim": config["input_dim"],
        "used_feature_cols": checkpoint_feature_cols,
        "config": {
            "use_direct_decoding": config.get("use_direct_decoding", False),
            "use_wind_adjacency": config.get("use_wind_adjacency", False),
            "use_learnable_alpha_gate": config.get("use_learnable_alpha_gate", False),
            "use_learnable_sigma": config.get("use_learnable_sigma", False),
            "use_node_embeddings": config.get("use_node_embeddings", True),
            "wind_direction_method": config.get("wind_direction_method"),
            "wind_temporal_graphs": config.get("wind_temporal_graphs", 1),
            "wind_temporal_graph_window": config.get("wind_temporal_graph_window"),
            "loss_type": config.get("loss_type"),
        },
        "load_info": load_info,
        "learned_wind_alpha": learned_alpha,
        "learned_distance_sigma": learned_sigma,
        "is_original_scale": is_original_scale,
        **metrics,
    }


def collect_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    if args.all:
        checkpoint_dir = Path(args.checkpoints_dir)
        paths = sorted(path for path in checkpoint_dir.glob(args.pattern) if path.is_file())
        if not paths:
            raise FileNotFoundError(f"No checkpoints matched {args.pattern!r} in {checkpoint_dir}")
        return paths

    if args.checkpoint:
        path = Path(args.checkpoint)
    else:
        path = DEFAULT_MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return [path]


def print_summary(result: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(result["file_name"])
    print("=" * 72)
    print(f"Architecture: {result.get('architecture_name')}")
    print(f"Input Dim:    {result.get('input_dim')}")
    print(f"Epoch:        {result.get('epoch')}")
    print(f"Val Loss:     {result.get('val_loss')}")
    print(f"Val MAE:      {result.get('val_mae')}")

    if result.get("load_info", {}).get("partial_load"):
        print("Checkpoint Load: partial")
        missing = result["load_info"].get("missing_keys", [])
        unexpected = result["load_info"].get("unexpected_keys", [])
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
    else:
        print("Checkpoint Load: strict")

    overall = result["overall"]
    print("\nOverall Test Metrics:")
    print(f"  RMSE: {overall['RMSE']:.4f}")
    print(f"  MAE:  {overall['MAE']:.4f}")
    print(f"  MAPE: {overall['MAPE']:.2f}%")
    print(f"  R2:   {overall['R2']:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one or more GCN-LSTM checkpoints.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a single checkpoint.")
    parser.add_argument("--all", action="store_true", help="Evaluate all checkpoints in the checkpoints directory.")
    parser.add_argument("--checkpoints-dir", type=str, default=str(CHECKPOINTS_DIR), help="Checkpoint directory.")
    parser.add_argument("--pattern", type=str, default="*.pt", help="Glob pattern used with --all.")
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH), help="Processed data directory.")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or omit for auto.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON path for aggregated results.")
    parser.add_argument("--save-predictions", action="store_true", help="Save per-checkpoint predictions/targets.")
    parser.add_argument(
        "--no-partial-load",
        action="store_true",
        help="Fail instead of falling back to non-strict checkpoint loading.",
    )
    return parser


def main() -> list[dict[str, Any]]:
    parser = build_arg_parser()
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data_path)
    checkpoint_paths = collect_checkpoint_paths(args)

    print("=" * 72)
    print("GCN-LSTM CHECKPOINT EVALUATION")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Data:   {data_path}")
    print(f"Count:  {len(checkpoint_paths)} checkpoint(s)")

    results = []
    failures = []

    for checkpoint_path in checkpoint_paths:
        try:
            result = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                data_path=data_path,
                device=device,
                save_predictions=args.save_predictions,
                allow_partial_load=not args.no_partial_load,
            )
            results.append(result)
            print_summary(result)
        except Exception as exc:
            failure = {"checkpoint": str(checkpoint_path), "error": str(exc)}
            failures.append(failure)
            print("\n" + "!" * 72)
            print(f"FAILED: {checkpoint_path.name}")
            print(exc)
            print("!" * 72)

    if len(results) > 1:
        ranked = sorted(results, key=lambda item: item["overall"]["MAE"])
        print("\n" + "=" * 72)
        print("Ranking By MAE")
        print("=" * 72)
        for idx, item in enumerate(ranked, start=1):
            print(
                f"{idx:>2}. {item['file_name']:<32} "
                f"MAE={item['overall']['MAE']:.4f} "
                f"RMSE={item['overall']['RMSE']:.4f} "
                f"R2={item['overall']['R2']:.4f}"
            )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any]
        if args.all:
            payload = {"results": results, "failures": failures}
        else:
            payload = results[0] if results else {"results": [], "failures": failures}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved JSON to {output_path}")

    if failures:
        print(f"\nCompleted with {len(failures)} failure(s).")
    else:
        print("\nEvaluation complete.")

    return results


if __name__ == "__main__":
    main()
