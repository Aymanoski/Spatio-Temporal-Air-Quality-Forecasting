"""
Flexible evaluation script for checkpointed spatio-temporal models.

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
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

# Add parent directory to path for model/training imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GCNLSTMModel, GraphTransformerModel
from train import (
    CONFIG as TRAIN_CONFIG,
    RevIN,
    build_dynamic_adjacency,
    fit_scalers_on_train,
    inverse_transform_targets,
    scale_data,
    scale_future_met,
    split_data,
)
from utils.window import compute_holiday_feature, create_windows
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
DELTA_FEATURE_COL = "pm25_delta"
HOLIDAY_FEATURE_COL = "holiday_flag"
DELTA_34_FEATURE_COLS = BASE_33_FEATURE_COLS[:17] + [DELTA_FEATURE_COL] + BASE_33_FEATURE_COLS[17:]
HOLIDAY_34_FEATURE_COLS = BASE_33_FEATURE_COLS[:17] + [HOLIDAY_FEATURE_COL] + BASE_33_FEATURE_COLS[17:]


def infer_variant_feature(config: dict[str, Any]) -> str | None:
    if bool(config.get("use_pm25_delta", False)):
        return DELTA_FEATURE_COL
    if bool(config.get("use_holiday_feature", False)):
        return HOLIDAY_FEATURE_COL

    arch_name = str(config.get("architecture_name", "")).lower()
    if "_delta" in arch_name:
        return DELTA_FEATURE_COL
    if "_holiday" in arch_name:
        return HOLIDAY_FEATURE_COL
    return None


def infer_ffn_dim(checkpoint_config: dict[str, Any], state_dict: dict[str, torch.Tensor], hidden_dim: int) -> int | None:
    configured = checkpoint_config.get("ffn_dim")
    if configured is not None:
        return int(configured)

    for key in (
        "encoder.transformer.layers.0.linear1.weight",
        "encoder.transformer.layers.1.linear1.weight",
    ):
        weight = state_dict.get(key)
        if weight is not None and weight.ndim == 2:
            return int(weight.shape[0])

    default_ffn = TRAIN_CONFIG.get("ffn_dim")
    if default_ffn is not None:
        return int(default_ffn)

    # Let model default apply when no explicit width is discoverable.
    return None


def get_window_suffix(config: dict[str, Any]) -> str:
    if bool(config.get("use_pm25_delta", False)):
        return "_delta"
    if bool(config.get("use_holiday_feature", False)):
        return "_holiday"
    return ""


def build_variant_windows(
    data_path: Path,
    input_len: int,
    horizon: int,
    use_pm25_delta: bool,
    use_holiday_feature: bool,
) -> tuple[np.ndarray, np.ndarray]:
    data_tensor = np.load(data_path / "data_tensor.npy")

    if use_holiday_feature:
        metadata = load_processed_metadata(data_path)
        timestamps = metadata.get("timestamps")
        if timestamps is None:
            raise ValueError("metadata.save is missing timestamps required to rebuild holiday windows.")
        if len(timestamps) != len(data_tensor):
            raise ValueError(
                "Timestamp length mismatch while rebuilding holiday windows: "
                f"{len(timestamps)} != {len(data_tensor)}"
            )

        holiday = compute_holiday_feature(timestamps)
        holiday_3d = np.tile(holiday[:, np.newaxis, np.newaxis], (1, data_tensor.shape[1], 1)).astype(np.float32)
        data_tensor = np.concatenate([data_tensor[:, :, :17], holiday_3d, data_tensor[:, :, 17:]], axis=2)

    X, Y = create_windows(
        data_tensor,
        input_len=input_len,
        horizon=horizon,
        future_met_indices=None,
        add_pm25_delta=use_pm25_delta,
    )
    return X.astype(np.float32), Y.astype(np.float32)


def infer_model_type(checkpoint_config: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> str:
    configured = checkpoint_config.get("model_type")
    if configured in {"gcn_lstm", "graph_transformer"}:
        return str(configured)

    has_transformer_head = any(key.startswith("head.") for key in state_dict)
    has_transformer_encoder = any(key.startswith("encoder.transformer.layers.") for key in state_dict)
    if has_transformer_head or has_transformer_encoder:
        return "graph_transformer"

    return "gcn_lstm"


def infer_graph_conv(checkpoint_config: dict[str, Any], state_dict: dict[str, torch.Tensor], model_type: str) -> str:
    configured = checkpoint_config.get("graph_conv")
    if configured in {"gcn", "gat"}:
        return str(configured)

    if model_type == "graph_transformer":
        if any(key.startswith("encoder.gat_layers.") for key in state_dict):
            return "gat"
        if "encoder.gcn_weight" in state_dict:
            return "gcn"

    has_gat_signatures = any(
        (
            key.startswith("encoder.layers.")
            or key.startswith("decoder.layers.")
            or key.startswith("encoder.gat_layers.")
        )
        and (".attn_vec" in key or ".W." in key or ".W_src." in key or ".W_dst." in key)
        for key in state_dict
    )
    return "gat" if has_gat_signatures else "gcn"


def infer_gat_version(checkpoint_config: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> str:
    configured = checkpoint_config.get("gat_version")
    if configured in {"v1", "v2"}:
        return str(configured)

    has_v2_signatures = any(".W_src." in key or ".W_dst." in key for key in state_dict)
    return "v2" if has_v2_signatures else "v1"


def infer_layer_count(state_dict: dict[str, torch.Tensor], pattern: str) -> int | None:
    regex = re.compile(pattern)
    indices: set[int] = set()

    for key in state_dict:
        match = regex.match(key)
        if match:
            indices.add(int(match.group(1)))

    if not indices:
        return None
    return max(indices) + 1


def infer_num_tf_layers(checkpoint_config: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> int:
    configured = checkpoint_config.get("num_tf_layers")
    if configured is not None:
        return int(configured)

    inferred = infer_layer_count(state_dict, r"^encoder\.transformer\.layers\.(\d+)\.")
    if inferred is not None:
        return inferred

    return int(TRAIN_CONFIG.get("num_tf_layers", 2))


def infer_num_gat_layers(checkpoint_config: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> int:
    configured = checkpoint_config.get("num_gat_layers")
    if configured is not None:
        return int(configured)

    inferred = infer_layer_count(state_dict, r"^encoder\.gat_layers\.(\d+)\.")
    if inferred is not None:
        return inferred

    return int(TRAIN_CONFIG.get("num_gat_layers", 1))


def build_model_from_config(config: dict[str, Any]) -> torch.nn.Module:
    model_type = str(config.get("model_type", "gcn_lstm"))

    if model_type == "graph_transformer":
        return GraphTransformerModel(
            input_dim=int(config.get("input_dim", 33)),
            hidden_dim=int(config.get("hidden_dim", 64)),
            output_dim=int(config.get("output_dim", 1)),
            num_nodes=int(config.get("num_nodes", 12)),
            num_tf_layers=int(config.get("num_tf_layers", 2)),
            num_heads=int(config.get("num_heads", 4)),
            ffn_dim=config.get("ffn_dim", None),
            dropout=float(config.get("dropout", 0.1)),
            horizon=int(config.get("horizon", 6)),
            use_node_embeddings=bool(config.get("use_node_embeddings", True)),
            use_learnable_alpha_gate=bool(config.get("use_learnable_alpha_gate", False)),
            initial_wind_alpha=float(config.get("wind_alpha", 0.6)),
            graph_conv=str(config.get("graph_conv", "gcn")),
            num_gat_layers=int(config.get("num_gat_layers", 1)),
            gat_version=str(config.get("gat_version", "v1")),
            use_post_temporal_gat=bool(config.get("use_post_temporal_gat", False)),
            use_temporal_attention_head=bool(config.get("use_temporal_attention_head", False)),
            use_t24_residual=bool(config.get("use_t24_residual", False)),
            initial_t24_alpha=float(config.get("initial_t24_alpha", 0.3)),
            future_met_dim=int(config.get("future_met_dim", 0)),
            use_multiscale_temporal=bool(config.get("use_multiscale_temporal", False)),
            local_window=int(config.get("local_window", 6)),
            n_local_layers=int(config.get("n_local_layers", 1)),
            use_horizon_residual_weights=bool(config.get("use_horizon_residual_weights", False)),
            use_learnable_static_adj=bool(config.get("use_learnable_static_adj", False)),
            initial_distance_sigma=float(config.get("distance_sigma", 1800.0)),
            use_multitask=bool(config.get("use_multitask", False)),
            n_aux_targets=int(config.get("n_aux_targets", 5)),
            use_station_horizon_bias=bool(config.get("use_station_horizon_bias", False)),
            use_regime_conditioning=bool(config.get("use_regime_conditioning", False)),
        )

    if model_type == "gcn_lstm":
        return GCNLSTMModel(
            input_dim=int(config.get("input_dim", 33)),
            hidden_dim=int(config.get("hidden_dim", 64)),
            output_dim=int(config.get("output_dim", 1)),
            num_nodes=int(config.get("num_nodes", 12)),
            num_layers=int(config.get("num_layers", 2)),
            num_heads=int(config.get("num_heads", 4)),
            dropout=float(config.get("dropout", 0.1)),
            horizon=int(config.get("horizon", 6)),
            use_direct_decoding=bool(config.get("use_direct_decoding", False)),
            use_learnable_alpha_gate=bool(config.get("use_learnable_alpha_gate", False)),
            initial_wind_alpha=float(config.get("wind_alpha", 0.6)),
            use_node_embeddings=bool(config.get("use_node_embeddings", True)),
            use_attention=bool(config.get("use_attention", True)),
            graph_conv=str(config.get("graph_conv", "gcn")),
        )

    raise ValueError(f"Unsupported model_type in checkpoint config: {model_type}")


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


def get_current_feature_cols(
    data_path: Path,
    observed_feature_dim: int | None = None,
    variant_feature: str | None = None,
) -> list[str]:
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
    if observed_feature_dim == len(DELTA_34_FEATURE_COLS):
        if variant_feature == DELTA_FEATURE_COL:
            return list(DELTA_34_FEATURE_COLS)
        if variant_feature == HOLIDAY_FEATURE_COL:
            return list(HOLIDAY_34_FEATURE_COLS)
        # Fallback: preserve input rank even when metadata is unavailable.
        return list(DELTA_34_FEATURE_COLS)
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
    if input_dim == 34:
        variant_feature = infer_variant_feature(config)
        if variant_feature == DELTA_FEATURE_COL:
            return list(DELTA_34_FEATURE_COLS)
        if variant_feature == HOLIDAY_FEATURE_COL:
            return list(HOLIDAY_34_FEATURE_COLS)
        raise ValueError(
            "Cannot disambiguate 34-feature layout for checkpoint. "
            "Expected use_pm25_delta or use_holiday_feature in config."
        )
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
    checkpoint_config = dict(checkpoint.get("config", {}))
    config = dict(TRAIN_CONFIG)
    config.update(checkpoint_config)

    state_dict = checkpoint["model_state_dict"]
    config["model_type"] = infer_model_type(checkpoint_config, state_dict)
    config["graph_conv"] = infer_graph_conv(checkpoint_config, state_dict, config["model_type"])
    config["gat_version"] = infer_gat_version(checkpoint_config, state_dict)

    if config["model_type"] == "graph_transformer":
        config["num_tf_layers"] = infer_num_tf_layers(checkpoint_config, state_dict)
        config["ffn_dim"] = infer_ffn_dim(
            checkpoint_config,
            state_dict,
            hidden_dim=int(config.get("hidden_dim", 64)),
        )
        if config["graph_conv"] == "gat":
            config["num_gat_layers"] = infer_num_gat_layers(checkpoint_config, state_dict)

    config["use_direct_decoding"] = (
        config["model_type"] == "graph_transformer" or "decoder.step_queries" in state_dict
    )
    config["use_attention"] = any(key.startswith("decoder.attention.") for key in state_dict)
    config["use_learnable_alpha_gate"] = "alpha_logit" in state_dict
    config["use_learnable_sigma"] = "log_sigma" in state_dict
    config["use_node_embeddings"] = any(key.startswith("encoder.node_embed.") for key in state_dict)

    # Transformer variant flags that affect module topology.
    # Prefer explicit checkpoint config when present, otherwise infer from state dict keys.
    config["use_temporal_attention_head"] = bool(
        checkpoint_config.get(
            "use_temporal_attention_head",
            any(key.startswith("head.horizon_scorers") for key in state_dict),
        )
    )
    config["use_post_temporal_gat"] = bool(
        checkpoint_config.get(
            "use_post_temporal_gat",
            any(key.startswith("post_gat.") or key.startswith("post_gat_norm.") for key in state_dict),
        )
    )
    config["use_t24_residual"] = bool(
        checkpoint_config.get(
            "use_t24_residual",
            "t24_logit" in state_dict,
        )
    )
    if "initial_t24_alpha" in checkpoint_config:
        config["initial_t24_alpha"] = float(checkpoint_config["initial_t24_alpha"])

    # Additional topology flags that must never inherit current TRAIN_CONFIG defaults.
    config["use_horizon_residual_weights"] = bool(
        checkpoint_config.get(
            "use_horizon_residual_weights",
            "horizon_residual_logits" in state_dict,
        )
    )
    config["use_learnable_static_adj"] = bool(
        checkpoint_config.get(
            "use_learnable_static_adj",
            "static_adj_logits" in state_dict,
        )
    )
    config["use_multitask"] = bool(
        checkpoint_config.get(
            "use_multitask",
            any(key.startswith("aux_head.") for key in state_dict),
        )
    )
    config["use_station_horizon_bias"] = bool(
        checkpoint_config.get(
            "use_station_horizon_bias",
            "station_horizon_bias" in state_dict,
        )
    )
    config["use_regime_conditioning"] = bool(
        checkpoint_config.get(
            "use_regime_conditioning",
            any(key.startswith("regime_proj.") for key in state_dict),
        )
    )

    # Detect oracle future meteorology from state dict
    config["use_future_met"] = "head.future_met_proj.weight" in state_dict
    if config["use_future_met"]:
        config["future_met_dim"] = int(state_dict["head.future_met_proj.weight"].shape[1])
    else:
        config["future_met_dim"] = 0

    # These flags affect only adjacency construction (no model parameter signature).
    # Explicitly read from checkpoint config with safe False defaults so that old
    # checkpoints are never evaluated with the wrong adjacency, regardless of what
    # TRAIN_CONFIG currently contains.
    config["use_per_timestep_adj"] = bool(checkpoint_config.get("use_per_timestep_adj", False))
    config["use_physics_guided_adj"] = bool(checkpoint_config.get("use_physics_guided_adj", False))
    config["use_transport_time_weight"] = bool(checkpoint_config.get("use_transport_time_weight", False))
    config["use_pm25_delta"] = bool(checkpoint_config.get("use_pm25_delta", False))
    config["use_holiday_feature"] = bool(checkpoint_config.get("use_holiday_feature", False))
    config["use_per_station_norm"] = bool(checkpoint_config.get("use_per_station_norm", False))
    config["use_revin"] = bool(checkpoint_config.get("use_revin", False))
    config["revin_feature_indices"] = list(checkpoint_config.get("revin_feature_indices", [0]))
    config["residual_window"] = int(checkpoint_config.get("residual_window", 1))
    config["use_trend_residual"] = bool(checkpoint_config.get("use_trend_residual", False))
    config["use_multiscale_temporal"] = bool(checkpoint_config.get("use_multiscale_temporal", False))
    config["local_window"] = int(checkpoint_config.get("local_window", config.get("local_window", 6)))
    config["n_local_layers"] = int(checkpoint_config.get("n_local_layers", config.get("n_local_layers", 1)))
    config["met_forecast_mode"] = str(checkpoint_config.get("met_forecast_mode", config.get("met_forecast_mode", "oracle")))

    arch_name = str(
        checkpoint_config.get("architecture_name")
        or checkpoint.get("architecture", {}).get("name", "")
    ).lower()

    if "use_persistence_residual" in checkpoint_config:
        config["use_persistence_residual"] = bool(checkpoint_config["use_persistence_residual"])
    else:
        config["use_persistence_residual"] = "residual" in arch_name

    if "use_log_transform" in checkpoint_config:
        config["use_log_transform"] = bool(checkpoint_config["use_log_transform"])
    else:
        config["use_log_transform"] = "log1p" in arch_name

    if "target_feature_idx" in checkpoint_config:
        config["target_feature_idx"] = int(checkpoint_config["target_feature_idx"])
    else:
        config["target_feature_idx"] = 0

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
    model = build_model_from_config(config).to(device)

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


def load_raw_data(data_path: Path, input_len: int = 24) -> tuple:
    X = np.load(data_path / f"X_{input_len}.npy")
    Y = np.load(data_path / f"Y_{input_len}.npy")
    adj = np.load(data_path / "adjacency.npy")
    z_path = data_path / f"Z_{input_len}.npy"
    Z = np.load(z_path) if z_path.exists() else None
    return X, Y, adj, Z


def load_raw_data_for_config(data_path: Path, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    input_len = int(config.get("input_len", 24))
    horizon = int(config.get("horizon", 6))
    suffix = get_window_suffix(config)

    x_path = data_path / f"X_{input_len}{suffix}.npy"
    y_path = data_path / f"Y_{input_len}{suffix}.npy"

    if x_path.exists() and y_path.exists():
        X = np.load(x_path)
        Y = np.load(y_path)
    elif suffix:
        X, Y = build_variant_windows(
            data_path=data_path,
            input_len=input_len,
            horizon=horizon,
            use_pm25_delta=bool(config.get("use_pm25_delta", False)),
            use_holiday_feature=bool(config.get("use_holiday_feature", False)),
        )
    else:
        X = np.load(data_path / f"X_{input_len}.npy")
        Y = np.load(data_path / f"Y_{input_len}.npy")

    adj = np.load(data_path / "adjacency.npy")
    z_path = data_path / f"Z_{input_len}.npy"
    Z = np.load(z_path) if z_path.exists() else None
    return X, Y, adj, Z


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
    device: str,
    Z_test: np.ndarray | None = None,
) -> np.ndarray:
    use_wind_adj = config.get("use_wind_adjacency", False)
    use_residual = config.get("use_persistence_residual", False)
    use_t24 = config.get("use_t24_residual", False)
    use_revin = config.get("use_revin", False)
    use_trend_residual = config.get("use_trend_residual", False)
    use_horizon_residual_weights = config.get("use_horizon_residual_weights", False)
    use_per_station_norm = config.get("use_per_station_norm", False)
    residual_window = int(config.get("residual_window", 1))
    target_feature_idx = int(config.get("target_feature_idx", 0))
    met_cols = list(config.get("future_met_indices", []))
    met_forecast_mode = str(config.get("met_forecast_mode", "oracle")).lower()
    batch_size = int(config.get("batch_size", 32))
    adj_tensor = torch.as_tensor(adj, dtype=torch.float32, device=device)
    all_preds = []
    revin = RevIN(config.get("revin_feature_indices", [0])) if use_revin else None

    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            batch_x = torch.as_tensor(X_test[start:start + batch_size], dtype=torch.float32, device=device)

            if use_wind_adj:
                alpha_override = model.get_wind_alpha() if hasattr(model, "get_wind_alpha") else None
                sigma_override = model.get_distance_sigma() if hasattr(model, "get_distance_sigma") else None
                static_adj_override = model.get_static_adj() if hasattr(model, "get_static_adj") else None
                adj_batch = build_dynamic_adjacency(
                    batch_x,
                    config,
                    device,
                    alpha_override=alpha_override,
                    sigma_override=sigma_override,
                    static_adj_override=static_adj_override,
                )
            else:
                adj_batch = adj_tensor

            # Future meteorology batch (oracle, only when checkpoint was trained with it)
            z_batch = None
            if Z_test is not None:
                z_batch = torch.as_tensor(Z_test[start:start + batch_size], dtype=torch.float32, device=device)
            elif met_forecast_mode == "persistence" and met_cols:
                last_met = batch_x[:, -1:, :, met_cols]
                z_batch = last_met.expand(-1, int(config["horizon"]), -1, -1).contiguous()

            x_model = revin.normalize(batch_x) if revin is not None else batch_x
            pred = model.predict(x_model, adj_batch, horizon=config["horizon"], future_met=z_batch)

            if use_residual:
                if residual_window > 1:
                    y_last = x_model[:, -residual_window:, :, target_feature_idx].mean(dim=1)
                else:
                    y_last = x_model[:, -1, :, target_feature_idx]

                if use_per_station_norm:
                    y_last = (
                        y_last * config["_feat_pm25_scale"]
                        + config["_feat_pm25_mean"]
                        - config["_target_means"]
                    ) / config["_target_scales"]

                if use_trend_residual and x_model.shape[1] >= 6:
                    slope = (x_model[:, -1, :, target_feature_idx] - x_model[:, -6, :, target_feature_idx]) / 5.0
                    steps = torch.arange(1, pred.shape[1] + 1, device=x_model.device, dtype=x_model.dtype).view(1, -1, 1)
                    prior = y_last.unsqueeze(1) + slope.unsqueeze(1) * steps
                else:
                    prior = y_last.unsqueeze(1).expand_as(pred)

                if use_horizon_residual_weights and hasattr(model, "get_horizon_residual_weights"):
                    hw = model.get_horizon_residual_weights()
                    if hw is not None:
                        pred = pred + hw.to(device=pred.device, dtype=pred.dtype).view(1, -1, 1) * prior
                    else:
                        pred = pred + prior
                else:
                    pred = pred + prior

            if revin is not None:
                pred = revin.denormalize(pred)

            if use_t24 and hasattr(model, "get_t24_alpha"):
                t24_alpha = model.get_t24_alpha()
                if t24_alpha is not None:
                    y_t24 = batch_x[:, 0, :, target_feature_idx]
                    pred = pred + t24_alpha.to(device=pred.device, dtype=pred.dtype) * y_t24.unsqueeze(1).expand_as(pred)

            all_preds.append(pred.detach().cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def inverse_transform_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_scaler: Any | None,
    use_log_transform: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if target_scaler is None:
        return predictions, targets, False

    predictions_inv = inverse_transform_targets(predictions, target_scaler)
    targets_inv = inverse_transform_targets(targets, target_scaler)

    # Keep evaluator semantics aligned with train.py compute_metrics/validate.
    if use_log_transform:
        predictions_inv = np.expm1(predictions_inv)
        targets_inv = np.expm1(targets_inv)
        predictions_inv = np.clip(predictions_inv, 0.0, None)

    return predictions_inv, targets_inv, True


def set_per_station_residual_tensors(
    config: dict[str, Any],
    feature_scaler: Any | None,
    target_scaler: Any | None,
    device: str,
) -> None:
    if not bool(config.get("use_per_station_norm", False)):
        return
    if feature_scaler is None or not isinstance(target_scaler, list):
        return

    config["_feat_pm25_mean"] = torch.tensor(float(feature_scaler.mean_[0]), dtype=torch.float32, device=device)
    config["_feat_pm25_scale"] = torch.tensor(float(feature_scaler.scale_[0]), dtype=torch.float32, device=device)
    config["_target_means"] = torch.tensor([sc.mean_[0] for sc in target_scaler], dtype=torch.float32, device=device)
    config["_target_scales"] = torch.tensor([sc.scale_[0] for sc in target_scaler], dtype=torch.float32, device=device)


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_path: Path,
    device: str,
    save_predictions: bool = False,
    allow_partial_load: bool = True
) -> dict[str, Any]:
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
    X_val, Y_val = val_data
    X_test, Y_test = test_data

    feature_scaler, target_scaler, already_scaled = fit_scalers_on_train(X_train, Y_train, config)
    set_per_station_residual_tensors(config, feature_scaler, target_scaler, device)

    if already_scaled:
        X_test_scaled = X_test.astype(np.float32)
        Y_test_scaled = Y_test.astype(np.float32)
        Z_test_scaled = None
    else:
        X_test_scaled, Y_test_scaled = scale_data(X_test, Y_test, feature_scaler, target_scaler, config)

        # Scale future met test split if this checkpoint used oracle future meteorology
        Z_test_scaled = None
        if config.get("use_future_met") and raw_Z is not None:
            n = len(aligned_X)
            val_end = int(n * (config["train_ratio"] + config["val_ratio"]))
            Z_test_raw = raw_Z[val_end:]
            Z_test_scaled = scale_future_met(Z_test_raw, feature_scaler, config)

    model, load_info = load_model_for_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
        allow_partial_load=allow_partial_load,
    )

    predictions_scaled = run_model_predictions(
        model, X_test_scaled, adj, config, device=device, Z_test=Z_test_scaled
    )
    predictions, targets, is_original_scale = inverse_transform_predictions(
        predictions_scaled,
        Y_test_scaled,
        target_scaler,
        use_log_transform=bool(config.get("use_log_transform", False)),
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
            "model_type": config.get("model_type"),
            "graph_conv": config.get("graph_conv"),
            "gat_version": config.get("gat_version"),
            "num_tf_layers": config.get("num_tf_layers"),
            "ffn_dim": config.get("ffn_dim"),
            "num_gat_layers": config.get("num_gat_layers"),
            "use_direct_decoding": config.get("use_direct_decoding", False),
            "use_wind_adjacency": config.get("use_wind_adjacency", False),
            "use_learnable_alpha_gate": config.get("use_learnable_alpha_gate", False),
            "use_learnable_sigma": config.get("use_learnable_sigma", False),
            "use_node_embeddings": config.get("use_node_embeddings", True),
            "use_temporal_attention_head": config.get("use_temporal_attention_head", False),
            "use_post_temporal_gat": config.get("use_post_temporal_gat", False),
            "use_t24_residual": config.get("use_t24_residual", False),
            "use_horizon_residual_weights": config.get("use_horizon_residual_weights", False),
            "use_learnable_static_adj": config.get("use_learnable_static_adj", False),
            "use_multitask": config.get("use_multitask", False),
            "use_station_horizon_bias": config.get("use_station_horizon_bias", False),
            "use_regime_conditioning": config.get("use_regime_conditioning", False),
            "use_future_met": config.get("use_future_met", False),
            "met_forecast_mode": config.get("met_forecast_mode"),
            "wind_direction_method": config.get("wind_direction_method"),
            "wind_temporal_graphs": config.get("wind_temporal_graphs", 1),
            "wind_temporal_graph_window": config.get("wind_temporal_graph_window"),
            "use_pm25_delta": config.get("use_pm25_delta", False),
            "use_holiday_feature": config.get("use_holiday_feature", False),
            "use_persistence_residual": config.get("use_persistence_residual", False),
            "use_per_station_norm": config.get("use_per_station_norm", False),
            "residual_window": config.get("residual_window", 1),
            "use_trend_residual": config.get("use_trend_residual", False),
            "use_revin": config.get("use_revin", False),
            "target_feature_idx": config.get("target_feature_idx", 0),
            "use_log_transform": config.get("use_log_transform", False),
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
    print("CHECKPOINT EVALUATION")
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
