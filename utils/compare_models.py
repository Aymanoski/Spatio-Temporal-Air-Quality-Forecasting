"""
Utility for comparing models trained on different hardware and architectures.
"""

import torch
import numpy as np
import os
from pathlib import Path


def print_table(headers, rows):
    """Simple table printer without external dependencies."""
    # Calculate column widths
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Print header
    header_str = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * len(header_str))

    # Print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))


def load_checkpoint_info(checkpoint_path):
    """Load checkpoint and extract metadata."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    info = {
        'path': checkpoint_path,
        'epoch': checkpoint.get('epoch', 'N/A'),
        'val_loss': checkpoint.get('val_loss', 'N/A'),
        'train_loss': checkpoint.get('train_loss', 'N/A'),
        'timestamp': checkpoint.get('timestamp', 'N/A'),
    }

    # Hardware info
    if 'hardware' in checkpoint:
        hw = checkpoint['hardware']
        info['hardware_tag'] = hw.get('tag', 'unknown')
        info['device'] = hw.get('device', 'N/A')
        info['gpu_name'] = hw.get('cuda_device_name', 'CPU')
    else:
        info['hardware_tag'] = 'unknown'
        info['device'] = 'N/A'
        info['gpu_name'] = 'N/A'

    # Architecture info
    if 'architecture' in checkpoint:
        arch = checkpoint['architecture']
        info['architecture_name'] = arch.get('name', 'unknown')
        info['num_params'] = arch.get('num_params', 'N/A')
        info['use_direct_decoding'] = arch.get('use_direct_decoding', False)
        info['use_wind_adjacency'] = arch.get('use_wind_adjacency', False)
    else:
        info['architecture_name'] = 'unknown'
        info['num_params'] = 'N/A'
        info['use_direct_decoding'] = 'N/A'
        info['use_wind_adjacency'] = 'N/A'

    # Config
    if 'config' in checkpoint:
        config = checkpoint['config']
        info['hidden_dim'] = config.get('hidden_dim', 'N/A')
        info['num_layers'] = config.get('num_layers', 'N/A')
        info['learning_rate'] = config.get('learning_rate', 'N/A')
        info['batch_size'] = config.get('batch_size', 'N/A')
        info['loss_type'] = config.get('loss_type', 'N/A')

    return info, checkpoint


def compare_checkpoints(*checkpoint_paths):
    """
    Compare multiple model checkpoints.

    Args:
        *checkpoint_paths: Variable number of paths to checkpoint files

    Returns:
        Comparison table as string
    """
    if len(checkpoint_paths) < 2:
        raise ValueError("Need at least 2 checkpoints to compare")

    print("=" * 80)
    print("MODEL COMPARISON REPORT")
    print("=" * 80)

    # Load all checkpoints
    checkpoints_info = []
    for path in checkpoint_paths:
        try:
            info, _ = load_checkpoint_info(path)
            checkpoints_info.append(info)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            continue

    if len(checkpoints_info) < 2:
        print("Error: Could not load enough checkpoints for comparison")
        return

    # Print general comparison
    print("\n1. GENERAL INFORMATION")
    print("-" * 80)
    headers = ["Metric"] + [f"Model {i+1}" for i in range(len(checkpoints_info))]

    rows = [
        ["Architecture"] + [info['architecture_name'] for info in checkpoints_info],
        ["Hardware"] + [info['hardware_tag'] for info in checkpoints_info],
        ["GPU/CPU"] + [info['gpu_name'] for info in checkpoints_info],
        ["Timestamp"] + [info['timestamp'] for info in checkpoints_info],
        ["Final Epoch"] + [info['epoch'] for info in checkpoints_info],
    ]

    print_table(headers, rows)

    # Print architecture details
    print("\n2. ARCHITECTURE DETAILS")
    print("-" * 80)
    rows = [
        ["Parameters"] + [f"{info['num_params']:,}" if isinstance(info['num_params'], int) else info['num_params'] for info in checkpoints_info],
        ["Hidden Dim"] + [str(info.get('hidden_dim', 'N/A')) for info in checkpoints_info],
        ["Num Layers"] + [str(info.get('num_layers', 'N/A')) for info in checkpoints_info],
        ["Direct Decoding"] + [str(info['use_direct_decoding']) for info in checkpoints_info],
        ["Wind Adjacency"] + [str(info['use_wind_adjacency']) for info in checkpoints_info],
    ]

    print_table(headers, rows)

    # Print training configuration
    print("\n3. TRAINING CONFIGURATION")
    print("-" * 80)
    rows = [
        ["Learning Rate"] + [str(info.get('learning_rate', 'N/A')) for info in checkpoints_info],
        ["Batch Size"] + [str(info.get('batch_size', 'N/A')) for info in checkpoints_info],
        ["Loss Type"] + [str(info.get('loss_type', 'N/A')) for info in checkpoints_info],
    ]

    print_table(headers, rows)

    # Print performance metrics
    print("\n4. VALIDATION PERFORMANCE")
    print("-" * 80)

    val_losses = [info['val_loss'] for info in checkpoints_info]
    train_losses = [info['train_loss'] for info in checkpoints_info]

    rows = [
        ["Validation Loss"] + [f"{loss:.6f}" if isinstance(loss, (int, float)) else str(loss) for loss in val_losses],
        ["Training Loss"] + [f"{loss:.6f}" if isinstance(loss, (int, float)) else str(loss) for loss in train_losses],
    ]

    print_table(headers, rows)

    # Performance comparison analysis
    print("\n5. ANALYSIS & RECOMMENDATIONS")
    print("-" * 80)

    # Find best validation loss
    valid_losses = [(i, loss) for i, loss in enumerate(val_losses) if isinstance(loss, (int, float))]
    if valid_losses:
        best_idx, best_loss = min(valid_losses, key=lambda x: x[1])
        print(f"\n✓ Best Validation Loss: Model {best_idx + 1} ({best_loss:.6f})")
        print(f"  Architecture: {checkpoints_info[best_idx]['architecture_name']}")
        print(f"  Hardware: {checkpoints_info[best_idx]['hardware_tag']}")

    # Hardware considerations
    print("\n⚠ HARDWARE COMPARISON NOTES:")
    print("  - Different hardware affects:")
    print("    • Training speed (iterations/sec)")
    print("    • Batch processing efficiency")
    print("    • Numerical precision (fp16 vs fp32)")
    print("  - Performance metrics (loss, accuracy) should be directly comparable")
    print("  - Lower validation loss = better model, regardless of hardware")
    print("  - Training time differences are expected and don't affect model quality")

    # Architecture differences
    arch_names = set(info['architecture_name'] for info in checkpoints_info)
    if len(arch_names) > 1:
        print("\n⚠ ARCHITECTURE DIFFERENCES DETECTED:")
        print("  - Different architectures may have different capacities")
        print("  - Compare on validation/test metrics, not just loss")
        print("  - Consider model complexity vs performance trade-off")

    print("\n" + "=" * 80)

    return checkpoints_info


def print_checkpoint_summary(checkpoint_path):
    """Print summary of a single checkpoint."""
    info, checkpoint = load_checkpoint_info(checkpoint_path)

    print("\n" + "=" * 60)
    print(f"CHECKPOINT SUMMARY: {Path(checkpoint_path).name}")
    print("=" * 60)

    print("\nArchitecture:")
    print(f"  Name: {info['architecture_name']}")
    print(f"  Parameters: {info['num_params']:,}" if isinstance(info['num_params'], int) else f"  Parameters: {info['num_params']}")
    print(f"  Direct Decoding: {info['use_direct_decoding']}")
    print(f"  Wind Adjacency: {info['use_wind_adjacency']}")

    print("\nHardware:")
    print(f"  Tag: {info['hardware_tag']}")
    print(f"  Device: {info['device']}")
    print(f"  GPU: {info['gpu_name']}")

    print("\nTraining:")
    print(f"  Epoch: {info['epoch']}")
    print(f"  Validation Loss: {info['val_loss']:.6f}" if isinstance(info['val_loss'], (int, float)) else f"  Validation Loss: {info['val_loss']}")
    print(f"  Training Loss: {info['train_loss']:.6f}" if isinstance(info['train_loss'], (int, float)) else f"  Training Loss: {info['train_loss']}")
    print(f"  Timestamp: {info['timestamp']}")

    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Compare multiple models:")
        print("    python compare_models.py <checkpoint1> <checkpoint2> [checkpoint3...]")
        print("\n  Show single model info:")
        print("    python compare_models.py <checkpoint>")
        sys.exit(1)

    checkpoint_paths = sys.argv[1:]

    if len(checkpoint_paths) == 1:
        # Single checkpoint summary
        print_checkpoint_summary(checkpoint_paths[0])
    else:
        # Compare multiple checkpoints
        compare_checkpoints(*checkpoint_paths)
