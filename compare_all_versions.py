"""
Compare all trained versions and generate comprehensive ablation study report.
"""

import glob
import os
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.evaluate_comparison import compare_models


VERSION_PATTERNS = [
    "v1_baseline_*_best.pt",
    "v2_direct_decoding_*_best.pt",
    "v3_evt_loss_*_best.pt",
    "v4_wind_adjacency_*_best.pt",
    "v5_full_optimized_*_best.pt",
]


def find_all_checkpoints(checkpoint_dir="models/checkpoints/"):
    """
    Find all checkpoint files automatically.

    Returns:
        Dictionary mapping checkpoint names to checkpoint paths
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return {}

    found_checkpoints = {}

    # Keep ablation versions first so the incremental story remains easy to read.
    for pattern in VERSION_PATTERNS:
        matches = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))
        if matches:
            version_key = pattern.split("_*")[0]
            found_checkpoints[version_key] = matches[0]

    # Add any remaining checkpoints in deterministic filename order.
    known_paths = set(found_checkpoints.values())
    for path in sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt"))):
        if path not in known_paths:
            found_checkpoints[Path(path).stem] = path

    return found_checkpoints


def print_ablation_analysis(checkpoints):
    """
    Print ablation study analysis showing incremental improvements.
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY ANALYSIS")
    print("=" * 80)

    print("\nIncremental Feature Contributions:")
    print("-" * 80)

    version_order = [
        "v1_baseline",
        "v2_direct_decoding",
        "v3_evt_loss",
        "v4_wind_adjacency",
        "v5_full_optimized",
    ]

    feature_map = {
        "v1_baseline": "Baseline (Autoregressive + MSE + Static Adj)",
        "v2_direct_decoding": "+ Direct Multi-Horizon Decoding",
        "v3_evt_loss": "+ EVT Hybrid Loss (Extreme Value)",
        "v4_wind_adjacency": "+ Wind-Aware Dynamic Adjacency",
        "v5_full_optimized": "+ Adaptive EVT Schedule + Asymmetric Penalty",
    }

    for i, version in enumerate(version_order, 1):
        print(f"\n{i}. {feature_map[version]}")
        if version in checkpoints:
            print(f"   Checkpoint: {Path(checkpoints[version]).name}")
        else:
            print("   Status: NOT TRAINED YET")

    extra_checkpoints = [
        Path(path).name for key, path in checkpoints.items() if key not in feature_map
    ]
    if extra_checkpoints:
        print("\nAdditional Checkpoints Included:")
        print("-" * 80)
        for checkpoint_name in extra_checkpoints:
            print(f"  - {checkpoint_name}")

    print("\n" + "=" * 80)


def main():
    """Main comparison function."""
    print("=" * 80)
    print("VERSION COMPARISON TOOL")
    print("=" * 80)

    print("\n[1/2] Searching for checkpoints...")
    checkpoints = find_all_checkpoints()

    if not checkpoints:
        print("\nNo checkpoint files found!")
        print("Please train versions first using:")
        print("  python train_versions.py --all")
        return

    print(f"\nFound {len(checkpoints)} checkpoint files:")
    for checkpoint_name, path in checkpoints.items():
        print(f"  - {checkpoint_name}: {Path(path).name}")

    print_ablation_analysis(checkpoints)

    print("\n[2/2] Running comprehensive evaluation comparison...")
    print("-" * 80)

    checkpoint_paths = list(checkpoints.values())

    try:
        compare_models(*checkpoint_paths, data_path="data/processed/")

        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE!")
        print("=" * 80)

        print("\nKey Insights for Ablation Study:")
        print("-" * 80)

        print("\n1. Feature Contribution Ranking:")
        print("   Compare RMSE improvements from v1 -> v2 -> v3 -> v4 -> v5")
        print("   The version with biggest drop shows most important feature")

        print("\n2. Extreme Event Performance:")
        print("   Check 90th percentile metrics - EVT loss should shine here")

        print("\n3. Long-Horizon Performance:")
        print("   Compare +6h RMSE - Direct decoding should reduce degradation")

        print("\n4. Overall Winner:")
        print("   Compare the ablation models against any extra checkpoints")
        print("   to see whether tuning beats architectural changes")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\nError during comparison: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
