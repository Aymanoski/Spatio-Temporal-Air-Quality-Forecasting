"""
Compare all trained versions and generate comprehensive ablation study report.
"""

import os
import sys
import glob
from pathlib import Path

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.evaluate_comparison import compare_models


def find_version_checkpoints(checkpoint_dir="models/checkpoints/"):
    """
    Find all version checkpoints automatically.
    
    Returns:
        Dictionary mapping version names to checkpoint paths
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return {}
    
    # Look for version checkpoints (v1_*, v2_*, etc.)
    patterns = [
        'v1_baseline_*_best.pt',
        'v2_direct_decoding_*_best.pt',
        'v3_evt_loss_*_best.pt',
        'v4_wind_adjacency_*_best.pt',
        'v5_full_optimized_*_best.pt'
    ]
    
    found_checkpoints = {}
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(checkpoint_dir, pattern))
        if matches:
            # Take the first match if multiple exist
            version_key = pattern.split('_*')[0]  # Extract v1, v2, etc.
            found_checkpoints[version_key] = matches[0]
    
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
    
    version_order = ['v1_baseline', 'v2_direct_decoding', 'v3_evt_loss', 
                     'v4_wind_adjacency', 'v5_full_optimized']
    
    # Map version to feature added
    feature_map = {
        'v1_baseline': 'Baseline (Autoregressive + MSE + Static Adj)',
        'v2_direct_decoding': '+ Direct Multi-Horizon Decoding',
        'v3_evt_loss': '+ EVT Hybrid Loss (Extreme Value)',
        'v4_wind_adjacency': '+ Wind-Aware Dynamic Adjacency',
        'v5_full_optimized': '+ Adaptive EVT Schedule + Asymmetric Penalty'
    }
    
    for i, version in enumerate(version_order, 1):
        if version in checkpoints:
            print(f"\n{i}. {feature_map[version]}")
            print(f"   Checkpoint: {Path(checkpoints[version]).name}")
        else:
            print(f"\n{i}. {feature_map[version]}")
            print(f"   Status: NOT TRAINED YET")
    
    print("\n" + "=" * 80)


def main():
    """Main comparison function."""
    print("=" * 80)
    print("VERSION COMPARISON TOOL")
    print("=" * 80)
    
    # Find checkpoints
    print("\n[1/2] Searching for version checkpoints...")
    checkpoints = find_version_checkpoints()
    
    if not checkpoints:
        print("\nNo version checkpoints found!")
        print("Please train versions first using:")
        print("  python train_versions.py --all")
        return
    
    print(f"\nFound {len(checkpoints)} version checkpoints:")
    for version, path in sorted(checkpoints.items()):
        print(f"  ✓ {version}: {Path(path).name}")
    
    # Print ablation structure
    print_ablation_analysis(checkpoints)
    
    # Run comparison
    print("\n[2/2] Running comprehensive evaluation comparison...")
    print("-" * 80)
    
    checkpoint_paths = [checkpoints[v] for v in sorted(checkpoints.keys())]
    
    try:
        results = compare_models(*checkpoint_paths, data_path="data/processed/")
        
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
        print("   v5 should be best overall, but check if simpler versions")
        print("   achieve 90% of performance with less complexity")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\nError during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
