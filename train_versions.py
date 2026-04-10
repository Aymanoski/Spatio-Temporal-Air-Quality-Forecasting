"""
Systematic version testing for GCN-LSTM ablation study.
Runs 5 versions to isolate the contribution of each architectural component.
"""

import torch
import sys
import os
from train import train, CONFIG

# ============================================================================
# Version Configurations
# ============================================================================

VERSION_CONFIGS = {
    'v1_baseline': {
        'name': 'Version 1: Baseline',
        'description': 'Autoregressive decoder + MSE loss + static adjacency',
        'config': {
            **CONFIG,
            'use_wind_adjacency': False,
            'loss_type': 'mse',
            'use_direct_decoding': False,
            'batch_size': 64,
            'learning_rate': 2e-3,
            'architecture_name': 'v1_baseline',
            'use_versioned_checkpoint': True,
        }
    },
    
    'v2_direct_decoding': {
        'name': 'Version 2: + Direct Decoding',
        'description': 'Direct multi-horizon decoder (no autoregression)',
        'config': {
            **CONFIG,
            'use_wind_adjacency': False,
            'loss_type': 'mse',
            'use_direct_decoding': True,  # NEW: Direct decoding
            'batch_size': 64,
            'learning_rate': 2e-3,
            'architecture_name': 'v2_direct_decoding',
            'use_versioned_checkpoint': True,
        }
    },
    
    'v3_evt_loss': {
        'name': 'Version 3: + EVT Hybrid Loss',
        'description': 'Direct decoding + EVT loss (extreme value awareness)',
        'config': {
            **CONFIG,
            'use_wind_adjacency': False,
            'loss_type': 'evt_hybrid',  # NEW: EVT loss
            'use_direct_decoding': True,
            'batch_size': 64,
            'learning_rate': 1.5e-3,  # Slightly lower for EVT stability
            'evt_lambda': 0.05,
            'evt_use_lambda_schedule': False,
            'evt_asymmetric_penalty': False,
            'architecture_name': 'v3_evt_loss',
            'use_versioned_checkpoint': True,
        }
    },
    
    'v4_wind_adjacency': {
        'name': 'Version 4: + Wind-Aware Adjacency',
        'description': 'Direct decoding + EVT + dynamic wind-aware graphs',
        'config': {
            **CONFIG,
            'use_wind_adjacency': True,  # NEW: Wind adjacency
            'loss_type': 'evt_hybrid',
            'use_direct_decoding': True,
            'batch_size': 32,  # Lower for dynamic adjacency memory cost
            'learning_rate': 1e-3,
            'evt_lambda': 0.05,
            'evt_use_lambda_schedule': False,
            'evt_asymmetric_penalty': False,
            'architecture_name': 'v4_wind_adjacency',
            'use_versioned_checkpoint': True,
        }
    },
    
    'v5_full_optimized': {
        'name': 'Version 5: Full + Adaptive Tuning',
        'description': 'All features + adaptive EVT lambda schedule',
        'config': {
            **CONFIG,
            'use_wind_adjacency': True,
            'loss_type': 'evt_hybrid',
            'use_direct_decoding': True,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'evt_lambda': 0.05,  # Initial lambda
            'evt_use_lambda_schedule': True,  # NEW: Adaptive lambda
            'evt_asymmetric_penalty': True,   # NEW: Asymmetric penalty
            'evt_under_penalty_multiplier': 2.0,
            'evt_lambda_schedule': {
                'initial': 0.05,
                'mid': 0.12,
                'final': 0.25,
                'warmup_epochs': 25,
                'mid_epochs': 50,
                'transition': 'smooth'
            },
            'architecture_name': 'v5_full_optimized',
            'use_versioned_checkpoint': True,
        }
    },
}


# ============================================================================
# Helper Functions
# ============================================================================

def print_version_info(version_key):
    """Print version configuration details."""
    version = VERSION_CONFIGS[version_key]
    config = version['config']
    
    print("\n" + "=" * 70)
    print(f"{version['name']}")
    print("=" * 70)
    print(f"Description: {version['description']}")
    print("\nConfiguration:")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Direct Decoding: {config['use_direct_decoding']}")
    print(f"  Wind Adjacency: {config['use_wind_adjacency']}")
    print(f"  Loss Type: {config['loss_type']}")
    
    if config['loss_type'] == 'evt_hybrid':
        print(f"  EVT Lambda: {config.get('evt_lambda', 0.05)}")
        print(f"  EVT Lambda Schedule: {config.get('evt_use_lambda_schedule', False)}")
        print(f"  EVT Asymmetric Penalty: {config.get('evt_asymmetric_penalty', False)}")
    
    print(f"\nCheckpoint: models/checkpoints/{config['architecture_name']}_{config['hardware_tag']}_best.pt")
    print("=" * 70)


def train_version(version_key):
    """Train a specific version."""
    version = VERSION_CONFIGS[version_key]
    config = version['config']
    
    print_version_info(version_key)
    
    print("\n[STARTING TRAINING]")
    print("-" * 70)
    
    model, history, metrics = train(config)
    
    print("\n[TRAINING COMPLETE]")
    print("-" * 70)
    
    return model, history, metrics


def train_all_versions(start_from=None, up_to=None):
    """
    Train all versions sequentially.
    
    Args:
        start_from: Version key to start from (e.g., 'v3_evt_loss' to skip v1 and v2)
        up_to: Version key to stop at (e.g., 'v3_evt_loss' to train only first 3)
    """
    version_keys = list(VERSION_CONFIGS.keys())

    if start_from and start_from not in version_keys:
        print(f"Error: Unknown version '{start_from}'")
        print(f"Available versions: {', '.join(version_keys)}")
        return

    if up_to and up_to not in version_keys:
        print(f"Error: Unknown version '{up_to}'")
        print(f"Available versions: {', '.join(version_keys)}")
        return

    if start_from and up_to:
        start_idx = version_keys.index(start_from)
        end_idx = version_keys.index(up_to)
        if start_idx > end_idx:
            print(f"Error: start-from version '{start_from}' comes after up-to version '{up_to}'")
            return
        version_keys = version_keys[start_idx:end_idx + 1]
        print(f"\n>>> Training range: {start_from} -> {up_to}")
    elif start_from:
        start_idx = version_keys.index(start_from)
        version_keys = version_keys[start_idx:]
        print(f"\n>>> Starting from {start_from} (skipping previous versions)")
    elif up_to:
        end_idx = version_keys.index(up_to)
        version_keys = version_keys[:end_idx + 1]
        print(f"\n>>> Training up to {up_to} (starting from first version)")
    
    print("\n" + "=" * 70)
    print("ABLATION STUDY: TRAINING ALL VERSIONS")
    print("=" * 70)
    print(f"\nTotal versions to train: {len(version_keys)}")
    print(f"Versions: {', '.join(version_keys)}")
    print("\nThis will take several hours. Each version:")
    print("  - Trains for up to 100 epochs")
    print("  - Has early stopping (patience=15)")
    print("  - Saves best checkpoint automatically")
    print("\nYou can stop and resume later using --start-from or --up-to options.")
    
    results = {}
    
    for i, version_key in enumerate(version_keys, 1):
        print(f"\n\n{'#' * 70}")
        print(f"# TRAINING VERSION {i}/{len(version_keys)}: {version_key}")
        print(f"{'#' * 70}\n")
        
        try:
            model, history, metrics = train_version(version_key)
            results[version_key] = {
                'metrics': metrics,
                'history': history,
                'status': 'completed'
            }
            
            print(f"\n✓ {version_key} completed successfully!")
            print(f"  Test RMSE: {metrics['RMSE']:.4f}")
            print(f"  Test MAE: {metrics['MAE']:.4f}")
            print(f"  Test R²: {metrics.get('R2', 'N/A')}")
            
        except Exception as e:
            print(f"\n✗ {version_key} failed with error: {e}")
            results[version_key] = {
                'status': 'failed',
                'error': str(e)
            }
            
            # Ask user if they want to continue
            print("\nOptions:")
            print("  1. Continue with next version")
            print("  2. Stop here")
            choice = input("Your choice (1/2): ").strip()
            
            if choice == '2':
                print("Stopping training sequence.")
                break
    
    # Print summary
    print("\n\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    for version_key, result in results.items():
        status = result['status']
        if status == 'completed':
            print(f"\n✓ {version_key}: COMPLETED")
            print(f"  RMSE: {result['metrics']['RMSE']:.4f}")
            print(f"  MAE: {result['metrics']['MAE']:.4f}")
        else:
            print(f"\n✗ {version_key}: FAILED - {result.get('error', 'unknown error')}")
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Compare results: python utils/evaluate_comparison.py models/checkpoints/v*_best.pt")
    print("  2. Analyze specific checkpoint: python utils/compare_models.py models/checkpoints/v1_baseline_T4_best.pt")
    print("=" * 70)
    
    return results


def list_versions():
    """List all available versions with descriptions."""
    print("\n" + "=" * 70)
    print("AVAILABLE VERSIONS")
    print("=" * 70)
    
    for key, version in VERSION_CONFIGS.items():
        config = version['config']
        print(f"\n{key}:")
        print(f"  Name: {version['name']}")
        print(f"  Description: {version['description']}")
        print(f"  Features:")
        print(f"    - Direct Decoding: {config['use_direct_decoding']}")
        print(f"    - Wind Adjacency: {config['use_wind_adjacency']}")
        print(f"    - Loss Type: {config['loss_type']}")
        print(f"    - Batch Size: {config['batch_size']}")
        print(f"    - Learning Rate: {config['learning_rate']}")
    
    print("\n" + "=" * 70)


# ============================================================================
# CLI Interface
# ============================================================================

def print_usage():
    """Print usage instructions."""
    print("\nUsage:")
    print("  Train all versions:")
    print("    python train_versions.py --all")
    print("\n  Train first N versions (e.g., first 3):")
    print("    python train_versions.py --all --up-to v3_evt_loss")
    print("\n  Train specific version:")
    print("    python train_versions.py --version v1_baseline")
    print("\n  Resume from specific version:")
    print("    python train_versions.py --all --start-from v3_evt_loss")
    print("\n  Train a specific range:")
    print("    python train_versions.py --all --start-from v2_direct_decoding --up-to v4_wind_adjacency")
    print("\n  List available versions:")
    print("    python train_versions.py --list")
    print("\nAvailable versions:")
    for key in VERSION_CONFIGS.keys():
        print(f"  - {key}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage()
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == '--list':
        list_versions()
    
    elif command == '--all':
        start_from = None
        up_to = None
        args = sys.argv[2:]

        i = 0
        while i < len(args):
            arg = args[i]

            if arg == '--start-from':
                if i + 1 >= len(args):
                    print("Error: Missing value for --start-from")
                    print_usage()
                    sys.exit(1)
                if start_from is not None:
                    print("Error: --start-from specified more than once")
                    print_usage()
                    sys.exit(1)
                start_from = args[i + 1]
                i += 2
            elif arg == '--up-to':
                if i + 1 >= len(args):
                    print("Error: Missing value for --up-to")
                    print_usage()
                    sys.exit(1)
                if up_to is not None:
                    print("Error: --up-to specified more than once")
                    print_usage()
                    sys.exit(1)
                up_to = args[i + 1]
                i += 2
            else:
                print(f"Error: Unknown argument '{arg}' for --all")
                print_usage()
                sys.exit(1)

        train_all_versions(start_from=start_from, up_to=up_to)
    
    elif command == '--version':
        if len(sys.argv) < 3:
            print("Error: Please specify version key")
            print_usage()
            sys.exit(1)
        
        version_key = sys.argv[2]
        if version_key not in VERSION_CONFIGS:
            print(f"Error: Unknown version '{version_key}'")
            print_usage()
            sys.exit(1)
        
        train_version(version_key)
    
    else:
        print(f"Error: Unknown command '{command}'")
        print_usage()
        sys.exit(1)
