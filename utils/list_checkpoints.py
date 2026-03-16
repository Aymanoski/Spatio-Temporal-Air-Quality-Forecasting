"""
Quick checkpoint overview - list all checkpoints in the directory.

Usage:
    python list_checkpoints.py
"""

import os
import torch
from pathlib import Path


def list_checkpoints(checkpoint_dir='models/checkpoints'):
    """List all checkpoint files with key information."""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return

    # Find all checkpoint files
    checkpoint_files = list(Path(checkpoint_dir).glob('*.pt')) + \
                       list(Path(checkpoint_dir).glob('*.pth')) + \
                       list(Path(checkpoint_dir).glob('*.ckpt'))

    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print("=" * 80)
    print(f"CHECKPOINTS in {checkpoint_dir}")
    print("=" * 80)

    checkpoints_info = []

    for ckpt_path in sorted(checkpoint_files):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            # Extract key info
            info = {
                'file': ckpt_path.name,
                'size_mb': ckpt_path.stat().st_size / (1024 * 1024),
                'val_loss': ckpt.get('val_loss', 'N/A'),
                'epoch': ckpt.get('epoch', 'N/A'),
            }

            # Get architecture and hardware if available
            if 'architecture' in ckpt:
                info['arch'] = ckpt['architecture'].get('name', 'N/A')
                info['params'] = ckpt['architecture'].get('num_params', 'N/A')
            else:
                info['arch'] = 'N/A'
                info['params'] = 'N/A'

            if 'hardware' in ckpt:
                info['hardware'] = ckpt['hardware'].get('tag', 'N/A')
            else:
                info['hardware'] = 'N/A'

            checkpoints_info.append(info)

        except Exception as e:
            print(f"\nWarning: Could not load {ckpt_path.name}: {e}")
            continue

    # Print table
    if checkpoints_info:
        print()
        # Header
        print(f"{'File':<40} {'Arch':<15} {'Hardware':<15} {'Val Loss':<12} {'Epoch':<8} {'Size (MB)':<10}")
        print("-" * 110)

        # Rows
        for info in checkpoints_info:
            val_loss_str = f"{info['val_loss']:.6f}" if isinstance(info['val_loss'], (int, float)) else str(info['val_loss'])
            print(f"{info['file']:<40} {info['arch']:<15} {info['hardware']:<15} {val_loss_str:<12} {str(info['epoch']):<8} {info['size_mb']:<10.2f}")

        print("\n" + "=" * 80)
        print(f"Total: {len(checkpoints_info)} checkpoint(s)")

        # Find best model
        valid_losses = [(i, info['val_loss']) for i, info in enumerate(checkpoints_info)
                       if isinstance(info['val_loss'], (int, float))]

        if valid_losses:
            best_idx, best_loss = min(valid_losses, key=lambda x: x[1])
            print(f"\nBest model: {checkpoints_info[best_idx]['file']} (val_loss: {best_loss:.6f})")

        print("=" * 80)


if __name__ == "__main__":
    list_checkpoints()
