"""
Migrate old checkpoint to new versioned format with metadata.

Usage:
    python migrate_checkpoint.py
"""

import torch
import os
import time


def migrate_checkpoint(
    old_path='models/checkpoints/best_model.pt',
    new_architecture_name='gcn_lstm_v1',
    new_hardware_tag='integrated_gpu'
):
    """
    Migrate old checkpoint to new format with metadata.

    Args:
        old_path: Path to old checkpoint
        new_architecture_name: Architecture name for versioning
        new_hardware_tag: Hardware tag (e.g., 'integrated_gpu', 'colab_t4')
    """
    if not os.path.exists(old_path):
        print(f"Error: Checkpoint not found at {old_path}")
        return False

    print("=" * 60)
    print("CHECKPOINT MIGRATION")
    print("=" * 60)

    # Load old checkpoint
    print(f"\n[1/3] Loading old checkpoint: {old_path}")
    old_checkpoint = torch.load(old_path, map_location='cpu', weights_only=False)

    # Create new checkpoint with metadata
    print(f"[2/3] Adding metadata...")

    # Extract config if available
    config = old_checkpoint.get('config', {})

    # Count parameters from model state dict
    num_params = sum(p.numel() for p in old_checkpoint['model_state_dict'].values())

    new_checkpoint = {
        'epoch': old_checkpoint.get('epoch', 'N/A'),
        'model_state_dict': old_checkpoint['model_state_dict'],
        'optimizer_state_dict': old_checkpoint.get('optimizer_state_dict'),
        'val_loss': old_checkpoint.get('val_loss', 'N/A'),
        'train_loss': old_checkpoint.get('train_loss', 'N/A'),
        'config': config,
        'hardware': {
            'device': 'cpu',  # Old checkpoint didn't save this
            'tag': new_hardware_tag,
            'cuda_available': False,
            'cuda_device_name': None
        },
        'architecture': {
            'name': new_architecture_name,
            'num_params': num_params,
            'use_direct_decoding': config.get('use_direct_decoding', False),
            'use_wind_adjacency': config.get('use_wind_adjacency', False)
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'migrated': True,
        'original_path': old_path
    }

    # Create new filename
    new_filename = f"{new_architecture_name}_{new_hardware_tag}_best.pt"
    new_path = os.path.join(os.path.dirname(old_path), new_filename)

    # Save new checkpoint
    print(f"[3/3] Saving to: {new_path}")
    torch.save(new_checkpoint, new_path)

    print("\n" + "=" * 60)
    print("Migration Summary:")
    print("-" * 60)
    print(f"  Old file: {old_path}")
    print(f"  New file: {new_path}")
    print(f"  Architecture: {new_architecture_name}")
    print(f"  Hardware tag: {new_hardware_tag}")
    print(f"  Parameters: {num_params:,}")
    print(f"  Validation Loss: {new_checkpoint['val_loss']}")
    print(f"  Epoch: {new_checkpoint['epoch']}")
    print("=" * 60)

    print("\nMigration complete!")
    print(f"  Original checkpoint kept at: {old_path}")
    print(f"  New versioned checkpoint: {new_path}")

    return True


if __name__ == "__main__":
    # Migrate the existing best_model.pt to versioned format
    migrate_checkpoint(
        old_path='models/checkpoints/best_model.pt',
        new_architecture_name='gcn_lstm_v1',
        new_hardware_tag='integrated_gpu'
    )
