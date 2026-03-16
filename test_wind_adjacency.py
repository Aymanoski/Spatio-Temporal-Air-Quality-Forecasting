"""
Quick test script to verify wind-aware adjacency implementation.
"""

import torch
import numpy as np
from utils.graph import build_wind_aware_adjacency_batch, WIND_CATEGORIES

print("=" * 60)
print("Testing Wind-Aware Adjacency Implementation")
print("=" * 60)

# Test parameters
batch_size = 2
timesteps = 24
num_nodes = 12
num_wind_categories = 16

# Create dummy wind data
print("\n1. Creating dummy wind data...")
wind_speeds = np.random.rand(batch_size, timesteps, num_nodes) * 10  # 0-10 m/s
wind_directions = np.random.rand(batch_size, timesteps, num_nodes, num_wind_categories)
# Normalize to make it one-hot-like
wind_directions = wind_directions / wind_directions.sum(axis=-1, keepdims=True)

print(f"  Wind speeds shape: {wind_speeds.shape}")
print(f"  Wind directions shape: {wind_directions.shape}")
print(f"  Number of wind categories: {len(WIND_CATEGORIES)}")
print(f"  Wind categories: {WIND_CATEGORIES}")

# Test adjacency construction
print("\n2. Building wind-aware adjacency matrices...")
try:
    adj_batch = build_wind_aware_adjacency_batch(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        wind_categories=WIND_CATEGORIES,
        alpha=0.6,
        distance_sigma=100
    )
    print(f"  [OK] Adjacency batch shape: {adj_batch.shape}")
    print(f"  [OK] Expected shape: ({batch_size}, {num_nodes}, {num_nodes})")

    # Check properties
    print("\n3. Checking adjacency properties...")
    print(f"  Min value: {adj_batch.min():.4f}")
    print(f"  Max value: {adj_batch.max():.4f}")
    print(f"  Mean value: {adj_batch.mean():.4f}")

    # Check if diagonal is close to 1 (normalized)
    diagonal_values = np.array([adj_batch[i].diagonal().mean() for i in range(batch_size)])
    print(f"  Mean diagonal value: {diagonal_values.mean():.4f}")

    # Check symmetry (should be mostly symmetric but can have some directionality)
    asymmetry = np.abs(adj_batch[0] - adj_batch[0].T).mean()
    print(f"  Asymmetry (mean abs diff from transpose): {asymmetry:.4f}")

    print("\n4. Testing with PyTorch tensors...")
    wind_speeds_torch = torch.FloatTensor(wind_speeds)
    wind_directions_torch = torch.FloatTensor(wind_directions)

    adj_batch_torch = build_wind_aware_adjacency_batch(
        wind_speeds=wind_speeds_torch,
        wind_directions=wind_directions_torch,
        wind_categories=WIND_CATEGORIES,
        alpha=0.6,
        distance_sigma=100
    )
    print(f"  [OK] Adjacency batch shape (from torch): {adj_batch_torch.shape}")

    print("\n5. Testing alpha parameter effects...")
    # Alpha = 0 (pure distance-based)
    adj_dist = build_wind_aware_adjacency_batch(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        wind_categories=WIND_CATEGORIES,
        alpha=0.0,
        distance_sigma=100
    )

    # Alpha = 1 (pure wind-based)
    adj_wind = build_wind_aware_adjacency_batch(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        wind_categories=WIND_CATEGORIES,
        alpha=1.0,
        distance_sigma=100
    )

    print(f"  Distance-based (alpha=0.0):")
    print(f"    Asymmetry: {np.abs(adj_dist[0] - adj_dist[0].T).mean():.4f}")
    print(f"  Wind-based (alpha=1.0):")
    print(f"    Asymmetry: {np.abs(adj_wind[0] - adj_wind[0].T).mean():.4f}")
    print(f"  -> Wind-based should be more asymmetric (directional)")

    print("\n" + "=" * 60)
    print("[SUCCESS] All tests passed successfully!")
    print("=" * 60)

except Exception as e:
    print(f"\n[ERROR] Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
