"""
Profile computational cost of GCN-LSTM models.
Measures FLOPs, MACs, memory, and actual execution time.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GCNLSTMModel


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def estimate_flops_gcn(in_features, out_features, num_nodes, batch_size):
    """Estimate FLOPs for GCN layer: A @ X @ W"""
    # X @ W: (batch, nodes, in) @ (in, out) = batch * nodes * in * out * 2
    xw_flops = batch_size * num_nodes * in_features * out_features * 2
    # A @ (X @ W): (nodes, nodes) @ (batch, nodes, out) = batch * nodes * nodes * out * 2
    axw_flops = batch_size * num_nodes * num_nodes * out_features * 2
    return xw_flops + axw_flops


def estimate_flops_lstm_cell(input_dim, hidden_dim, num_nodes, batch_size):
    """Estimate FLOPs for LSTM cell gates."""
    # 4 gates, each: input_dim -> hidden_dim and hidden_dim -> hidden_dim
    # gates = W_i @ x + W_h @ h + b
    input_proj = batch_size * num_nodes * input_dim * hidden_dim * 4 * 2
    hidden_proj = batch_size * num_nodes * hidden_dim * hidden_dim * 4 * 2
    # Activations (sigmoid, tanh): ~10 FLOPs each
    activations = batch_size * num_nodes * hidden_dim * 4 * 10
    # Element-wise operations (i*g, f*c, etc.)
    elementwise = batch_size * num_nodes * hidden_dim * 6
    return input_proj + hidden_proj + activations + elementwise


def estimate_flops_attention(hidden_dim, num_heads, seq_len, num_nodes, batch_size):
    """Estimate FLOPs for multi-head attention."""
    head_dim = hidden_dim // num_heads
    # Q, K, V projections: 3 * (batch * nodes * hidden * hidden * 2)
    qkv_proj = 3 * batch_size * num_nodes * hidden_dim * hidden_dim * 2
    # Attention scores: Q @ K^T for each head
    # (batch*nodes, heads, 1, head_dim) @ (batch*nodes, heads, head_dim, seq_len)
    attn_scores = batch_size * num_nodes * num_heads * head_dim * seq_len * 2
    # Softmax: ~5 FLOPs per element
    softmax = batch_size * num_nodes * num_heads * seq_len * 5
    # Attention @ V
    attn_v = batch_size * num_nodes * num_heads * seq_len * head_dim * 2
    # Output projection
    out_proj = batch_size * num_nodes * hidden_dim * hidden_dim * 2
    return qkv_proj + attn_scores + softmax + attn_v + out_proj


def estimate_flops_dynamic_adjacency(batch_size, num_nodes, timesteps, num_categories):
    """Estimate FLOPs for dynamic wind-aware adjacency computation."""
    # Wind aggregation (einsum)
    wind_agg = batch_size * timesteps * num_nodes * 2  # speeds
    wind_agg += batch_size * timesteps * num_nodes * num_categories * 2  # directions

    # Argmax for direction
    argmax = batch_size * num_nodes * num_categories

    # Transport direction computation
    transport = batch_size * num_nodes * 3  # add, mod

    # Angle difference (batch, N, N)
    angle_diff = batch_size * num_nodes * num_nodes * 5  # abs, comparison, subtraction

    # Cosine alignment
    cosine = batch_size * num_nodes * num_nodes * 15  # deg2rad, cos, add, div

    # Distance decay (precomputed, but still applied)
    dist_decay = num_nodes * num_nodes * 3  # pow, div, exp

    # Wind factor
    wind_factor = batch_size * num_nodes * 5  # div, tanh

    # Combine A_wind
    combine = batch_size * num_nodes * num_nodes * 4  # multiply, add

    # Row normalization
    normalize = batch_size * num_nodes * num_nodes + batch_size * num_nodes * 2

    return wind_agg + argmax + transport + angle_diff + cosine + dist_decay + wind_factor + combine + normalize


def estimate_model_flops(config, batch_size=32):
    """Estimate total FLOPs for one forward pass."""
    hidden_dim = config['hidden_dim']
    input_dim = config['input_dim']
    num_nodes = config['num_nodes']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    seq_len = config.get('input_len', 24)
    horizon = config['horizon']
    use_direct = config.get('use_direct_decoding', False)
    use_wind_adj = config.get('use_wind_adjacency', False)

    total_flops = 0
    breakdown = {}

    # Input projection
    input_proj = batch_size * seq_len * num_nodes * input_dim * hidden_dim * 2
    breakdown['input_projection'] = input_proj
    total_flops += input_proj

    # Positional encoding (negligible)
    pos_enc = batch_size * seq_len * num_nodes * hidden_dim * 2
    breakdown['positional_encoding'] = pos_enc
    total_flops += pos_enc

    # Encoder: seq_len timesteps, num_layers layers
    encoder_flops = 0
    for t in range(seq_len):
        for layer in range(num_layers):
            # GCN for input
            encoder_flops += estimate_flops_gcn(hidden_dim, hidden_dim, num_nodes, batch_size)
            # GCN for hidden
            encoder_flops += estimate_flops_gcn(hidden_dim, hidden_dim, num_nodes, batch_size)
            # LSTM cell
            encoder_flops += estimate_flops_lstm_cell(hidden_dim, hidden_dim, num_nodes, batch_size)
            # LayerNorm
            encoder_flops += batch_size * num_nodes * hidden_dim * 5
    breakdown['encoder'] = encoder_flops
    total_flops += encoder_flops

    # Decoder
    decoder_flops = 0
    for h in range(horizon):
        # Attention over encoder outputs
        decoder_flops += estimate_flops_attention(hidden_dim, num_heads, seq_len, num_nodes, batch_size)

        # Context projection
        decoder_flops += batch_size * num_nodes * hidden_dim * 2 * hidden_dim * 2

        for layer in range(num_layers):
            # GCN for input
            decoder_flops += estimate_flops_gcn(hidden_dim, hidden_dim, num_nodes, batch_size)
            # GCN for hidden
            decoder_flops += estimate_flops_gcn(hidden_dim, hidden_dim, num_nodes, batch_size)
            # LSTM cell
            decoder_flops += estimate_flops_lstm_cell(hidden_dim, hidden_dim, num_nodes, batch_size)
            # LayerNorm
            decoder_flops += batch_size * num_nodes * hidden_dim * 5

        # Output projection
        decoder_flops += batch_size * num_nodes * hidden_dim * 1 * 2

        if use_direct:
            # Direct decoder: clone hidden states per step
            decoder_flops += num_layers * batch_size * num_nodes * hidden_dim * 2  # h and c clone

    breakdown['decoder'] = decoder_flops
    total_flops += decoder_flops

    # Dynamic adjacency
    if use_wind_adj:
        adj_flops = estimate_flops_dynamic_adjacency(batch_size, num_nodes, seq_len, 16)
        breakdown['dynamic_adjacency'] = adj_flops
        total_flops += adj_flops
    else:
        breakdown['dynamic_adjacency'] = 0

    return total_flops, breakdown


def profile_with_torch(model, X_sample, adj, config, device, num_runs=10, warmup=3):
    """Profile model using PyTorch's profiler."""
    model = model.to(device)
    model.eval()

    X_tensor = torch.FloatTensor(X_sample).to(device)
    if adj.ndim == 2:
        adj_tensor = torch.FloatTensor(adj).to(device)
    else:
        adj_tensor = torch.FloatTensor(adj).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(X_tensor, adj_tensor)

    # Synchronize
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(X_tensor, adj_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    # Memory
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(X_tensor, adj_tensor)
        memory_bytes = torch.cuda.max_memory_allocated()
    else:
        memory_bytes = 0

    return {
        'mean_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'min_time_ms': np.min(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
        'memory_mb': memory_bytes / (1024 * 1024)
    }


def profile_model_from_checkpoint(checkpoint_path, batch_size=32):
    """Load model from checkpoint and profile it."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']

    model = GCNLSTMModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        num_nodes=config['num_nodes'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        horizon=config['horizon'],
        use_direct_decoding=config.get('use_direct_decoding', False)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Count parameters
    trainable, total = count_parameters(model)

    # Estimate FLOPs
    config['input_len'] = 24  # Assuming 24 timesteps
    flops, breakdown = estimate_model_flops(config, batch_size)

    # Create dummy data for timing
    X_sample = np.random.randn(batch_size, 24, config['num_nodes'], config['input_dim']).astype(np.float32)
    adj = np.random.randn(config['num_nodes'], config['num_nodes']).astype(np.float32)

    # Profile
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timing = profile_with_torch(model, X_sample, adj, config, device)

    return {
        'checkpoint': os.path.basename(checkpoint_path),
        'config': config,
        'parameters': {
            'trainable': trainable,
            'total': total
        },
        'flops': {
            'total': flops,
            'breakdown': breakdown,
            'gflops': flops / 1e9
        },
        'timing': timing
    }


def compare_models(*checkpoint_paths, batch_size=32):
    """Compare computational cost of multiple models."""
    print("=" * 80)
    print("MODEL COMPUTATIONAL COST COMPARISON")
    print("=" * 80)
    print(f"\nBatch size: {batch_size}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    results = []
    for path in checkpoint_paths:
        print(f"\nProfiling: {os.path.basename(path)}")
        result = profile_model_from_checkpoint(path, batch_size)
        results.append(result)

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Parameters
    print("\n1. PARAMETERS")
    print("-" * 60)
    for r in results:
        print(f"  {r['checkpoint']}:")
        print(f"    Trainable: {r['parameters']['trainable']:,}")
        print(f"    Total:     {r['parameters']['total']:,}")

    # FLOPs
    print("\n2. FLOPs (per forward pass, batch={})".format(batch_size))
    print("-" * 60)
    for r in results:
        print(f"  {r['checkpoint']}:")
        print(f"    Total GFLOPs: {r['flops']['gflops']:.2f}")
        print(f"    Breakdown:")
        for component, flops in r['flops']['breakdown'].items():
            pct = (flops / r['flops']['total']) * 100
            print(f"      {component}: {flops/1e6:.2f}M ({pct:.1f}%)")

    # Timing
    print("\n3. EXECUTION TIME")
    print("-" * 60)
    for r in results:
        print(f"  {r['checkpoint']}:")
        print(f"    Mean: {r['timing']['mean_time_ms']:.2f} ms")
        print(f"    Std:  {r['timing']['std_time_ms']:.2f} ms")
        print(f"    Min:  {r['timing']['min_time_ms']:.2f} ms")
        print(f"    Max:  {r['timing']['max_time_ms']:.2f} ms")

    # Memory (GPU only)
    if torch.cuda.is_available():
        print("\n4. GPU MEMORY")
        print("-" * 60)
        for r in results:
            print(f"  {r['checkpoint']}: {r['timing']['memory_mb']:.2f} MB")

    # Summary comparison
    if len(results) == 2:
        print("\n5. COMPARISON SUMMARY")
        print("-" * 60)
        r1, r2 = results

        flop_ratio = r2['flops']['total'] / r1['flops']['total']
        time_ratio = r2['timing']['mean_time_ms'] / r1['timing']['mean_time_ms']
        param_ratio = r2['parameters']['trainable'] / r1['parameters']['trainable']

        print(f"  {r2['checkpoint']} vs {r1['checkpoint']}:")
        print(f"    FLOPs:      {flop_ratio:.2f}x {'more' if flop_ratio > 1 else 'less'}")
        print(f"    Time:       {time_ratio:.2f}x {'slower' if time_ratio > 1 else 'faster'}")
        print(f"    Parameters: {param_ratio:.2f}x {'more' if param_ratio > 1 else 'less'}")

        # Efficiency
        efficiency_1 = r1['flops']['total'] / r1['timing']['mean_time_ms']
        efficiency_2 = r2['flops']['total'] / r2['timing']['mean_time_ms']
        print(f"\n  Computational Efficiency (GFLOPs/sec):")
        print(f"    {r1['checkpoint']}: {efficiency_1/1e6:.2f}")
        print(f"    {r2['checkpoint']}: {efficiency_2/1e6:.2f}")

    print("\n" + "=" * 80)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python profile_models.py <checkpoint1> [checkpoint2] [--batch_size N]")
        print("\nExample:")
        print("  python profile_models.py models/checkpoints/model1.pt models/checkpoints/model2.pt")
        sys.exit(1)

    # Parse arguments
    checkpoint_paths = []
    batch_size = 32

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--batch_size':
            batch_size = int(sys.argv[i + 1])
            i += 2
        else:
            checkpoint_paths.append(sys.argv[i])
            i += 1

    if len(checkpoint_paths) == 1:
        result = profile_model_from_checkpoint(checkpoint_paths[0], batch_size)
        print(f"\nModel: {result['checkpoint']}")
        print(f"Parameters: {result['parameters']['trainable']:,}")
        print(f"GFLOPs: {result['flops']['gflops']:.2f}")
        print(f"Time: {result['timing']['mean_time_ms']:.2f} ms")
    else:
        compare_models(*checkpoint_paths, batch_size=batch_size)
