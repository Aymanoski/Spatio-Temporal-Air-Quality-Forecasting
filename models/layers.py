"""
Core layers for GCN-LSTM Encoder-Decoder model.
- GraphConvolution: Spatial feature extraction via graph convolution
- GraphLSTMCell: Combined GCN + LSTM for spatio-temporal learning
- MultiHeadAttention: Temporal attention mechanism for decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConvolution(nn.Module):
    """
    Graph Convolutional Layer.
    Performs spatial aggregation using normalized adjacency matrix.
    
    X' = A_hat @ X @ W + b
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Args:
            x: Node features (batch, num_nodes, in_features)
            adj: Normalized adjacency matrix
                 - Static: (num_nodes, num_nodes)
                 - Dynamic: (batch, num_nodes, num_nodes)
        Returns:
            out: (batch, num_nodes, out_features)
        """
        # x @ W: (batch, num_nodes, out_features)
        support = torch.matmul(x, self.weight)

        # A_hat @ (x @ W): spatial aggregation
        # torch.matmul handles both static (2D) and dynamic (3D) adjacency:
        # - Static: (num_nodes, num_nodes) @ (batch, num_nodes, out_features) -> (batch, num_nodes, out_features)
        # - Dynamic: (batch, num_nodes, num_nodes) @ (batch, num_nodes, out_features) -> (batch, num_nodes, out_features)
        out = torch.matmul(adj, support)

        if self.bias is not None:
            out = out + self.bias

        return out


class GraphLSTMCell(nn.Module):
    """
    Graph LSTM Cell: Combines GCN spatial extraction with LSTM temporal modeling.
    
    At each timestep:
    1. Apply GCN to extract spatial features from node representations
    2. Feed GCN output into LSTM gates for temporal modeling
    
    This captures both spatial dependencies (via GCN) and temporal dynamics (via LSTM).
    """
    
    def __init__(self, input_dim, hidden_dim, num_nodes, bias=True, graph_conv='gcn', dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        # Spatial feature extraction on input and hidden state
        # graph_conv='gcn': original GraphConvolution (default, backward-compatible)
        # graph_conv='gat': GraphAttentionLayer with wind-aware adjacency bias
        if graph_conv == 'gat':
            self.gcn_i = GraphAttentionLayer(input_dim, hidden_dim, dropout=dropout)
            self.gcn_h = GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout)
        else:
            self.gcn_i = GraphConvolution(input_dim, hidden_dim)
            self.gcn_h = GraphConvolution(hidden_dim, hidden_dim)

        # LSTM gates: input, forget, cell, output
        # Combined linear transformation for efficiency
        self.gates = nn.Linear(hidden_dim * 2, hidden_dim * 4, bias=bias)
        
    def forward(self, x, hidden, adj):
        """
        Args:
            x: Input at current timestep (batch, num_nodes, input_dim)
            hidden: Tuple of (h, c) each (batch, num_nodes, hidden_dim)
            adj: Adjacency matrix
                 - Static: (num_nodes, num_nodes)
                 - Dynamic: (batch, num_nodes, num_nodes)
        Returns:
            h_new, c_new: Updated hidden and cell states
        """
        h, c = hidden

        # Apply GCN to input and hidden state (spatial aggregation)
        # Using LeakyReLU to prevent dead neurons and improve gradient flow
        x_gcn = F.leaky_relu(self.gcn_i(x, adj), negative_slope=0.1)  # (batch, num_nodes, hidden_dim)
        h_gcn = F.leaky_relu(self.gcn_h(h, adj), negative_slope=0.1)  # (batch, num_nodes, hidden_dim)
        
        # Concatenate for gate computation
        combined = torch.cat([x_gcn, h_gcn], dim=-1)  # (batch, num_nodes, hidden_dim*2)
        
        # Compute all gates at once
        gates = self.gates(combined)  # (batch, num_nodes, hidden_dim*4)
        
        # Split into individual gates
        i, f, g, o = gates.chunk(4, dim=-1)
        
        # Apply gate activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell candidate
        o = torch.sigmoid(o)  # Output gate
        
        # Update cell state and hidden state
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden and cell states to zeros."""
        h = torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=device)
        return h, c


class GraphAttentionLayer(nn.Module):
    """
    Multi-head Graph Attention Layer — supports GATv1 and GATv2.

    version='v1'  (standard GAT):
        e_ij = LeakyReLU(a^T [W h_i || W h_j])
        Attention is "static": LeakyReLU is applied after the dot product with a,
        making the score decomposable into independent source and target terms.

    version='v2'  (GATv2, Brody et al. 2022):
        e_ij = a^T LeakyReLU(W_src h_i + W_dst h_j)
        Attention is "dynamic": LeakyReLU is applied before the dot product,
        breaking the decomposability and allowing neighbour rankings to depend
        on the query node. More expressive than v1 at the same depth.

    Both versions share:
      - Wind-aware adjacency as additive bias before softmax:
            adj[i,j] > 0  →  +adj[i,j] (boost wind-aligned edges)
            adj[i,j] = 0  →  -1e9      (hard mask)
      - Multi-head with head concatenation.
      - LeakyReLU(0.2).
      - Dropout on attention weights.

    Args:
        in_features:  Input feature dimension
        out_features: Output feature dimension (must be divisible by num_heads)
        num_heads:    Attention heads. Auto-selected if None: prefer 4, then 2, then 1.
        dropout:      Dropout probability applied to attention weights
        version:      'v1' (standard GAT) or 'v2' (GATv2)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = None,
        dropout: float = 0.1,
        version: str = 'v1',
        use_edge_features: bool = False,
    ):
        super().__init__()

        # Auto-select num_heads
        if num_heads is None:
            for h in [4, 2, 1]:
                if out_features % h == 0:
                    num_heads = h
                    break

        assert out_features % num_heads == 0, (
            f"out_features ({out_features}) must be divisible by num_heads ({num_heads})"
        )
        assert version in ('v1', 'v2'), f"version must be 'v1' or 'v2', got '{version}'"

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.version = version
        self.use_edge_features = use_edge_features

        if version == 'v2':
            # Two separate projections: source and target nodes projected independently,
            # then summed before the nonlinearity.
            self.W_src = nn.Linear(in_features, out_features, bias=False)
            self.W_dst = nn.Linear(in_features, out_features, bias=False)
            # Attention vector per head: a_h ∈ R^(head_dim)  [applied after LeakyReLU]
            self.attn_vec = nn.Parameter(torch.empty(num_heads, self.head_dim))
        else:
            # Shared projection for both source and target nodes.
            self.W = nn.Linear(in_features, out_features, bias=False)
            # Attention vector per head: a_h ∈ R^(2*head_dim)
            self.attn_vec = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))

        nn.init.xavier_uniform_(self.attn_vec)

        self.bias = nn.Parameter(torch.zeros(out_features))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout)

        # Edge-conditioned value projection: maps scalar adj weight → hidden_dim.
        # Zero-init so training starts identical to baseline GAT.
        if use_edge_features:
            self.W_edge = nn.Linear(1, out_features, bias=False)
            nn.init.zeros_(self.W_edge.weight)
        else:
            self.W_edge = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor, geo_bias: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:        (B, N, in_features)
            adj:      (N, N) static  or  (B, N, N) dynamic — wind-aware normalized adjacency
            geo_bias: (N, N, H) optional static geographic distance bias, broadcast over B
        Returns:
            out: (B, N, out_features)
        """
        assert x.dim() == 3, f"GAT expects (B, N, F), got {tuple(x.shape)}"
        assert x.size(-1) == self.in_features, (
            f"in_features mismatch: expected {self.in_features}, got {x.size(-1)}"
        )

        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim

        if self.version == 'v2':
            # GATv2: e_ij = a^T LeakyReLU(W_src h_i + W_dst h_j)
            Wh_src = self.W_src(x).view(B, N, H, D)   # (B, N, H, D)
            Wh_dst = self.W_dst(x).view(B, N, H, D)   # (B, N, H, D)

            # Expand and add for all (i, j) pairs
            src_i = Wh_src.unsqueeze(2).expand(-1, -1, N, -1, -1)  # (B, N, N, H, D)
            dst_j = Wh_dst.unsqueeze(1).expand(-1, N, -1, -1, -1)  # (B, N, N, H, D)

            # Nonlinearity before dot product — this is what makes v2 "dynamic"
            e = self.leaky_relu(src_i + dst_j)                      # (B, N, N, H, D)
            attn_scores = (e * self.attn_vec.view(1, 1, 1, H, D)).sum(-1)  # (B, N, N, H)

            # Values for aggregation: reuse W_dst projections
            Wh_val = Wh_dst
        else:
            # Standard GATv1: e_ij = LeakyReLU(a^T [W h_i || W h_j])
            Wh = self.W(x).view(B, N, H, D)

            Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1, -1)   # (B, N, N, H, D)
            Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1, -1)   # (B, N, N, H, D)
            Wh_cat = torch.cat([Wh_i, Wh_j], dim=-1)            # (B, N, N, H, 2D)

            attn_scores = (Wh_cat * self.attn_vec.view(1, 1, 1, H, -1)).sum(-1)
            attn_scores = self.leaky_relu(attn_scores)           # (B, N, N, H)

            Wh_val = Wh

        # Wind-aware adjacency bias (shared between v1 and v2)
        if adj.dim() == 2:
            adj_b = adj.unsqueeze(0)
        else:
            adj_b = adj

        adj_bias = torch.where(
            adj_b > 1e-9,
            adj_b,
            adj_b.new_full(adj_b.shape, -1e9)
        )
        attn_scores = attn_scores + adj_bias.unsqueeze(-1)

        if geo_bias is not None:
            # geo_bias: (N, N, H) → (1, N, N, H) broadcast over batch
            attn_scores = attn_scores + geo_bias.unsqueeze(0)

        # Softmax over source nodes, dropout
        attn_weights = F.softmax(attn_scores, dim=2)   # (B, N, N, H)
        attn_weights = self.dropout(attn_weights)

        # Weighted aggregation
        attn_w = attn_weights.permute(0, 3, 1, 2)     # (B, H, N_dst, N_src)
        Wh_p   = Wh_val.permute(0, 2, 1, 3)           # (B, H, N_src, D)
        out = torch.matmul(attn_w, Wh_p)              # (B, H, N_dst, D)

        # Edge-conditioned value addition: out_i += Σ_j α_ij * W_edge(adj_ij)
        # Conditions the aggregated message on the edge scalar (wind+distance weight).
        # W_edge is zero-init so this term starts at 0, identical to baseline.
        if self.W_edge is not None:
            # adj_b: (B, N_dst, N_src) — scalar edge weight per pair
            edge_vals = self.W_edge(adj_b.unsqueeze(-1))          # (B, N_dst, N_src, out_features)
            edge_vals = edge_vals.view(
                adj_b.shape[0], N, N, self.num_heads, self.head_dim
            )                                                       # (B, N_dst, N_src, H, D)
            edge_vals_p = edge_vals.permute(0, 3, 1, 2, 4)        # (B, H, N_dst, N_src, D)
            edge_out = (attn_w.unsqueeze(-1) * edge_vals_p).sum(dim=3)  # (B, H, N_dst, D)
            out = out + edge_out

        out = out.permute(0, 2, 1, 3).reshape(B, N, self.out_features)
        out = out + self.bias

        assert out.shape == (B, N, self.out_features), (
            f"GAT output shape error: expected ({B}, {N}, {self.out_features}), got {tuple(out.shape)}"
        )
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism for temporal context modeling.
    
    Allows the decoder to attend to different aspects of the encoder outputs,
    capturing multiple temporal views for improved forecasting.
    
    For each node independently, attends over the temporal sequence.
    """
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Decoder hidden state (batch, num_nodes, hidden_dim)
            key: Encoder outputs (batch, seq_len, num_nodes, hidden_dim)
            value: Encoder outputs (batch, seq_len, num_nodes, hidden_dim)
            mask: Optional attention mask
        Returns:
            context: Attended context (batch, num_nodes, hidden_dim)
            attn_weights: Attention weights (batch, num_nodes, num_heads, seq_len)
        """
        batch_size = query.size(0)
        num_nodes = query.size(1)
        seq_len = key.size(1)
        
        # Reshape key and value: (batch, seq_len, num_nodes, hidden_dim) -> (batch, num_nodes, seq_len, hidden_dim)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        
        # Process each node's attention independently
        # Reshape to combine batch and nodes: (batch * num_nodes, ...)
        query_flat = query.reshape(batch_size * num_nodes, self.hidden_dim)  # (B*N, H)
        key_flat = key.reshape(batch_size * num_nodes, seq_len, self.hidden_dim)  # (B*N, T, H)
        value_flat = value.reshape(batch_size * num_nodes, seq_len, self.hidden_dim)  # (B*N, T, H)
        
        # Project Q, K, V
        Q = self.q_proj(query_flat)  # (B*N, H)
        K = self.k_proj(key_flat)    # (B*N, T, H)
        V = self.v_proj(value_flat)  # (B*N, T, H)
        
        # Reshape for multi-head attention
        # Q: (B*N, num_heads, head_dim)
        Q = Q.view(batch_size * num_nodes, self.num_heads, self.head_dim)
        # K, V: (B*N, T, num_heads, head_dim) -> (B*N, num_heads, T, head_dim)
        K = K.view(batch_size * num_nodes, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size * num_nodes, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        # Q: (B*N, num_heads, head_dim) -> (B*N, num_heads, 1, head_dim)
        Q = Q.unsqueeze(2)
        # Q @ K^T: (B*N, num_heads, 1, head_dim) @ (B*N, num_heads, head_dim, T) -> (B*N, num_heads, 1, T)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B*N, num_heads, 1, T)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (B*N, num_heads, 1, T) @ (B*N, num_heads, T, head_dim) -> (B*N, num_heads, 1, head_dim)
        context = torch.matmul(attn_weights, V)
        
        # Reshape back: (B*N, num_heads, 1, head_dim) -> (B*N, hidden_dim)
        context = context.squeeze(2).reshape(batch_size * num_nodes, self.hidden_dim)
        
        # Output projection
        context = self.out_proj(context)
        
        # Reshape to (batch, num_nodes, hidden_dim)
        context = context.view(batch_size, num_nodes, self.hidden_dim)
        
        # Reshape attention weights for output: (B*N, num_heads, 1, T) -> (batch, num_nodes, num_heads, T)
        attn_weights = attn_weights.squeeze(2).view(batch_size, num_nodes, self.num_heads, seq_len)
        
        return context, attn_weights


class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal sequences.
    Adds positional information to help the model understand temporal order.
    """
    
    def __init__(self, hidden_dim, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_nodes, hidden_dim)
        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        # Add positional encoding: broadcast over batch and nodes
        x = x + self.pe[:seq_len].unsqueeze(0).unsqueeze(2)
        return self.dropout(x)
