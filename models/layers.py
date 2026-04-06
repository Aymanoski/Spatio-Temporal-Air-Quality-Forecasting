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
    
    def __init__(self, input_dim, hidden_dim, num_nodes, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # GCN for spatial feature extraction on input
        self.gcn_i = GraphConvolution(input_dim, hidden_dim)
        
        # GCN for spatial feature extraction on hidden state
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
        # No intermediate activation — graph-aggregated features feed directly into gate
        # linear transform, matching the paper's formulation:
        # i_t = σ(W_i(AX_t) + U_i(AH_{t-1}))
        x_gcn = self.gcn_i(x, adj)  # (batch, num_nodes, hidden_dim)
        h_gcn = self.gcn_h(h, adj)  # (batch, num_nodes, hidden_dim)
        
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
