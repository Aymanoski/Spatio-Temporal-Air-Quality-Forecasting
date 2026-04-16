"""
Graph Transformer model for spatio-temporal air quality forecasting.

Architecture:
    Input (B, T=24, N=12, F=33)
    → input projection (Linear F→H)
    → add learnable node identity embeddings (B, T, N, H)
    → GAT spatial aggregation across all T timesteps at once
        Pre-LN → GraphAttentionLayer → residual
        Handles both static (N,N) and dynamic (B,N,N) adjacency
    → reshape (B, T, N, H) → (B*N, T, H)
    → sinusoidal positional encoding over T
    → small Transformer encoder (2 layers, Pre-LN, shared weights across nodes)
    → full sequence retained → (B, N, T, H)
    → cross-attention horizon decoder:
        horizon queries (horizon, H) attend to full T-step sequence via MHA
        Pre-LN on both queries and keys/values; residual on queries
        shared 2-layer MLP → scalar per horizon
    → output (B, horizon, N, output_dim)

Design rationale:
  - Temporal modeling is the dominant bottleneck on this dataset (empirical finding).
  - T=24 makes Transformer attention trivially cheap (24^2=576 per head).
  - GAT is vectorised over T — no per-timestep loop.
  - Shared Transformer weights across N nodes keeps param count low.
  - Cross-attention decoder: each horizon query learns which timesteps to attend to.
    H+1 naturally focuses on recent history; H+6 aggregates longer patterns.
    Directly addresses horizon degradation from the previous single-token bottleneck.
  - Direct decoding: no autoregression, no error accumulation.
  - API-compatible with GCNLSTMModel (same forward / predict / get_wind_alpha signature).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphAttentionLayer


class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for shape (batch, seq_len, hidden_dim)."""

    def __init__(self, hidden_dim: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        return self.dropout(x)


class SpatioTemporalTransformerEncoder(nn.Module):
    """
    Spatio-temporal encoder.

    Steps:
      1. Input projection + node embeddings
      2. GCN: aggregate neighbour information at every timestep simultaneously
         (one batched matmul — no loop over T)
      3. Reshape to (B*N, T, H) and apply small Transformer encoder per node
      4. Return full sequence: (B, N, T, H) for cross-attention decoding

    The GCN uses a raw weight parameter rather than re-using GraphConvolution so
    that the 4-D input (B, T, N, H) can be handled without a timestep loop.
    Supports both static (N, N) and dynamic (B, N, N) adjacency matrices.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_nodes: int,
        num_tf_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = None,
        dropout: float = 0.1,
        use_node_embeddings: bool = True,
        graph_conv: str = 'gcn',
        num_gat_layers: int = 1,
        gat_version: str = 'v1',
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.use_node_embeddings = use_node_embeddings
        self.graph_conv = graph_conv

        if ffn_dim is None:
            ffn_dim = hidden_dim * 2  # compact: 2× hidden, not the usual 4×

        # --- Input projection ---
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # --- Node identity embeddings ---
        if use_node_embeddings:
            self.node_embed = nn.Embedding(num_nodes, hidden_dim)
            nn.init.normal_(self.node_embed.weight, mean=0.0, std=0.01)
        else:
            self.node_embed = None

        # --- Spatial layers ---
        if graph_conv == 'gat':
            # Stack of GAT layers, each with its own Pre-LN and residual.
            # Layer k aggregates k-hop neighbourhood information.
            self.gat_layers = nn.ModuleList([
                GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout, version=gat_version)
                for _ in range(num_gat_layers)
            ])
            self.gat_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)
            ])
        else:
            # GCN: single layer, vectorised over all T at once via raw weight matrix
            self.gcn_norm = nn.LayerNorm(hidden_dim)
            self.gcn_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            self.gcn_bias = nn.Parameter(torch.zeros(hidden_dim))
            nn.init.xavier_uniform_(self.gcn_weight)

        # --- Temporal: Transformer encoder ---
        self.pos_encoding = TemporalPositionalEncoding(hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            norm_first=True,   # Pre-LN: more stable, matches codebase convention
            batch_first=True,  # (batch, seq, dim) convention
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_tf_layers,
            norm=nn.LayerNorm(hidden_dim),     # final normalisation
            enable_nested_tensor=False,        # not supported with norm_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, T, N, F)
            adj: (N, N) static  or  (B, N, N) dynamic
        Returns:
            (B, N, T, H) — full sequence for cross-attention decoding
        """
        B, T, N, _ = x.shape

        # 1. Input projection: (B, T, N, H)
        x = self.input_proj(x)

        # 2. Node embeddings — add learnable station identity
        if self.node_embed is not None:
            node_ids = torch.arange(N, device=x.device)
            emb = self.node_embed(node_ids)             # (N, H)
            x = x + emb.unsqueeze(0).unsqueeze(0)       # broadcast over B, T

        # 3. Spatial aggregation (stacked GAT or single GCN, Pre-LN + residual per layer)
        if self.graph_conv == 'gat':
            # Expand adj once for all T — same graph used at every timestep.
            # (N,N) or (B,N,N) → (B*T, N, N)
            if adj.dim() == 2:
                adj_flat = adj.unsqueeze(0).expand(B * T, -1, -1)
            else:
                adj_flat = adj.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, N)

            for gat_layer, gat_norm in zip(self.gat_layers, self.gat_norms):
                x_norm = gat_norm(x)                                         # (B, T, N, H)
                x_flat = x_norm.reshape(B * T, N, self.hidden_dim)
                gat_out = gat_layer(x_flat, adj_flat).reshape(B, T, N, self.hidden_dim)
                x = x + gat_out                                              # residual
        else:
            # GCN: single layer, vectorised over all T via batched matmul
            x_norm = self.gcn_norm(x)                                        # (B, T, N, H)
            support = torch.matmul(x_norm, self.gcn_weight) + self.gcn_bias  # (B, T, N, H)
            if adj.dim() == 2:
                gcn_out = torch.matmul(adj, support)                         # (B, T, N, H)
            else:
                adj_t = adj.unsqueeze(1).expand(-1, T, -1, -1)
                gcn_out = torch.matmul(adj_t, support)                       # (B, T, N, H)
            gcn_out = F.leaky_relu(gcn_out, negative_slope=0.1)
            x = x + gcn_out                                                  # residual

        # 4. Temporal Transformer — share weights across nodes
        # (B, T, N, H) → permute → (B, N, T, H) → reshape → (B*N, T, H)
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)

        x = self.pos_encoding(x)   # sinusoidal PE over T

        # Full bidirectional attention: no causal mask needed (we have the full history)
        x = self.transformer(x)    # (B*N, T, H)

        # Return full sequence — the cross-attention decoder will decide what to attend to
        # per horizon step, rather than collapsing everything to a single token here.
        return x.reshape(B, N, T, self.hidden_dim)   # (B, N, T, H)


class DirectHorizonHead(nn.Module):
    """
    Direct multi-horizon prediction head.

    Predicts all forecast steps jointly (no autoregression).
    For each step t a learnable query is added to the encoder summary, then a
    small 2-layer MLP maps to the scalar output.

    Vectorised over the horizon dimension for efficiency.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 1,
        horizon: int = 6,
        dropout: float = 0.1,
        max_horizon: int = 24,
    ):
        super().__init__()
        self.horizon = horizon
        self.max_horizon = max(horizon, max_horizon)
        self.hidden_dim = hidden_dim

        # One learnable query per forecast step (up to max_horizon for flexibility)
        self.step_queries = nn.Parameter(torch.empty(self.max_horizon, hidden_dim))
        nn.init.xavier_uniform_(self.step_queries)

        # Pre-LN → Linear → GELU → Dropout → Linear
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, final_h: torch.Tensor, horizon: int = None) -> torch.Tensor:
        """
        Args:
            final_h: (B, N, H)
            horizon: optional override (must be <= max_horizon)
        Returns:
            (B, horizon, N, output_dim)
        """
        if horizon is None:
            horizon = self.horizon
        if horizon > self.max_horizon:
            raise ValueError(
                f"Requested horizon {horizon} > max_horizon {self.max_horizon}. "
                "Re-initialise DirectHorizonHead with a larger max_horizon."
            )

        B, N, H = final_h.shape

        # Vectorised: expand final_h and broadcast step queries
        # final_h: (B, N, H) → (B, N, horizon, H)
        final_h_exp = final_h.unsqueeze(2).expand(-1, -1, horizon, -1)
        # step_queries: (horizon, H) → (1, 1, horizon, H)
        queries = self.step_queries[:horizon].view(1, 1, horizon, H)
        combined = final_h_exp + queries   # (B, N, horizon, H)

        # Flatten batch dims for MLP, then restore
        combined = combined.reshape(B * N * horizon, H)
        combined = self.norm(combined)
        combined = F.gelu(self.fc1(combined))
        combined = self.drop(combined)
        out = self.fc2(combined)           # (B*N*horizon, output_dim)

        out = out.reshape(B, N, horizon, -1)
        out = out.permute(0, 2, 1, 3)     # (B, horizon, N, output_dim)
        return out


class CrossAttentionHorizonHead(nn.Module):
    """
    Cross-attention horizon decoder.

    Each forecast horizon gets a learned query vector that attends to the full
    T-step encoder sequence via multi-head cross-attention. This lets H+1 focus
    on the most recent timesteps and H+6 aggregate longer temporal patterns —
    something a single-token bottleneck cannot do.

    Input:  encoder_seq (B, N, T, H) — full encoder output (all T timesteps)
    Output: (B, horizon, N, output_dim)

    Design:
      - Pre-LN on both queries and encoder sequence before attention (consistent
        with the rest of the codebase).
      - Residual connection from horizon queries through attention output.
      - Shared 2-layer MLP after attention (horizon-specific context already
        captured by the attention weights; shared MLP just projects to output).
      - No masking on keys — encoder sequence is fully observed history.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 1,
        horizon: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_horizon: int = 24,
    ):
        super().__init__()
        self.horizon = horizon
        self.max_horizon = max(horizon, max_horizon)
        self.hidden_dim = hidden_dim

        # One learnable query per forecast step (up to max_horizon)
        self.horizon_queries = nn.Parameter(torch.empty(self.max_horizon, hidden_dim))
        nn.init.xavier_uniform_(self.horizon_queries)

        # Pre-LN applied to queries and encoder sequence before attention
        self.norm_q  = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)

        # Cross-attention: Q = horizon queries, K = V = encoder sequence
        # batch_first=True: all tensors are (batch, seq, dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Shared output MLP: norm → fc1 → GELU → dropout → fc2
        # Shared because attention weights already carry the horizon-specific signal.
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, encoder_seq: torch.Tensor, horizon: int = None) -> torch.Tensor:
        """
        Args:
            encoder_seq: (B, N, T, H) — full encoder sequence
            horizon:     optional override (must be <= max_horizon)
        Returns:
            (B, horizon, N, output_dim)
        """
        if horizon is None:
            horizon = self.horizon
        if horizon > self.max_horizon:
            raise ValueError(
                f"Requested horizon {horizon} > max_horizon {self.max_horizon}. "
                "Re-initialise CrossAttentionHorizonHead with a larger max_horizon."
            )

        B, N, T, H = encoder_seq.shape

        # Flatten nodes into batch dim so attention runs independently per node
        seq = encoder_seq.reshape(B * N, T, H)              # (B*N, T, H)

        # Expand horizon queries over batch*node
        queries = self.horizon_queries[:horizon]             # (horizon, H)
        queries = queries.unsqueeze(0).expand(B * N, -1, -1)  # (B*N, horizon, H)

        # Pre-LN before attention (consistent with Pre-LN convention in this codebase)
        q  = self.norm_q(queries)                           # (B*N, horizon, H)
        kv = self.norm_kv(seq)                              # (B*N, T, H)

        # Cross-attention: each horizon query attends to all T encoder timesteps
        attn_out, _ = self.cross_attn(q, kv, kv)           # (B*N, horizon, H)

        # Residual: add attention output back to original (un-normed) queries
        attn_out = queries + attn_out                       # (B*N, horizon, H)

        # Shared MLP across all horizons
        out = self.norm_out(attn_out)                       # (B*N, horizon, H)
        out = out.reshape(B * N * horizon, H)
        out = F.gelu(self.fc1(out))
        out = self.drop(out)
        out = self.fc2(out)                                 # (B*N*horizon, output_dim)

        out = out.reshape(B, N, horizon, -1)
        out = out.permute(0, 2, 1, 3)                       # (B, horizon, N, output_dim)
        return out


class GraphTransformerModel(nn.Module):
    """
    Lightweight spatio-temporal Transformer for PM2.5 forecasting.

    Compared with GCNLSTMModel this model replaces the recurrent GraphLSTM
    temporal backbone with a small Transformer encoder while preserving:
      - dynamic wind-aware adjacency (plugged directly into the GCN step)
      - learnable alpha gate
      - learnable node identity embeddings
      - direct multi-horizon decoding (no autoregression)

    The interface is identical to GCNLSTMModel so the training script needs only
    a one-line model-type switch.

    Parameter count (hidden=64, 2 tf_layers, 4 heads):
      ~comparable to or slightly below the GCNLSTMModel baseline.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_nodes: int,
        num_tf_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = None,
        dropout: float = 0.1,
        horizon: int = 6,
        use_node_embeddings: bool = True,
        use_learnable_alpha_gate: bool = False,
        initial_wind_alpha: float = 0.6,
        graph_conv: str = 'gcn',
        num_gat_layers: int = 1,
        gat_version: str = 'v1',
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.use_learnable_alpha_gate = use_learnable_alpha_gate

        # Learnable alpha gate for wind/distance mixing in adjacency construction
        if use_learnable_alpha_gate:
            alpha = float(initial_wind_alpha)
            alpha = min(max(alpha, 1e-4), 1.0 - 1e-4)
            self.alpha_logit = nn.Parameter(
                torch.logit(torch.tensor(alpha, dtype=torch.float32))
            )
        else:
            self.register_parameter("alpha_logit", None)

        self.encoder = SpatioTemporalTransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_tf_layers=num_tf_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_node_embeddings=use_node_embeddings,
            graph_conv=graph_conv,
            num_gat_layers=num_gat_layers,
            gat_version=gat_version,
        )

        self.head = CrossAttentionHorizonHead(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            horizon=horizon,
            num_heads=num_heads,
            dropout=dropout,
            max_horizon=max(horizon, 24),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        target=None,
        horizon: int = 6,
        teacher_forcing_ratio: float = 0.0,
    ):
        """
        Args:
            x:   (B, T, N, F)
            adj: (N, N) static  or  (B, N, N) dynamic
            target, teacher_forcing_ratio: ignored — direct decoding only
            horizon: forecast steps
        Returns:
            predictions:     (B, horizon, N, output_dim)
            attention_weights: None  (kept for API compatibility)
        """
        enc_seq = self.encoder(x, adj)              # (B, N, T, H)
        predictions = self.head(enc_seq, horizon)   # (B, horizon, N, output_dim)
        return predictions, None

    def predict(self, x: torch.Tensor, adj: torch.Tensor, horizon: int = 6) -> torch.Tensor:
        """
        Inference without teacher forcing.

        Returns:
            (B, horizon, N)  — squeezed output dimension
        """
        self.eval()
        with torch.no_grad():
            enc_seq = self.encoder(x, adj)              # (B, N, T, H)
            predictions = self.head(enc_seq, horizon)   # (B, horizon, N, output_dim)
        return predictions.squeeze(-1)  # (B, horizon, N)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_wind_alpha(self):
        """Return current alpha in [0, 1], or None if not learnable."""
        if not self.use_learnable_alpha_gate or self.alpha_logit is None:
            return None
        return torch.sigmoid(self.alpha_logit)
