"""
Graph Transformer model for spatio-temporal air quality forecasting.

Architecture:
    Input (B, T=24, N=12, F=33)
    → input projection (Linear F→H)
    → add learnable node identity embeddings (B, T, N, H)
    → GCN spatial aggregation across all T timesteps at once
        Pre-LN → (B,T,N,H) @ (H,H) → adj @ support → LeakyReLU → residual
        Handles both static (N,N) and dynamic (B,N,N) adjacency in one matmul
    → reshape (B, T, N, H) → (B*N, T, H)
    → sinusoidal positional encoding over T
    → small Transformer encoder (2 layers, Pre-LN, shared weights across nodes)
    → either:
        (a) last timestep token → (B, N, H)
            → optional post-temporal spatial GAT (Pre-LN + GAT + residual)
            → direct multi-horizon head:
               for each step t: final_h + learned step_query[t] → 2-layer MLP → scalar
        (b) full sequence → (B, N, T, H)   [use_temporal_attention_head=True]
            → horizon-conditioned temporal attention head:
               for each horizon h: softmax attention over T → pooled context_h → MLP → scalar
    → output (B, horizon, N, output_dim)

Design rationale:
  - Temporal modeling is the dominant bottleneck on this dataset (empirical finding).
  - T=24 makes Transformer attention trivially cheap (24^2=576 per head).
  - GCN is vectorised over T — no per-timestep loop.
  - Shared Transformer weights across N nodes keeps param count low.
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
      4. Return last-timestep representation: (B, N, H)

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
        return_full_sequence: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.use_node_embeddings = use_node_embeddings
        self.graph_conv = graph_conv
        self.return_full_sequence = return_full_sequence

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
            return_full_sequence=False: (B, N, H) — last-token summary per node
            return_full_sequence=True:  (B, N, T, H) — full temporal sequence per node
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

        if self.return_full_sequence:
            # Return all T token representations for attention-based heads.
            # Shape: (B, N, T, H)
            return x.reshape(B, N, T, self.hidden_dim)

        # Last timestep: represents the model state after seeing the full 24h window.
        # Forecasting-natural — analogous to an LSTM's final hidden state.
        x = x[:, -1, :]            # (B*N, H)

        # Reshape to (B, N, H)
        return x.reshape(B, N, self.hidden_dim)


class DirectHorizonHead(nn.Module):
    """
    Direct multi-horizon prediction head.

    Predicts all forecast steps jointly (no autoregression).
    For each step t a learnable query is added to the encoder summary, then a
    small 2-layer MLP maps to the scalar output.

    Vectorised over the horizon dimension for efficiency.

    Optional: future_met_dim > 0 enables oracle future meteorology fusion.
    Future met features (shape: B, horizon, N, F_met) are projected to hidden_dim
    and added to the combined representation before the MLP. This gives the decoder
    access to observed wind/met conditions during the forecast window — information
    the encoder cannot see.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 1,
        horizon: int = 6,
        dropout: float = 0.1,
        max_horizon: int = 24,
        future_met_dim: int = 0,
    ):
        super().__init__()
        self.horizon = horizon
        self.max_horizon = max(horizon, max_horizon)
        self.hidden_dim = hidden_dim
        self.future_met_dim = future_met_dim

        # One learnable query per forecast step (up to max_horizon for flexibility)
        self.step_queries = nn.Parameter(torch.empty(self.max_horizon, hidden_dim))
        nn.init.xavier_uniform_(self.step_queries)

        # Optional future meteorology projection: F_met → hidden_dim
        if future_met_dim > 0:
            self.future_met_proj = nn.Linear(future_met_dim, hidden_dim)
        else:
            self.future_met_proj = None

        # Pre-LN → Linear → GELU → Dropout → Linear
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, final_h: torch.Tensor, horizon: int = None,
                future_met: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            final_h:    (B, N, H)
            horizon:    optional override (must be <= max_horizon)
            future_met: (B, horizon, N, F_met) or None
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

        # Future meteorology: project and add to combined representation.
        # future_met: (B, horizon, N, F_met) → permute → (B, N, horizon, F_met) → project → (B, N, horizon, H)
        if self.future_met_proj is not None and future_met is not None:
            fm = future_met.permute(0, 2, 1, 3)          # (B, N, horizon, F_met)
            combined = combined + self.future_met_proj(fm)  # additive, keeps dim=H

        # Flatten batch dims for MLP, then restore
        combined = combined.reshape(B * N * horizon, H)
        combined = self.norm(combined)
        combined = F.gelu(self.fc1(combined))
        combined = self.drop(combined)
        out = self.fc2(combined)           # (B*N*horizon, output_dim)

        out = out.reshape(B, N, horizon, -1)
        out = out.permute(0, 2, 1, 3)     # (B, horizon, N, output_dim)
        return out


class HorizonAttentionHead(nn.Module):
    """
    Horizon-conditioned temporal attention head.

    Instead of using only the last encoder token, each forecast horizon
    learns its own soft attention over all T timesteps. This preserves
    temporal evolution patterns (trend, slope, periodicity) that the last-
    token summary discards — patterns critical for H4-H6 prediction.

    For each horizon h:
      1. Score each timestep: score_t = scorer_h · x_t   (linear H → 1)
      2. Normalize over T:    weights = softmax(scores)
      3. Weighted pool:       context_h = Σ_t weights_t · x_t   (B, N, H)
      4. Predict:             LayerNorm → fc1 → GELU → Dropout → fc2 → scalar

    The MLP (fc1, fc2) is shared across all horizons — only the temporal
    attention scorer is per-horizon. This limits parameter growth while
    keeping horizon-specific temporal focus.

    Why this might work when cross-attention failed:
      - Simpler scoring (linear H→1 vs full QKV projections) → more stable gradients
      - No interaction between horizon queries → independent optimization per step
      - Does NOT compete with persistence residual (residual is added after head output)
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

        # Per-horizon temporal attention scorers: (max_horizon, H)
        # scorer_h · x_t gives the unnormalized attention score for timestep t at horizon h
        self.horizon_scorers = nn.Parameter(torch.empty(self.max_horizon, hidden_dim))
        nn.init.xavier_uniform_(self.horizon_scorers)

        # Shared MLP: same weights applied to each horizon's pooled context
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, horizon: int = None) -> torch.Tensor:
        """
        Args:
            x:       (B, N, T, H) — full encoder sequence per node
            horizon: optional override (must be <= max_horizon)
        Returns:
            (B, horizon, N, output_dim)
        """
        if horizon is None:
            horizon = self.horizon
        if horizon > self.max_horizon:
            raise ValueError(
                f"Requested horizon {horizon} > max_horizon {self.max_horizon}."
            )

        B, N, T, H = x.shape

        # Flatten batch and node dimensions for vectorised attention
        x_flat = x.reshape(B * N, T, H)                           # (B*N, T, H)

        # Compute per-horizon attention scores over all T timesteps.
        # horizon_scorers[:horizon]: (horizon, H) → transpose: (H, horizon)
        # scores: (B*N, T, horizon)
        scores = torch.matmul(x_flat, self.horizon_scorers[:horizon].t())

        # Softmax over the time dimension → attention weights
        weights = F.softmax(scores, dim=1)                         # (B*N, T, horizon)

        # Weighted temporal pooling per horizon:
        # context[b, s, h] = Σ_t weights[b, t, s] * x_flat[b, t, h]
        # einsum: 'bth, bts -> bsh'  (t is summed, s=horizon step, h=feature dim)
        context = torch.einsum('bth,bts->bsh', x_flat, weights)   # (B*N, horizon, H)

        # Shared MLP applied to each (node, horizon) context vector
        context = context.reshape(B * N * horizon, H)
        context = self.norm(context)
        context = F.gelu(self.fc1(context))
        context = self.drop(context)
        out = self.fc2(context)                                    # (B*N*horizon, output_dim)

        out = out.reshape(B, N, horizon, -1)
        out = out.permute(0, 2, 1, 3)                             # (B, horizon, N, output_dim)
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
        use_post_temporal_gat: bool = False,
        use_temporal_attention_head: bool = False,
        use_t24_residual: bool = False,
        initial_t24_alpha: float = 0.3,
        future_met_dim: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.use_learnable_alpha_gate = use_learnable_alpha_gate
        self.use_post_temporal_gat = use_post_temporal_gat
        self.use_temporal_attention_head = use_temporal_attention_head
        self.use_t24_residual = use_t24_residual

        # Learnable gate for t-24 daily anchor.
        # prediction = model_delta + y_last + σ(t24_logit) * y_{t-24}
        # Initialized to initial_t24_alpha so the model starts with a small but
        # nonzero daily anchor and learns the optimal weight end-to-end.
        if use_t24_residual:
            t24_alpha = float(initial_t24_alpha)
            t24_alpha = min(max(t24_alpha, 1e-4), 1.0 - 1e-4)
            self.t24_logit = nn.Parameter(
                torch.logit(torch.tensor(t24_alpha, dtype=torch.float32))
            )
        else:
            self.register_parameter("t24_logit", None)

        if use_temporal_attention_head and use_post_temporal_gat:
            raise ValueError(
                "use_temporal_attention_head and use_post_temporal_gat are mutually exclusive: "
                "post_gat requires (B, N, H) but temporal_attention_head uses (B, N, T, H)."
            )

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
            return_full_sequence=use_temporal_attention_head,
        )

        # Post-temporal spatial refinement (optional).
        # Applied after the Transformer produces (B, N, H) node summaries.
        # Each node aggregates its neighbours' temporal summaries via the same
        # wind-aware adjacency used in the pre-temporal GAT.
        # Pre-LN + GAT + residual — identical pattern to the encoder's spatial step.
        # Ablation: set use_post_temporal_gat=False to recover the base model.
        if use_post_temporal_gat:
            self.post_gat_norm = nn.LayerNorm(hidden_dim)
            self.post_gat = GraphAttentionLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                dropout=dropout,
                version=gat_version,
            )
        else:
            self.post_gat_norm = None
            self.post_gat = None

        if use_temporal_attention_head:
            self.head = HorizonAttentionHead(
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                horizon=horizon,
                dropout=dropout,
                max_horizon=max(horizon, 24),
            )
        else:
            self.head = DirectHorizonHead(
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                horizon=horizon,
                dropout=dropout,
                max_horizon=max(horizon, 24),
                future_met_dim=future_met_dim,
            )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        target=None,
        horizon: int = 6,
        teacher_forcing_ratio: float = 0.0,
        future_met: torch.Tensor = None,
    ):
        """
        Args:
            x:          (B, T, N, F)
            adj:        (N, N) static  or  (B, N, N) dynamic
            target, teacher_forcing_ratio: ignored — direct decoding only
            horizon:    forecast steps
            future_met: (B, horizon, N, F_met) or None — oracle future meteorology
        Returns:
            predictions:     (B, horizon, N, output_dim)
            attention_weights: None  (kept for API compatibility)
        """
        # (B, N, H) when use_temporal_attention_head=False
        # (B, N, T, H) when use_temporal_attention_head=True
        enc_out = self.encoder(x, adj)

        # Post-temporal spatial refinement (only active when temporal_attention_head=False;
        # mutual exclusion is enforced in __init__).
        if self.use_post_temporal_gat:
            h_norm = self.post_gat_norm(enc_out)   # Pre-LN: (B, N, H)
            gat_out = self.post_gat(h_norm, adj)    # (B, N, H)
            enc_out = enc_out + gat_out             # residual

        predictions = self.head(enc_out, horizon, future_met=future_met)  # (B, horizon, N, output_dim)
        return predictions, None

    def predict(self, x: torch.Tensor, adj: torch.Tensor, horizon: int = 6,
                future_met: torch.Tensor = None) -> torch.Tensor:
        """
        Inference without teacher forcing.

        Args:
            future_met: (B, horizon, N, F_met) or None — oracle future meteorology
        Returns:
            (B, horizon, N)  — squeezed output dimension
        """
        self.eval()
        with torch.no_grad():
            enc_out = self.encoder(x, adj)
            if self.use_post_temporal_gat:
                h_norm = self.post_gat_norm(enc_out)
                gat_out = self.post_gat(h_norm, adj)
                enc_out = enc_out + gat_out
            predictions = self.head(enc_out, horizon, future_met=future_met)
        return predictions.squeeze(-1)  # (B, horizon, N)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_wind_alpha(self):
        """Return current wind/distance mixing alpha in [0, 1], or None if not learnable."""
        if not self.use_learnable_alpha_gate or self.alpha_logit is None:
            return None
        return torch.sigmoid(self.alpha_logit)

    def get_t24_alpha(self):
        """Return current t-24 residual gate in [0, 1], or None if not enabled."""
        if not self.use_t24_residual or self.t24_logit is None:
            return None
        return torch.sigmoid(self.t24_logit)
