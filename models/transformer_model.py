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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphAttentionLayer
from utils.graph import STATIONS, haversine


class TransportDelayModule(nn.Module):
    """
    Physics-informed transport delay feature injection.

    For each (target i, source j) pair, computes the wind-driven transport delay:
        tau_ij = dist_ij [km] / (wspm_j [m/s] * 3.6 + eps)   [hours = timestep units]

    Retrieves delayed source features via linear interpolation over the lookback window,
    and aggregates across source stations using the wind-aware adjacency.

    Injection point: Post-Transformer fusion via cross-attention.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_nodes: int,
        wind_speed_idx: int = 10,
        wspm_mean: float = 0.0,
        wspm_scale: float = 1.0,
        max_delay_hours: float = 24.0,
        wind_window: int = 4,
    ):
        super().__init__()
        self.wind_speed_idx = wind_speed_idx
        self.max_delay_hours = float(max_delay_hours)
        self.wind_window = wind_window

        # Scaler params so we can denormalize wspm back to raw m/s for physics
        self.register_buffer('wspm_mean', torch.tensor(float(wspm_mean)))
        self.register_buffer('wspm_scale', torch.tensor(float(wspm_scale)))

        # Precompute pairwise distance matrix (N, N) in km
        station_names = list(STATIONS.keys())
        dist_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist_matrix[i, j] = haversine(
                        STATIONS[station_names[i]], STATIONS[station_names[j]]
                    )
        self.register_buffer('dist_km', torch.from_numpy(dist_matrix))  # (N, N)

    def forward(self, x_raw: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_raw: (B, T, N, F)  raw input before input_proj
            adj:   (B, N, N) or (N, N)  wind-aware adjacency for aggregation weights
        Returns:
            (B, N, F)  delayed feature contribution in raw feature space
        """
        B, T, N, F = x_raw.shape
        device = x_raw.device

        # --- Step 1: Compute tau_ij ---
        # Aggregate wspm over recent window, then denormalize to raw m/s
        wind_w = min(self.wind_window, T)
        wspm_scaled = x_raw[:, -wind_w:, :, self.wind_speed_idx]          # (B, wind_w, N)
        wspm_raw = (
            wspm_scaled.mean(dim=1) * self.wspm_scale + self.wspm_mean
        ).clamp(min=0.0)                                                    # (B, N)

        # tau[b, i, j] = dist[i,j] / (wspm_j [m/s] * 3.6 [→ km/h] + eps)
        speed_kmh = wspm_raw * 3.6 + 1e-3                                  # (B, N), km/h
        # dist_km: (N, N) → unsqueeze batch; speed: (B, N) → insert i-dim
        tau = self.dist_km.unsqueeze(0) / speed_kmh.unsqueeze(1)           # (B, N, N)
        tau = tau.clamp(0.0, self.max_delay_hours)

        # --- Step 2: Linear interpolation over lookback window ---
        # t_query[b,i,j] = fractional index of the delayed timestep for pair (i,j)
        t_query = float(T - 1) - tau                                        # (B, N, N)
        t0 = t_query.floor().long().clamp(0, T - 1)                        # (B, N, N)
        t1 = (t0 + 1).clamp(0, T - 1)                                      # (B, N, N)
        w1 = (t_query - t0.float()).unsqueeze(-1)                           # (B, N, N, 1)

        # x_perm[b, j, t, f] = features of source j at time t
        x_perm = x_raw.permute(0, 2, 1, 3)                                 # (B, N, T, F)

        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, N)
        j_idx = torch.arange(N, device=device).view(1, 1, N).expand(B, N, N)

        X0 = x_perm[b_idx, j_idx, t0]                                      # (B, N, N, F)
        X1 = x_perm[b_idx, j_idx, t1]                                      # (B, N, N, F)
        X_delayed = (1.0 - w1) * X0 + w1 * X1                              # (B, N, N, F)

        # --- Step 3: Aggregate over source stations j with wind adjacency weights ---
        if adj.dim() == 2:
            adj_w = adj.unsqueeze(0).expand(B, -1, -1)
        else:
            adj_w = adj                                                      # (B, N, N)

        # X_delay[b, i, f] = sum_j  adj_w[b,i,j] * X_delayed[b,i,j,f]
        X_delay = torch.einsum('bij,bijf->bif', adj_w, X_delayed)           # (B, N, F)

        return X_delay                                                      # (B, N, F)


class DelayCrossAttentionFusion(nn.Module):
    """
    Post-Transformer cross-attention fusion of temporal summary with delay context.

    Pre-LN style (consistent with the rest of the model):
      - norm_h normalizes the query (temporal summary) before attention
      - norm_z normalizes the projected delay key/value before attention
      - out_proj is zero-initialized → attn_out = 0 at init → returns h_temporal unchanged
        This guarantees the module starts identical to the baseline.

    query  = norm_h(H_last)     — what we want to enrich
    key    = norm_z(delay_proj(Z_raw))  — delay context
    value  = norm_z(delay_proj(Z_raw))

    H_fused = H_last + cross_attn(query, key, value)   [residual, no final norm]
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        # Project raw delayed features (F) to hidden dim (H) for key/value
        self.delay_proj = nn.Linear(input_dim, hidden_dim)

        # Pre-LN: separate norms for query and key/value streams
        self.norm_h = nn.LayerNorm(hidden_dim)
        self.norm_z = nn.LayerNorm(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # Zero-init out_proj: attn_out = 0 at init → H_fused = H_last + 0 = H_last
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(self, h_temporal: torch.Tensor, z_delay_raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_temporal:  (B, N, H) — last-token summary from Transformer encoder
            z_delay_raw: (B, N, F) — aggregated delayed features from TransportDelayModule
        Returns:
            (B, N, H) — fused representation; equals h_temporal at init
        """
        z_proj = self.delay_proj(z_delay_raw)                        # (B, N, H)
        q = self.norm_h(h_temporal)                                   # Pre-LN on query
        kv = self.norm_z(z_proj)                                      # Pre-LN on key/value
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)     # (B, N, H)
        return h_temporal + attn_out                                   # residual, no final norm


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


class TemporalCNN(nn.Module):
    """
    Dilated 1D TCN parallel branch for temporal modeling.

    4 layers with dilations [1, 2, 4, 8] and kernel_size=3 give a receptive field
    of 31 timesteps, covering the full 24h lookback window.

    Input/output: (B*N, T, H) — same shape as Transformer encoder output.
    Used as a parallel branch alongside the Transformer; contributions are
    scaled by a learned additive gate initialized to 0 (starts identical to baseline).
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for dilation in [1, 2, 4, 8]:
            # padding=dilation keeps output length == input length for kernel_size=3
            self.layers.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                          padding=dilation, dilation=dilation)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*N, T, H)
        for conv, norm in zip(self.layers, self.norms):
            x_norm = norm(x)
            # Conv1d expects (B*N, H, T)
            out = F.gelu(conv(x_norm.transpose(1, 2)).transpose(1, 2))
            out = self.drop(out)
            x = x + out  # Pre-LN residual
        return x  # (B*N, T, H)


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
        use_multiscale_temporal: bool = False,
        local_window: int = 6,
        n_local_layers: int = 1,
        use_tcn_branch: bool = False,
        use_edge_features: bool = False,
        use_dual_channel_spatial: bool = False,
        use_pm25_spatial_path: bool = False,
        use_temporal_first: bool = False,
        use_geo_embeddings: bool = False,
        use_transport_delay: bool = False,
        wind_speed_idx: int = 10,
        wspm_mean: float = 0.0,
        wspm_scale: float = 1.0,
        max_delay_hours: float = 24.0,
        delay_wind_window: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.use_node_embeddings = use_node_embeddings
        self.graph_conv = graph_conv
        self.return_full_sequence = return_full_sequence
        self.use_multiscale_temporal = use_multiscale_temporal
        self.use_tcn_branch = use_tcn_branch
        self.use_dual_channel_spatial = use_dual_channel_spatial
        self.use_pm25_spatial_path = use_pm25_spatial_path
        self.use_temporal_first = use_temporal_first
        self.use_geo_embeddings = use_geo_embeddings
        self.use_transport_delay = use_transport_delay

        if ffn_dim is None:
            ffn_dim = hidden_dim * 2  # compact: 2× hidden, not the usual 4×

        # --- Input projection ---
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # --- PM2.5-only spatial projection (split-input path) ---
        # GAT receives only the PM2.5 feature projected to hidden_dim; the Transformer
        # still receives the full 33-feature projection.  This gives the spatial module
        # a cleaner signal (pollution patterns only) while keeping full meteorological
        # context for temporal encoding.
        if use_pm25_spatial_path:
            self.pm25_proj = nn.Linear(1, hidden_dim)
        else:
            self.pm25_proj = None

        # --- Node identity embeddings ---
        if use_node_embeddings:
            self.node_embed = nn.Embedding(num_nodes, hidden_dim)
            nn.init.normal_(self.node_embed.weight, mean=0.0, std=0.01)
        else:
            self.node_embed = None

        # --- Geographic coordinate encoder ---
        # Fourier-encodes (lat, lon) → MLP → (N, hidden_dim), fused additively with node_embed.
        # geo_bias_proj maps normalized pairwise distances (N, N, 1) → (N, N, num_gat_heads),
        # added to GAT attention logits before softmax — static geographic distance prior.
        # Both are zero-initialized so training starts identical to the baseline.
        if use_geo_embeddings:
            station_names = list(STATIONS.keys())
            coords = np.array([STATIONS[s] for s in station_names], dtype=np.float32)  # (N, 2)
            coords_norm = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-8)  # (N, 2)
            num_freqs = 4
            freqs = (2 ** np.arange(num_freqs, dtype=np.float32)) * np.pi  # (4,)
            angles = coords_norm[:, :, None] * freqs[None, None, :]        # (N, 2, 4)
            fourier = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1).reshape(num_nodes, -1)  # (N, 16)
            self.register_buffer('coord_fourier', torch.from_numpy(fourier))

            fourier_dim = fourier.shape[1]
            self.coord_mlp = nn.Sequential(
                nn.Linear(fourier_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.zeros_(self.coord_mlp[2].weight)
            nn.init.zeros_(self.coord_mlp[2].bias)

            dist_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        dist_matrix[i, j] = haversine(STATIONS[station_names[i]], STATIONS[station_names[j]])
            dist_matrix /= dist_matrix.max() + 1e-8
            self.register_buffer('dist_matrix_norm', torch.from_numpy(dist_matrix))

            num_gat_heads = next(h for h in [4, 2, 1] if hidden_dim % h == 0)
            self.geo_bias_proj = nn.Linear(1, num_gat_heads)
            nn.init.zeros_(self.geo_bias_proj.weight)
            nn.init.zeros_(self.geo_bias_proj.bias)
        else:
            self.coord_mlp = None
            self.geo_bias_proj = None

        # --- Spatial layers ---
        if use_dual_channel_spatial:
            # Two independent GAT streams: one for distance, one for wind.
            # Each stream has its own parameters — neither can be suppressed by alpha.
            # Output: h = h_dist + h_wind (additive, shared residual norm per layer).
            assert graph_conv == 'gat', "use_dual_channel_spatial requires graph_conv='gat'"
            self.gat_dist_layers = nn.ModuleList([
                GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout, version=gat_version)
                for _ in range(num_gat_layers)
            ])
            self.gat_wind_layers = nn.ModuleList([
                GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout, version=gat_version)
                for _ in range(num_gat_layers)
            ])
            self.gat_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)
            ])
            self.gat_layers = None  # not used in dual-channel mode
        elif graph_conv == 'gat':
            # Stack of GAT layers, each with its own Pre-LN and residual.
            # Layer k aggregates k-hop neighbourhood information.
            self.gat_layers = nn.ModuleList([
                GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout,
                                    version=gat_version, use_edge_features=use_edge_features)
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

        # --- Optional local attention branch for multi-scale temporal modeling ---
        # Attends over only the last `local_window` timesteps (recent hours) using a
        # lighter 1-layer Transformer. Its last-token output is fused with the global
        # branch via a learned sigmoid gate.  Both branches share the same PE input.
        if use_multiscale_temporal:
            self.local_window = local_window
            local_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                norm_first=True,
                batch_first=True,
            )
            self.local_transformer = nn.TransformerEncoder(
                local_layer,
                num_layers=n_local_layers,
                norm=nn.LayerNorm(hidden_dim),
                enable_nested_tensor=False,
            )
            # Logit of the mixing weight on the local branch.
            # Init=0 → sigmoid(0)=0.5, balanced start.
            self.local_gate_logit = nn.Parameter(torch.zeros(1))
        else:
            self.local_transformer = None
            self.local_gate_logit = None

        # --- TCN parallel branch ---
        # Dilated 1D TCN run in parallel with the Transformer over (B*N, T, H).
        # Fused additively: output = transformer_out + tcn_gate * tcn_out
        # tcn_gate is a scalar parameter initialized to 0.0 so the branch contributes
        # nothing at training start — identical to Transformer-only baseline.
        if use_tcn_branch:
            self.tcn = TemporalCNN(hidden_dim, dropout=dropout)
            self.tcn_gate = nn.Parameter(torch.zeros(1))
        else:
            self.tcn = None
            self.tcn_gate = None

        self.dropout = nn.Dropout(dropout)

        # --- Transport delay module ---
        # Computes tau_ij from wind speed, interpolates delayed features,
        # aggregates with wind adj weights, projects to H. Zero-init → baseline-identical start.
        if use_transport_delay:
            self.transport_delay = TransportDelayModule(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_nodes=num_nodes,
                wind_speed_idx=wind_speed_idx,
                wspm_mean=wspm_mean,
                wspm_scale=wspm_scale,
                max_delay_hours=max_delay_hours,
                wind_window=delay_wind_window,
            )
            self.delay_fusion = DelayCrossAttentionFusion(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.transport_delay = None
            self.delay_fusion = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x:        (B, T, N, F)
            adj:      (N, N) static  or  (B, N, N) dynamic
            adj_wind: (B, N, N) wind-only adjacency — required when use_dual_channel_spatial=True
        Returns:
            return_full_sequence=False: (B, N, H) — last-token summary per node
            return_full_sequence=True:  (B, N, T, H) — full temporal sequence per node
        """
        B, T, N, _ = x.shape

        # Capture raw input before projection for transport delay computation
        x_raw = x  # (B, T, N, F) — read-only reference

        # 1. Input projection: (B, T, N, H)
        # Split-input path: Transformer gets full 33-feature projection;
        # GAT gets a separate PM2.5-only projection so spatial aggregation
        # works on clean pollution-pattern features rather than met-mixed features.
        if self.use_pm25_spatial_path:
            x_spatial = self.pm25_proj(x[:, :, :, 0:1])  # (B, T, N, H) — PM2.5 only
            x = self.input_proj(x)                        # (B, T, N, H) — full features
        else:
            x = self.input_proj(x)                        # (B, T, N, H)
            x_spatial = x                                 # GAT uses same representation

        # 2. Node embeddings — add learnable station identity to Transformer path
        if self.node_embed is not None:
            node_ids = torch.arange(N, device=x.device)
            emb = self.node_embed(node_ids)             # (N, H)
            if self.coord_mlp is not None:
                emb = emb + self.coord_mlp(self.coord_fourier)  # fuse geographic encoding
            x = x + emb.unsqueeze(0).unsqueeze(0)       # broadcast over B, T

        # Precompute static geographic distance bias for GAT attention (None if disabled)
        if self.geo_bias_proj is not None:
            geo_bias = self.geo_bias_proj(self.dist_matrix_norm.unsqueeze(-1))  # (N, N, H)
        else:
            geo_bias = None

        # 3. Spatial aggregation (stacked GAT or single GCN, Pre-LN + residual per layer)
        # Skipped entirely when use_temporal_first=True — spatial runs post-Transformer instead
        # (handled by GraphTransformerModel.post_gat after the encoder returns).
        if self.use_temporal_first:
            pass  # no pre-temporal spatial; Transformer encodes pure per-node features
        elif self.use_dual_channel_spatial:
            # Dual-channel: adj = A_dist, adj_wind = A_wind (both (B, N, N)).
            # Each has its own GAT; outputs are summed before the shared residual.
            adj_wind = kwargs.get('adj_wind', None)
            assert adj_wind is not None, "use_dual_channel_spatial=True requires adj_wind kwarg"

            def _flatten(a):
                if a.dim() == 2:
                    return a.unsqueeze(0).expand(B * T, -1, -1)
                return a.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, N)

            adj_dist_flat = _flatten(adj)
            adj_wind_flat = _flatten(adj_wind)

            for gat_d, gat_w, gat_norm in zip(self.gat_dist_layers, self.gat_wind_layers, self.gat_norms):
                x_norm = gat_norm(x)                                          # (B, T, N, H)
                x_flat = x_norm.reshape(B * T, N, self.hidden_dim)
                h_dist = gat_d(x_flat, adj_dist_flat).reshape(B, T, N, self.hidden_dim)
                h_wind = gat_w(x_flat, adj_wind_flat).reshape(B, T, N, self.hidden_dim)
                x = x + h_dist + h_wind                                       # additive residual

        elif self.graph_conv == 'gat':
            # Flatten adj to (B*T, N, N) so the GAT loop sees one matrix per
            # (sample, timestep) pair, regardless of how adj was constructed:
            #   (N, N)      — static: same graph for all B and T
            #   (B, N, N)   — dynamic: one aggregated graph per sample
            #   (B, T, N, N)— per-timestep: separate graph for each observed hour
            if adj.dim() == 2:
                adj_flat = adj.unsqueeze(0).expand(B * T, -1, -1)
            elif adj.dim() == 3:
                adj_flat = adj.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, N)
            else:  # (B, T, N, N) — per-timestep adjacency
                adj_flat = adj.reshape(B * T, N, N)

            for gat_layer, gat_norm in zip(self.gat_layers, self.gat_norms):
                # Pre-LN applied to the spatial input (PM2.5-only path or full path).
                # Residual is always added onto x (full-feature Transformer input).
                x_spatial_norm = gat_norm(x_spatial)                        # (B, T, N, H)
                x_flat = x_spatial_norm.reshape(B * T, N, self.hidden_dim)
                gat_out = gat_layer(x_flat, adj_flat, geo_bias=geo_bias).reshape(B, T, N, self.hidden_dim)
                x = x + gat_out                                              # residual onto full-feature path
                x_spatial = x_spatial + gat_out                             # keep spatial path consistent
        else:
            # GCN: single layer, vectorised over all T via batched matmul
            x_norm = self.gcn_norm(x)                                        # (B, T, N, H)
            support = torch.matmul(x_norm, self.gcn_weight) + self.gcn_bias  # (B, T, N, H)
            if adj.dim() == 2:
                gcn_out = torch.matmul(adj, support)                         # (B, T, N, H)
            elif adj.dim() == 3:
                adj_t = adj.unsqueeze(1).expand(-1, T, -1, -1)
                gcn_out = torch.matmul(adj_t, support)                       # (B, T, N, H)
            else:  # (B, T, N, N) — per-timestep adjacency
                gcn_out = torch.matmul(adj, support)                         # (B, T, N, H)
            gcn_out = F.leaky_relu(gcn_out, negative_slope=0.1)
            x = x + gcn_out                                                  # residual

        # 4. Temporal Transformer — share weights across nodes
        # (B, T, N, H) → permute → (B, N, T, H) → reshape → (B*N, T, H)
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)

        x = self.pos_encoding(x)   # sinusoidal PE over T

        # Global branch: full bidirectional attention over all T timesteps
        x_global = self.transformer(x)    # (B*N, T, H)

        # TCN parallel branch: dilated convolutions over (B*N, T, H).
        # Additive fusion: x_global + tcn_gate * tcn_out.
        # tcn_gate=0.0 at init → starts identical to Transformer-only baseline.
        if self.tcn is not None:
            x_tcn = self.tcn(x)                                  # (B*N, T, H) — same PE input
            x_global = x_global + self.tcn_gate * x_tcn

        if self.return_full_sequence:
            # Return all T token representations for attention-based heads.
            # Shape: (B, N, T, H)
            return x_global.reshape(B, N, T, self.hidden_dim)

        # Multi-scale: fuse global last-token with local (recent window) last-token.
        # Both branches see the same PE-encoded input; local branch attends over
        # only the last `local_window` timesteps using a lighter 1-layer Transformer.
        if self.use_multiscale_temporal:
            x_local_in = x[:, -self.local_window:, :]       # (B*N, T_local, H)
            x_local = self.local_transformer(x_local_in)    # (B*N, T_local, H)
            gate = torch.sigmoid(self.local_gate_logit)      # scalar in (0, 1)
            last_global = x_global[:, -1, :]                 # (B*N, H)
            last_local = x_local[:, -1, :]                   # (B*N, H)
            x_out = (1 - gate) * last_global + gate * last_local
        else:
            # Last timestep: analogous to an LSTM's final hidden state.
            x_out = x_global[:, -1, :]                       # (B*N, H)
            
        x_out = x_out.reshape(B, N, self.hidden_dim)
        
        # Transport delay injection: Post-Transformer fusion
        # Fuses target-specific delayed source information using cross-attention
        if self.transport_delay is not None:
            z_delay_raw = self.transport_delay(x_raw, adj)  # (B, N, F)
            x_out = self.delay_fusion(x_out, z_delay_raw)   # (B, N, H)

        return x_out


class MeteorologicalForecaster(nn.Module):
    """
    Graph-aware Transformer for predicting future meteorological conditions.

    Predicts 6-step-ahead met features from the 24-step historical met window.
    Designed as a learned replacement for oracle future meteorology in the PM2.5
    pipeline. Pre-trained on (X_met, Z_met) with MSE loss, then frozen during
    PM2.5 model training to generate Z_pred per batch.

    Architecture mirrors GraphTransformerModel's encoder:
      - Linear input projection + learnable node embeddings
      - 1-layer GATv1 spatial aggregation (uses same dynamic wind-aware adjacency)
      - 2-layer Transformer temporal encoder (Pre-LN, shared across nodes)
      - Direct multi-step output head: step_queries + last token → MLP → 21 met features

    Input:  (B, T=24, N=12, met_dim=21) — scaled met features from lookback window
    Output: (B, H=6,  N=12, met_dim=21) — predicted future met (same scale as Z_scaled)

    The 21 met features are: temp, pres, dewp, rain, wspm (scaled continuous, indices 0–4)
    + 16-category wind direction one-hot (indices 5–20). This matches the oracle Z format.
    """

    def __init__(
        self,
        met_dim: int = 21,
        hidden_dim: int = 64,
        num_nodes: int = 12,
        horizon: int = 6,
        num_tf_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = None,
        dropout: float = 0.1,
        gat_version: str = 'v1',
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.num_nodes = num_nodes

        if ffn_dim is None:
            ffn_dim = hidden_dim * 2

        self.input_proj = nn.Linear(met_dim, hidden_dim)

        self.node_embed = nn.Embedding(num_nodes, hidden_dim)
        nn.init.normal_(self.node_embed.weight, mean=0.0, std=0.01)

        # GAT spatial layer (Pre-LN + residual, identical pattern to PM2.5 encoder)
        self.gat_norm  = nn.LayerNorm(hidden_dim)
        self.gat_layer = GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout, version=gat_version)

        # Temporal Transformer
        self.pos_encoding = TemporalPositionalEncoding(hidden_dim, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_tf_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )

        # Direct multi-step output head
        self.step_queries = nn.Parameter(torch.empty(horizon, hidden_dim))
        nn.init.xavier_uniform_(self.step_queries)
        self.fc1  = nn.Linear(hidden_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, met_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_met: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_met: (B, T, N, met_dim) — scaled met features from lookback window
            adj:   (N,N) static or (B,N,N) dynamic adjacency
        Returns:
            (B, horizon, N, met_dim) — predicted future met in same scale as Z_scaled
        """
        B, T, N, _ = x_met.shape

        x = self.input_proj(x_met)                                    # (B, T, N, H)

        node_ids = torch.arange(N, device=x.device)
        x = x + self.node_embed(node_ids).unsqueeze(0).unsqueeze(0)  # broadcast over B, T

        # GAT spatial (Pre-LN + residual)
        if adj.dim() == 2:
            adj_flat = adj.unsqueeze(0).expand(B * T, -1, -1)
        elif adj.dim() == 3:
            adj_flat = adj.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, N)
        else:
            adj_flat = adj.reshape(B * T, N, N)

        x_norm  = self.gat_norm(x)
        x_flat  = x_norm.reshape(B * T, N, self.hidden_dim)
        gat_out = self.gat_layer(x_flat, adj_flat).reshape(B, T, N, self.hidden_dim)
        x = x + gat_out                                               # residual

        # Temporal Transformer (shared weights across nodes)
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        x = self.pos_encoding(x)
        x = self.transformer(x)                                       # (B*N, T, H)
        last = x[:, -1, :]                                            # (B*N, H)

        # Direct multi-step output: (B*N, 1, H) + (1, horizon, H) → (B*N, horizon, H)
        combined = last.unsqueeze(1) + self.step_queries.unsqueeze(0) # (B*N, horizon, H)
        combined = combined.reshape(B * N * self.horizon, self.hidden_dim)
        combined = F.gelu(self.fc1(combined))
        combined = self.drop(combined)
        out = self.fc2(combined)                                       # (B*N*horizon, met_dim)

        out = out.reshape(B, N, self.horizon, -1).permute(0, 2, 1, 3)  # (B, horizon, N, met_dim)
        return out


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
        use_multiscale_temporal: bool = False,
        local_window: int = 6,
        n_local_layers: int = 1,
        use_horizon_residual_weights: bool = False,
        use_learnable_static_adj: bool = False,
        initial_distance_sigma: float = 1800.0,
        use_multitask: bool = False,
        n_aux_targets: int = 5,
        use_station_horizon_bias: bool = False,
        use_regime_conditioning: bool = False,
        use_tcn_branch: bool = False,
        use_edge_features: bool = False,
        use_dual_channel_spatial: bool = False,
        use_probabilistic_output: bool = False,
        use_pm25_spatial_path: bool = False,
        use_temporal_first: bool = False,
        use_geo_embeddings: bool = False,
        use_transport_delay: bool = False,
        wind_speed_idx: int = 10,
        wspm_mean: float = 0.0,
        wspm_scale: float = 1.0,
        max_delay_hours: float = 24.0,
        delay_wind_window: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.use_learnable_alpha_gate = use_learnable_alpha_gate
        self.use_geo_embeddings = use_geo_embeddings
        # temporal-first forces post-temporal GAT on (that IS the spatial step)
        self.use_post_temporal_gat = use_post_temporal_gat or use_temporal_first
        self.use_temporal_attention_head = use_temporal_attention_head
        self.use_t24_residual = use_t24_residual
        self.use_horizon_residual_weights = use_horizon_residual_weights
        self.use_learnable_static_adj = use_learnable_static_adj
        self.use_multitask = use_multitask
        self.use_station_horizon_bias = use_station_horizon_bias
        self.use_probabilistic_output = use_probabilistic_output
        self.use_temporal_first = use_temporal_first

        # Station × horizon output bias (Exp 4): 72 learnable scalars in normalized space.
        # Zero-init so training starts identical to baseline; learns per-(step,node) correction.
        if use_station_horizon_bias:
            self.station_horizon_bias = nn.Parameter(torch.zeros(horizon, num_nodes))
        else:
            self.register_parameter('station_horizon_bias', None)

        # Soft regime conditioning (Exp 6): direct shortcut from last observed PM2.5 → head.
        # Zero-init weight/bias ensures training starts identical to baseline.
        # Skipped when use_temporal_attention_head=True (enc_out has different shape).
        if use_regime_conditioning:
            self.regime_proj = nn.Linear(1, hidden_dim)
            nn.init.zeros_(self.regime_proj.weight)
            nn.init.zeros_(self.regime_proj.bias)
        else:
            self.regime_proj = None

        # Horizon-dependent residual weights: σ(logit_h) scales the persistence prior
        # per horizon step. Initialized to logit(0.95) ≈ 2.94 so initial behavior
        # matches the uniform residual (≈1.0). Learns to decay toward later horizons.
        if use_horizon_residual_weights:
            self.horizon_residual_logits = nn.Parameter(
                torch.full((horizon,), 2.944)  # sigmoid(2.944) ≈ 0.95
            )
        else:
            self.register_parameter("horizon_residual_logits", None)

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

        # Learnable static adjacency: N×N parameter initialized from Gaussian distance decay.
        # Replaces the precomputed A_dist as the (1-alpha) static component.
        # Off-diagonal entries are parameterized as sigmoid(logit); diagonal is fixed to 1.0.
        if use_learnable_static_adj:
            station_names = list(STATIONS.keys())
            n = len(station_names)
            dist_matrix = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dist_matrix[i, j] = haversine(
                            STATIONS[station_names[i]], STATIONS[station_names[j]]
                        )
            dist_decay = np.exp(-dist_matrix ** 2 / initial_distance_sigma)
            np.fill_diagonal(dist_decay, 0.0)  # diagonal handled separately as self-loop
            dist_decay_clipped = np.clip(dist_decay, 0.01, 0.99)
            logit_init = torch.logit(torch.from_numpy(dist_decay_clipped))
            self.static_adj_logits = nn.Parameter(logit_init)
        else:
            self.register_parameter("static_adj_logits", None)

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
            use_multiscale_temporal=use_multiscale_temporal,
            local_window=local_window,
            n_local_layers=n_local_layers,
            use_tcn_branch=use_tcn_branch,
            use_edge_features=use_edge_features,
            use_dual_channel_spatial=use_dual_channel_spatial,
            use_pm25_spatial_path=use_pm25_spatial_path,
            use_temporal_first=use_temporal_first,
            use_geo_embeddings=use_geo_embeddings,
            use_transport_delay=use_transport_delay,
            wind_speed_idx=wind_speed_idx,
            wspm_mean=wspm_mean,
            wspm_scale=wspm_scale,
            max_delay_hours=max_delay_hours,
            delay_wind_window=delay_wind_window,
        )

        # Post-temporal spatial refinement (optional).
        # Applied after the Transformer produces (B, N, H) node summaries.
        # Each node aggregates its neighbours' temporal summaries via the same
        # wind-aware adjacency used in the pre-temporal GAT.
        # Pre-LN + GAT + residual — identical pattern to the encoder's spatial step.
        # Ablation: set use_post_temporal_gat=False to recover the base model.
        if self.use_post_temporal_gat:  # True when use_post_temporal_gat OR use_temporal_first
            self.post_gat_norm = nn.LayerNorm(hidden_dim)
            self.post_gat = GraphAttentionLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                dropout=dropout,
                version=gat_version,
                use_edge_features=use_edge_features,
            )
        else:
            self.post_gat_norm = None
            self.post_gat = None

        # Geographic distance bias for post_gat (active in temporal-first mode).
        # Zero-init so post_gat starts identical to baseline.
        if use_geo_embeddings and self.use_post_temporal_gat:
            station_names = list(STATIONS.keys())
            n = len(station_names)
            dist_matrix = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dist_matrix[i, j] = haversine(STATIONS[station_names[i]], STATIONS[station_names[j]])
            dist_matrix /= dist_matrix.max() + 1e-8
            self.register_buffer('post_dist_matrix_norm', torch.from_numpy(dist_matrix))
            num_post_gat_heads = next(h for h in [4, 2, 1] if hidden_dim % h == 0)
            self.post_geo_bias_proj = nn.Linear(1, num_post_gat_heads)
            nn.init.zeros_(self.post_geo_bias_proj.weight)
            nn.init.zeros_(self.post_geo_bias_proj.bias)
        else:
            self.post_geo_bias_proj = None

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

        # Auxiliary multi-task head: same architecture as main head but predicts
        # n_aux_targets pollutants (PM10, SO2, NO2, CO, O3) jointly.
        # Shares the encoder — backprop through aux_loss regularises shared representations.
        # Active during training only; predict() uses main head (PM2.5) exclusively.
        if use_multitask:
            self.aux_head = DirectHorizonHead(
                hidden_dim=hidden_dim,
                output_dim=n_aux_targets,
                horizon=horizon,
                dropout=dropout,
                max_horizon=max(horizon, 24),
            )
        else:
            self.aux_head = None

        # Optional Gaussian NLL variance head. The main head still predicts the mean.
        # predict() returns the mean only, keeping deterministic evaluation unchanged.
        if use_probabilistic_output:
            if use_temporal_attention_head:
                raise ValueError("use_probabilistic_output is not supported with temporal_attention_head")
            self.logvar_head = DirectHorizonHead(
                hidden_dim=hidden_dim,
                output_dim=1,
                horizon=horizon,
                dropout=dropout,
                max_horizon=max(horizon, 24),
                future_met_dim=future_met_dim,
            )
            nn.init.zeros_(self.logvar_head.fc2.weight)
            nn.init.zeros_(self.logvar_head.fc2.bias)
        else:
            self.logvar_head = None

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        target=None,
        horizon: int = 6,
        teacher_forcing_ratio: float = 0.0,
        future_met: torch.Tensor = None,
        adj_wind: torch.Tensor = None,
    ):
        """
        Args:
            x:          (B, T, N, F)
            adj:        (N, N) static  or  (B, N, N) dynamic
            target, teacher_forcing_ratio: ignored — direct decoding only
            horizon:    forecast steps
            future_met: (B, horizon, N, F_met) or None — oracle future meteorology
            adj_wind:   (B, N, N) wind-only adjacency — required when use_dual_channel_spatial=True
        Returns:
            predictions:     (B, horizon, N, output_dim)
            attention_weights: None  (kept for API compatibility)
        """
        # (B, N, H) when use_temporal_attention_head=False
        # (B, N, T, H) when use_temporal_attention_head=True
        enc_out = self.encoder(x, adj, adj_wind=adj_wind)

        # Post-temporal spatial refinement (only active when temporal_attention_head=False;
        # mutual exclusion is enforced in __init__).
        if self.use_post_temporal_gat:
            h_norm = self.post_gat_norm(enc_out)   # Pre-LN: (B, N, H)
            post_geo_bias = self.post_geo_bias_proj(self.post_dist_matrix_norm.unsqueeze(-1)) if self.post_geo_bias_proj is not None else None
            gat_out = self.post_gat(h_norm, adj, geo_bias=post_geo_bias)    # (B, N, H)
            enc_out = enc_out + gat_out             # residual

        # Soft regime conditioning: inject last-observed PM2.5 directly into enc_out.
        # Zero-init projection means gradient starts at baseline; learns only if useful.
        if self.regime_proj is not None and not self.use_temporal_attention_head:
            pm25_last = x[:, -1, :, 0:1]                       # (B, N, 1) normalized
            enc_out = enc_out + self.regime_proj(pm25_last)     # (B, N, H)

        predictions = self.head(enc_out, horizon, future_met=future_met)  # (B, horizon, N, output_dim)

        # Station × horizon output bias: additive correction in normalized prediction space.
        if self.station_horizon_bias is not None:
            predictions = predictions + self.station_horizon_bias.unsqueeze(0).unsqueeze(-1)  # (1, H, N, 1)

        aux_predictions = self.aux_head(enc_out, horizon) if self.use_multitask else None
        log_vars = self.logvar_head(enc_out, horizon, future_met=future_met) if self.logvar_head is not None else None
        return predictions, None, aux_predictions, log_vars

    def predict(self, x: torch.Tensor, adj: torch.Tensor, horizon: int = 6,
                future_met: torch.Tensor = None,
                adj_wind: torch.Tensor = None) -> torch.Tensor:
        """
        Inference without teacher forcing.

        Args:
            future_met: (B, horizon, N, F_met) or None — oracle future meteorology
            adj_wind:   (B, N, N) wind-only adjacency — required when use_dual_channel_spatial=True
        Returns:
            (B, horizon, N)  — squeezed output dimension
        """
        self.eval()
        with torch.no_grad():
            enc_out = self.encoder(x, adj, adj_wind=adj_wind)
            if self.use_post_temporal_gat:
                h_norm = self.post_gat_norm(enc_out)
                post_geo_bias = self.post_geo_bias_proj(self.post_dist_matrix_norm.unsqueeze(-1)) if self.post_geo_bias_proj is not None else None
                gat_out = self.post_gat(h_norm, adj, geo_bias=post_geo_bias)
                enc_out = enc_out + gat_out
            if self.regime_proj is not None and not self.use_temporal_attention_head:
                pm25_last = x[:, -1, :, 0:1]
                enc_out = enc_out + self.regime_proj(pm25_last)
            predictions = self.head(enc_out, horizon, future_met=future_met)
            if self.station_horizon_bias is not None:
                predictions = predictions + self.station_horizon_bias.unsqueeze(0).unsqueeze(-1)
        return predictions.squeeze(-1)  # (B, horizon, N)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_horizon_residual_weights(self):
        """Return per-horizon persistence weights in [0, 1], shape (H,), or None if disabled."""
        if not self.use_horizon_residual_weights or self.horizon_residual_logits is None:
            return None
        return torch.sigmoid(self.horizon_residual_logits)

    def get_wind_alpha(self):
        """Return current wind/distance mixing alpha in [0, 1], or None if not learnable."""
        if not self.use_learnable_alpha_gate or self.alpha_logit is None:
            return None
        return torch.sigmoid(self.alpha_logit)

    def get_static_adj(self):
        """Return row-normalized learnable static adjacency (N, N), or None if disabled.
        Off-diagonal entries are sigmoid(logits); diagonal is fixed at 1.0 before normalization."""
        if not self.use_learnable_static_adj or self.static_adj_logits is None:
            return None
        N = self.num_nodes
        device = self.static_adj_logits.device
        raw = torch.sigmoid(self.static_adj_logits)          # (N, N), all in (0, 1)
        mask = 1 - torch.eye(N, device=device)
        A = raw * mask + torch.eye(N, device=device)         # fix self-loops to 1.0
        row_sum = A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return A / row_sum                                   # (N, N), row-normalized

    def get_t24_alpha(self):
        """Return current t-24 residual gate in [0, 1], or None if not enabled."""
        if not self.use_t24_residual or self.t24_logit is None:
            return None
        return torch.sigmoid(self.t24_logit)
