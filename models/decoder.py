"""
Decoder module for GCN-LSTM model.
Graph LSTM layers enhanced with multi-head attention for sequence prediction.
"""

import torch
import torch.nn as nn
from .layers import GraphLSTMCell, MultiHeadAttention


class GraphLSTMDecoder(nn.Module):
    """
    Decoder with Graph LSTM layers and Multi-Head Attention.
    
    At each prediction step:
    1. Apply multi-head attention over encoder outputs (temporal context)
    2. Combine attention context with previous prediction
    3. Process through Graph LSTM layers (spatio-temporal modeling)
    4. Project to output dimension (PM2.5 prediction)
    
    The attention mechanism allows the decoder to focus on different
    temporal aspects of the encoder representations for better forecasting.
    """
    
    def __init__(
        self,
        output_dim,
        hidden_dim,
        num_nodes,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        
        # Input projection (previous prediction -> hidden_dim)
        self.input_proj = nn.Linear(output_dim, hidden_dim)
        
        # Multi-head attention for temporal context
        self.attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Attention context projection
        self.context_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Stacked Graph LSTM layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                GraphLSTMCell(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_nodes=num_nodes
                )
            )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection (hidden_dim -> output_dim, e.g., PM2.5)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, encoder_outputs, hidden_states, adj, target=None, horizon=6, teacher_forcing_ratio=0.5):
        """
        Args:
            encoder_outputs: Encoder outputs for attention (batch, enc_seq_len, num_nodes, hidden_dim)
            hidden_states: Initial hidden states from encoder - list of (h, c) per layer
            adj: Adjacency matrix
                 - Static: (num_nodes, num_nodes)
                 - Dynamic: (batch, num_nodes, num_nodes)
            target: Ground truth for teacher forcing (batch, horizon, num_nodes) - optional
            horizon: Number of prediction steps
            teacher_forcing_ratio: Probability of using teacher forcing
        Returns:
            predictions: Model predictions (batch, horizon, num_nodes, output_dim)
            attention_weights: Attention weights for visualization (batch, horizon, num_heads, num_nodes, enc_seq_len)
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        # Initialize first decoder input with zeros (or last encoder hidden state)
        # Using last hidden state of top encoder layer
        decoder_input = hidden_states[-1][0]  # (batch, num_nodes, hidden_dim)

        # Determine teacher forcing per sample (consistent across all timesteps for each sample)
        # This provides more stable training than per-timestep randomness
        if target is not None and teacher_forcing_ratio > 0:
            use_tf_mask = torch.rand(batch_size, device=device) < teacher_forcing_ratio  # (batch,)
        else:
            use_tf_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        predictions = []
        attention_weights_all = []

        for t in range(horizon):
            # Multi-head attention over encoder outputs
            # Query: current decoder hidden state
            # Key/Value: all encoder outputs
            context, attn_weights = self.attention(
                query=decoder_input,
                key=encoder_outputs,
                value=encoder_outputs
            )
            attention_weights_all.append(attn_weights)

            # Combine decoder input with attention context
            combined = torch.cat([decoder_input, context], dim=-1)  # (batch, num_nodes, hidden_dim*2)
            combined = self.context_proj(combined)  # (batch, num_nodes, hidden_dim)
            combined = self.dropout(combined)

            # Pass through Graph LSTM layers with Pre-LN pattern
            layer_input = combined
            for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
                h, c = hidden_states[i]

                # Pre-LN: normalize BEFORE the layer
                layer_input_norm = norm(layer_input)

                # Graph LSTM cell forward
                h_new, c_new = layer(layer_input_norm, (h, c), adj)

                # Residual connection (skip first layer)
                if i > 0:
                    h_new = h_new + layer_input

                # Dropout after residual
                h_new = self.dropout(h_new)

                # Update hidden state
                hidden_states[i] = (h_new, c_new)

                # Output of this layer is input to next layer
                layer_input = h_new

            # Project to output dimension
            output = self.output_proj(layer_input)  # (batch, num_nodes, output_dim)
            predictions.append(output)

            # Prepare next decoder input with per-sample teacher forcing
            if target is not None and t < horizon - 1:
                # Teacher forcing input: use ground truth
                tf_input = target[:, t, :].unsqueeze(-1)  # (batch, num_nodes, 1)
                tf_input = self.input_proj(tf_input)  # (batch, num_nodes, hidden_dim)

                # Autoregressive input: use own prediction
                ar_input = self.input_proj(output)  # (batch, num_nodes, hidden_dim)

                # Select based on per-sample mask
                # use_tf_mask: (batch,) -> (batch, 1, 1) for broadcasting
                mask = use_tf_mask.view(-1, 1, 1)
                decoder_input = torch.where(mask, tf_input, ar_input)
            else:
                # Pure autoregressive: use own prediction
                decoder_input = self.input_proj(output)

        # Stack predictions: (batch, horizon, num_nodes, output_dim)
        predictions = torch.stack(predictions, dim=1)

        # Stack attention weights: (batch, horizon, num_heads, num_nodes, enc_seq_len)
        attention_weights_all = torch.stack(attention_weights_all, dim=1)

        return predictions, attention_weights_all
    
    def inference(self, encoder_outputs, hidden_states, adj, horizon=6):
        """
        Inference mode without teacher forcing.
        """
        return self.forward(
            encoder_outputs=encoder_outputs,
            hidden_states=hidden_states,
            adj=adj,
            target=None,
            horizon=horizon,
            teacher_forcing_ratio=0.0
        )


class DirectMultiHorizonDecoder(nn.Module):
    """
    Direct multi-horizon decoder: predicts all future steps jointly.

    Each horizon step t has a learnable query embedding. For each step:
    1. Combine the step-specific query with the final encoder hidden state.
    2. Attend over the full encoder output sequence (temporal context).
    3. Process the attended context through Graph-LSTM layers, initialized
       from the encoder's final hidden states.
    4. Project to scalar PM2.5 per node.

    Steps are fully independent of each other — there is no autoregressive
    feedback from step t-1 to step t. This eliminates error accumulation
    and is the architectural fix for long-horizon degradation.
    """

    def __init__(
        self,
        output_dim,
        hidden_dim,
        num_nodes,
        horizon,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        max_horizon=24,  # Maximum supported horizon for flexibility
        use_attention=True
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.max_horizon = max(horizon, max_horizon)
        self.num_layers = num_layers
        self.use_attention = use_attention

        # Learnable step query embeddings: support up to max_horizon steps
        # This allows flexible horizon at inference time
        self.step_queries = nn.Parameter(torch.empty(self.max_horizon, hidden_dim))
        nn.init.xavier_uniform_(self.step_queries)

        if use_attention:
            # Multi-head attention over encoder outputs (shared across horizon steps)
            self.attention = MultiHeadAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            # Project [query || context] -> hidden_dim
            self.context_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            # No attention: project query directly to hidden_dim
            self.context_proj = nn.Linear(hidden_dim, hidden_dim)

        # Shared Graph-LSTM layers applied independently per horizon step
        self.layers = nn.ModuleList([
            GraphLSTMCell(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_nodes=num_nodes
            )
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Scalar output per node
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        encoder_outputs,
        hidden_states,
        adj,
        target=None,
        horizon=None,
        teacher_forcing_ratio=0.0  # ignored — no autoregression
    ):
        """
        Args:
            encoder_outputs: (batch, enc_seq_len, num_nodes, hidden_dim)
            hidden_states:   list of (h, c) tuples per encoder layer
            adj:             adjacency matrix — static (N,N) or dynamic (B,N,N)
            target:          unused; kept for API compatibility with autoregressive decoder
            horizon:         overrides self.horizon if given (must be <= max_horizon)
            teacher_forcing_ratio: ignored
        Returns:
            predictions:      (batch, horizon, num_nodes, output_dim)
            attention_weights:(batch, horizon, num_heads, num_nodes, enc_seq_len)
        """
        if horizon is None:
            horizon = self.horizon

        # Validate horizon is within supported range
        if horizon > self.max_horizon:
            raise ValueError(
                f"Requested horizon {horizon} exceeds max_horizon {self.max_horizon}. "
                f"Reinitialize decoder with larger max_horizon."
            )

        batch_size = encoder_outputs.size(0)

        # Final encoder hidden state (top layer) used as base query signal
        final_h = hidden_states[-1][0]  # (batch, num_nodes, hidden_dim)

        predictions = []
        attention_weights_all = []

        for t in range(horizon):
            # Step-specific learned query, broadcast over batch and nodes
            step_q = self.step_queries[t].view(1, 1, -1).expand(
                batch_size, self.num_nodes, -1
            )  # (batch, num_nodes, hidden_dim)

            # Combine learnable step intent with encoder's final hidden state
            query = step_q + final_h  # (batch, num_nodes, hidden_dim)

            if self.use_attention:
                # Attend over the full encoder sequence
                context, attn_weights = self.attention(
                    query=query,
                    key=encoder_outputs,
                    value=encoder_outputs
                )
                attention_weights_all.append(attn_weights)
                # Fuse query and context
                combined = torch.cat([query, context], dim=-1)  # (batch, num_nodes, hidden_dim*2)
                combined = self.context_proj(combined)           # (batch, num_nodes, hidden_dim)
            else:
                # No attention: project query directly
                attention_weights_all.append(torch.zeros(
                    batch_size, self.num_nodes, 1, 1, device=encoder_outputs.device
                ))
                combined = self.context_proj(query)  # (batch, num_nodes, hidden_dim)
            combined = self.dropout(combined)

            # Process through Graph-LSTM layers with Pre-LN pattern.
            # Each horizon step starts from the same encoder hidden states —
            # there is no recurrent dependency between forecast steps.
            # BUT within each step, layers must build on each other's states.
            layer_input = combined
            step_hidden_states = [(h.clone(), c.clone()) for h, c in hidden_states]  # Copy for this step

            for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
                h, c = step_hidden_states[i]

                # Pre-LN: normalize BEFORE the layer
                layer_input_norm = norm(layer_input)

                h_new, c_new = layer(layer_input_norm, (h, c), adj)

                # Residual connection (upper layers only)
                if i > 0:
                    h_new = h_new + layer_input

                h_new = self.dropout(h_new)

                # Update hidden state so next layer builds on this layer's output
                step_hidden_states[i] = (h_new, c_new)

                layer_input = h_new

            output = self.output_proj(layer_input)  # (batch, num_nodes, output_dim)
            predictions.append(output)

        # Stack to (batch, horizon, num_nodes, output_dim)
        predictions = torch.stack(predictions, dim=1)
        attention_weights_all = torch.stack(attention_weights_all, dim=1)

        return predictions, attention_weights_all

    def inference(self, encoder_outputs, hidden_states, adj, horizon=6):
        """Inference mode (identical to forward — no teacher forcing to disable)."""
        return self.forward(
            encoder_outputs=encoder_outputs,
            hidden_states=hidden_states,
            adj=adj,
            horizon=horizon
        )
