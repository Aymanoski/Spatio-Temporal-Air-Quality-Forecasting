"""
Full GCN-LSTM Encoder-Decoder Model with Multi-Head Attention.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    Input Sequence                       │
    │              (batch, seq_len, nodes, features)          │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │                      ENCODER                            │
    │  ┌──────────────────────────────────────────────────┐   │
    │  │           Graph LSTM Layer 1                     │   │
    │  │     GCN (spatial) + LSTM (temporal)              │   │
    │  └──────────────────────────────────────────────────┘   │
    │                         │                               │
    │  ┌──────────────────────▼───────────────────────────┐   │
    │  │           Graph LSTM Layer 2                     │   │
    │  │     GCN (spatial) + LSTM (temporal)              │   │
    │  └──────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    ┌─────────────────┐             ┌─────────────────┐
    │ Encoder Outputs │             │  Hidden States  │
    │ (for attention) │             │ (decoder init)  │
    └─────────────────┘             └─────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │                      DECODER                            │
    │  ┌──────────────────────────────────────────────────┐   │
    │  │         Multi-Head Attention                     │   │
    │  │   Attends to encoder outputs (temporal context)  │   │
    │  └──────────────────────────────────────────────────┘   │
    │                         │                               │
    │  ┌──────────────────────▼───────────────────────────┐   │
    │  │           Graph LSTM Layer 1                     │   │
    │  └──────────────────────────────────────────────────┘   │
    │                         │                               │
    │  ┌──────────────────────▼───────────────────────────┐   │
    │  │           Graph LSTM Layer 2                     │   │
    │  └──────────────────────────────────────────────────┘   │
    │                         │                               │
    │  ┌──────────────────────▼───────────────────────────┐   │
    │  │           Output Projection                      │   │
    │  │        (hidden_dim -> output_dim)                │   │
    │  └──────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │                   Predictions                           │
    │            (batch, horizon, nodes, output_dim)          │
    └─────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from .encoder import GraphLSTMEncoder
from .decoder import GraphLSTMDecoder, DirectMultiHorizonDecoder


class GCNLSTMModel(nn.Module):
    """
    GCN-LSTM Encoder-Decoder with Multi-Head Attention for Air Quality Forecasting.
    
    This model combines:
    - Graph Convolutional Networks (GCN) for spatial dependencies between monitoring stations
    - Long Short-Term Memory (LSTM) for temporal dynamics
    - Multi-Head Attention for enhanced temporal context in the decoder
    
    Args:
        input_dim: Number of input features per node (e.g., 33 for PM2.5 + meteo + wind)
        hidden_dim: Hidden dimension for Graph LSTM layers
        output_dim: Number of output features per node (e.g., 1 for PM2.5)
        num_nodes: Number of graph nodes (e.g., 12 stations)
        num_layers: Number of Graph LSTM layers in encoder/decoder
        num_heads: Number of attention heads
        dropout: Dropout probability
        horizon: Forecast horizon (required when use_direct_decoding=True)
        use_direct_decoding: If True, use DirectMultiHorizonDecoder instead of the
            autoregressive GraphLSTMDecoder. Eliminates error accumulation toward +6h.
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_nodes,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        horizon=6,
        use_direct_decoding=False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.use_direct_decoding = use_direct_decoding
        
        # Encoder: Stacked Graph LSTM layers
        self.encoder = GraphLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_layers=num_layers,
            dropout=dropout
        )
        
        if use_direct_decoding:
            # Direct multi-horizon decoder: all steps predicted jointly, no autoregression
            self.decoder = DirectMultiHorizonDecoder(
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_nodes=num_nodes,
                horizon=horizon,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_horizon=max(horizon, 24)  # Support up to 24h forecasts
            )
        else:
            # Autoregressive decoder: each step feeds into the next
            self.decoder = GraphLSTMDecoder(
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_nodes=num_nodes,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout
            )
    
    def forward(self, x, adj, target=None, horizon=6, teacher_forcing_ratio=0.5):
        """
        Forward pass through encoder-decoder.

        Args:
            x: Input sequence (batch, seq_len, num_nodes, input_dim)
            adj: Normalized adjacency matrix
                 - Static: (num_nodes, num_nodes)
                 - Dynamic: (batch, num_nodes, num_nodes)
            target: Ground truth for teacher forcing (batch, horizon, num_nodes) - optional
            horizon: Number of prediction steps
            teacher_forcing_ratio: Probability of using teacher forcing during training

        Returns:
            predictions: (batch, horizon, num_nodes, output_dim)
            attention_weights: (batch, horizon, num_heads, num_nodes, seq_len)
        """
        # Encode input sequence
        encoder_outputs, hidden_states = self.encoder(x, adj)
        
        # Decode to predictions
        predictions, attention_weights = self.decoder(
            encoder_outputs=encoder_outputs,
            hidden_states=hidden_states,
            adj=adj,
            target=target,
            horizon=horizon,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        return predictions, attention_weights
    
    def predict(self, x, adj, horizon=6):
        """
        Inference without teacher forcing.

        Args:
            x: Input sequence (batch, seq_len, num_nodes, input_dim)
            adj: Normalized adjacency matrix
                 - Static: (num_nodes, num_nodes)
                 - Dynamic: (batch, num_nodes, num_nodes)
            horizon: Number of prediction steps

        Returns:
            predictions: (batch, horizon, num_nodes, output_dim)
        """
        self.eval()
        with torch.no_grad():
            # Encode
            encoder_outputs, hidden_states = self.encoder(x, adj)
            
            # Decode without teacher forcing
            predictions, _ = self.decoder.inference(
                encoder_outputs=encoder_outputs,
                hidden_states=hidden_states,
                adj=adj,
                horizon=horizon
            )
        
        return predictions.squeeze(-1)  # (batch, horizon, num_nodes)
    
    def get_num_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config):
    """
    Factory function to create model from config dictionary.
    
    Args:
        config: Dictionary with model hyperparameters
    
    Returns:
        Initialized GCNLSTMModel
    """
    return GCNLSTMModel(
        input_dim=config.get('input_dim', 33),
        hidden_dim=config.get('hidden_dim', 64),
        output_dim=config.get('output_dim', 1),
        num_nodes=config.get('num_nodes', 12),
        num_layers=config.get('num_layers', 2),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1),
        horizon=config.get('horizon', 6),
        use_direct_decoding=config.get('use_direct_decoding', False)
    )
