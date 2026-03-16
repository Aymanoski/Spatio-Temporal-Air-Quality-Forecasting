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
            
            # Pass through Graph LSTM layers
            layer_input = combined
            for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
                h, c = hidden_states[i]
                
                # Graph LSTM cell forward
                h_new, c_new = layer(layer_input, (h, c), adj)
                
                # Residual connection
                if i > 0:
                    h_new = h_new + layer_input
                
                # Layer normalization
                h_new = norm(h_new)
                
                # Dropout
                h_new = self.dropout(h_new)
                
                # Update hidden state
                hidden_states[i] = (h_new, c_new)
                
                # Output of this layer is input to next layer
                layer_input = h_new
            
            # Project to output dimension
            output = self.output_proj(layer_input)  # (batch, num_nodes, output_dim)
            predictions.append(output)
            
            # Prepare next decoder input
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth
                # target: (batch, horizon, num_nodes) -> need to project to hidden_dim
                next_input = target[:, t, :].unsqueeze(-1)  # (batch, num_nodes, 1)
                decoder_input = self.input_proj(next_input)  # (batch, num_nodes, hidden_dim)
            else:
                # Autoregressive: use own prediction
                decoder_input = self.input_proj(output)  # (batch, num_nodes, hidden_dim)
        
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
