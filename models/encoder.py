"""
Encoder module for GCN-LSTM model.
Stacked Graph LSTM layers that extract and compress spatio-temporal features.
"""

import torch
import torch.nn as nn
from .layers import GraphLSTMCell, PositionalEncoding


class GraphLSTMEncoder(nn.Module):
    """
    Encoder with stacked Graph LSTM layers.
    
    Processes the input sequence through multiple Graph LSTM layers,
    each capturing spatio-temporal dependencies at different abstraction levels.
    
    Returns:
        - Final hidden states for decoder initialization
        - All encoder outputs for attention mechanism
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_nodes,
        num_layers=2,
        dropout=0.1,
        use_node_embeddings=True,
        graph_conv='gcn',
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.use_node_embeddings = use_node_embeddings
        self.graph_conv = graph_conv
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if use_node_embeddings:
            # Learnable node identity embeddings — injected after Pre-LN inside each
            # LSTM step so LayerNorm cannot re-center them away before the cell sees them.
            self.node_embed = nn.Embedding(num_nodes, hidden_dim)
            nn.init.normal_(self.node_embed.weight, mean=0.0, std=0.01)
        else:
            self.register_parameter("node_embed", None)

        # Positional encoding for temporal awareness
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Stacked Graph LSTM layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                GraphLSTMCell(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_nodes=num_nodes,
                    graph_conv=graph_conv,
                    dropout=dropout,
                )
            )
        
        # Layer normalization for stable training
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        """
        Args:
            x: Input sequence (batch, seq_len, num_nodes, input_dim)
            adj: Adjacency matrix
                 - Static: (num_nodes, num_nodes)
                 - Dynamic: (batch, num_nodes, num_nodes)
        Returns:
            encoder_outputs: All timestep outputs (batch, seq_len, num_nodes, hidden_dim)
            hidden_states: List of (h, c) for each layer - final states for decoder init
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        device = x.device
        
        # Project input to hidden dimension
        x = self.input_proj(x)  # (batch, seq_len, num_nodes, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Initialize hidden states for all layers
        hidden_states = []
        for layer in self.layers:
            hidden_states.append(layer.init_hidden(batch_size, device))
        
        # Store outputs for attention
        encoder_outputs = []
        
        if self.use_node_embeddings:
            # Compute node embeddings once per forward pass — shape (num_nodes, hidden_dim).
            # These are injected after Pre-LN at every layer and timestep so that
            # LayerNorm cannot re-center the station identity signal before the cell sees it.
            node_ids = torch.arange(num_nodes, device=device)
            node_embed = self.node_embed(node_ids)  # (num_nodes, hidden_dim)
        else:
            node_embed = None

        # Process sequence timestep by timestep
        for t in range(seq_len):
            # Current input: (batch, num_nodes, hidden_dim)
            layer_input = x[:, t, :, :]

            # Pass through each Graph LSTM layer with Pre-LN pattern
            for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
                h, c = hidden_states[i]

                # Pre-LN: normalize BEFORE the layer (more stable training)
                layer_input_norm = norm(layer_input)

                if node_embed is not None:
                    # Inject node identity after Pre-LN so normalization cannot
                    # re-center it. Broadcasts over the batch dimension.
                    layer_input_norm = layer_input_norm + node_embed.unsqueeze(0)

                # Graph LSTM cell forward
                h_new, c_new = layer(layer_input_norm, (h, c), adj)

                # Residual connection (skip first layer since input projection changes dims)
                if i > 0:
                    h_new = h_new + layer_input

                # Dropout after residual
                h_new = self.dropout(h_new)

                # Update hidden state
                hidden_states[i] = (h_new, c_new)

                # Output of this layer is input to next layer
                layer_input = h_new

            # Store final layer output for this timestep
            encoder_outputs.append(layer_input)
        
        # Stack outputs: (batch, seq_len, num_nodes, hidden_dim)
        encoder_outputs = torch.stack(encoder_outputs, dim=1)
        
        return encoder_outputs, hidden_states
    
    def init_hidden(self, batch_size, device):
        """Initialize all hidden states."""
        hidden_states = []
        for layer in self.layers:
            hidden_states.append(layer.init_hidden(batch_size, device))
        return hidden_states
