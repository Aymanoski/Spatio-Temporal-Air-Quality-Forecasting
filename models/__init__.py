"""
Models package for spatio-temporal air quality forecasting.

Components:
- layers: GraphConvolution, GraphLSTMCell, MultiHeadAttention
- encoder: GraphLSTMEncoder (stacked Graph LSTM)
- decoder: GraphLSTMDecoder (Graph LSTM + attention)
- model: GCNLSTMModel (full encoder-decoder, recurrent baseline)
- transformer_model: GraphTransformerModel (GCN + Transformer encoder, lightweight)
"""

from .layers import GraphConvolution, GraphLSTMCell, MultiHeadAttention
from .encoder import GraphLSTMEncoder
from .decoder import GraphLSTMDecoder, DirectMultiHorizonDecoder
from .model import GCNLSTMModel, create_model
from .transformer_model import GraphTransformerModel

__all__ = [
    'GraphConvolution',
    'GraphLSTMCell',
    'MultiHeadAttention',
    'GraphLSTMEncoder',
    'GraphLSTMDecoder',
    'DirectMultiHorizonDecoder',
    'GCNLSTMModel',
    'create_model',
    'GraphTransformerModel',
]
