import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForwardLayer(d_model, d_ff)
        
        self.normalization1 = LayerNormalization(d_model)
        self.normalization2 = LayerNormalization(d_model)
        self.normalization3 = LayerNormalization(d_model)
        
        self.dropout1 = DropoutLayer(dropout)
        self.dropout2 = DropoutLayer(dropout)
        self.dropout3 = DropoutLayer(dropout)
        
        self.residual1 = ResidualConnection()
        self.residual2 = ResidualConnection()
        self.residual3 = ResidualConnection()
        
    
    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self_attention = self.self_attention(x, x, x, mask=tgt_mask)
        x = self.residual1(x, self.dropout1(self_attention))
        x = self.normalization1(x)
        
        cross_attention = self.cross_attention(x, memory, memory, mask=tgt_mask)
        x = self.residual2(x, self.dropout2(cross_attention))
        x = self.normalization2(x)
        
        ffn = self.feed_forward(x)
        x = self.residual3(x, self.dropout3(ffn))
        x = self.normalization3(x)
        
        return x