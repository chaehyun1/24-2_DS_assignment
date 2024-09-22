import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Tuple

class QueryLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(QueryLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class KeyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(KeyLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ValueLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(ValueLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = q.size(-1) # queries and keys of dimension dk
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) # .transpose(-2, -1): seq_len과 d_k 차원을 서로 교환
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # mask가 0인 부분에 -1e9를 넣어줌 (softmax에서 0에 가까운 값이 나오도록)
        
        attention = F.softmax(scores, dim=-1)
        result = torch.matmul(attention, v)
        return result, attention
        
        

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.query_layers = QueryLayer(d_model, n_heads)
        self.key_layers = KeyLayer(d_model, n_heads)
        self.value_layers = ValueLayer(d_model, n_heads)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads * d_model, d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = Q.size(0) # Q, K, V의 batch_size는 같다. 
        
        Q = self.query_layers(Q).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        K = self.key_layers(K).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        V = self.value_layers(V).view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        
        x, attention = self.attention(Q, K, V, mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_model)
        
        x = self.fc(x)
        return x