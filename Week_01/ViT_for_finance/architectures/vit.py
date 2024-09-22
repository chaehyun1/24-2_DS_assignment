import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from timm.models.layers import trunc_normal_

class EmbeddingLayer(nn.Module):
    def __init__(self, in_channels, embedding_dim, img_size, patch_size): # in_channels는 이미지의 채널 수, embedding_dim은 임베딩 차원 수, img_size는 이미지의 크기, patch_size는 패치의 크기
        super().__init__()
        self.num_tokens = (img_size // patch_size) ** 2
        self.embed_dim = embedding_dim
        
        # patch embedding
        # 이미지 패치를 입력으로 받아, 각 패치를 특정 차원의 벡터로 변환
        # filter size: patch size x patch size, stride: patch size
        self.project = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        
        # class token
        # 모델의 최종 출력에서 입력 데이터의 전체 표현을 나타내기 위해 사용
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim)) 
        
        self.num_tokens += 1
        self.positionsal_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, embedding_dim))
        nn.init.normal_(self.cls_token, std=1e-6) # normal distribution으로 초기화
        trunc_normal_(self.positionsal_embedding, std=.02) # truncated normal distribution으로 초기화
        
    def forward(self, x):
        b, c, h, w = x.shape
        embed = self.project(x)
        z = embed.view(b, self.embed_dim, -1).permute(0, 2, 1)  

        cls_tokens = self.cls_token.expand(b, -1, -1)
        z = torch.cat([cls_tokens, z], dim=1)

        z = z + self.positionsal_embedding
        return z

    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim=192, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x 

#여기까지 Encoder 구현 끝!!


class VisionTransformer(nn.Module):
    def __init__(self, img_size=65, patch_size=8, in_chans=1, num_classes=10, embed_dim=64, depth=8,
                 num_heads=4, mlp_ratio=2., qkv_bias=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU
 
        self.patch_embed = EmbeddingLayer(in_chans, embed_dim, img_size, patch_size)
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        # final norm
        self.norm = norm_layer(embed_dim)

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)[:, 0]
        return x
    
