import torch.nn as nn
import torch
import math


class DotProductAttention(nn.Module):
    # https://mkh800.medium.com/%E7%AD%86%E8%A8%98-attention-%E5%8F%8A-transformer-%E6%9E%B6%E6%A7%8B%E7%90%86%E8%A7%A3-c9c5479fdc8a
    def __init__(self, d_k):
        super(DotProductAttention, self).__init__()
        self.scale = d_k ** -0.5

    def forward(self, q, k, v):
        # attention weight
        # mul scale
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weight = attn.softmax(dim= -1)   
        return torch.matmul(weight, v)
#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.dim = dim
        self.d_k = self.dim // self.num_heads

        self.to_qkv = nn.Linear(self.dim, 3 * self.dim, bias= False)
        # output
        self.W_o = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Dropout(p= attn_drop)
        )

        self.attention = DotProductAttention(self.d_k)

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''

        qkv = self.to_qkv(x)
        q, k, v = tuple(qkv.view(3, x.shape[0], self.num_heads, -1, self.d_k))
        context = self.attention(q, k, v)
        concat_content = context.view(x.shape[0], -1, self.dim)
        return self.W_o(concat_content)

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    