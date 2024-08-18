import torch.nn as nn
import torch
import math
from einops import rearrange

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.dim_head = dim // num_heads
        self.heads = num_heads
        # 1 / (self.dim_head) ^ 0.5
        self.scale = self.dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias= False)
        
        # output
        self.W_o = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(p= attn_drop)
        )

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        h = self.heads
        # first linear to qkv
        qkv = self.to_qkv(x)
        # divide to q, k and v
        q, k, v = tuple(qkv.view(3, x.shape[0], h, -1, self.dim_head))
        # scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weight = attn.softmax(dim= -1)   
        context = torch.matmul(weight, v)
        # concat multi-head
        concat_content = context.view(x.shape[0], -1, self.dim_head * h)
        # last linear
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
    