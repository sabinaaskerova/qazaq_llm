import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class SwiGLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x) + x

class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = config['eps']
        self.scale = nn.Parameter(torch.ones(config['d_model']))
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + config['eps']) * self.scale


class MultiHeadAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_queries = config['n_queries']
        self.n_keys = config['n_keys']
        self.n_values = config['n_values']
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        

        self.query = nn.Linear(self.d_model, self.n_queries * self.n_heads)
        self.key = nn.Linear(self.d_model, self.n_keys * self.n_heads)
        self.value = nn.Linear(self.d_model, self.n_values * self.n_heads)
        self.out = nn.Linear(self.n_values * self.n_heads, self.d_model)

    def forward(self, x):
        q = self.query(x).view(x.shape[0], -1, self.n_queries, self.n_heads)
        k = self.key(x).view(x.shape[0], -1, self.n_keys, self.n_heads)
        v = self.value(x).view(x.shape[0], -1, self.n_values, self.n_heads)
        print(q.shape, k.shape, v.shape)

        q = q.permute(0, 3, 2, 1)
        k = k.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        att = F.softmax((q @ k) / math.sqrt(self.n_keys), dim=-1)
        x = (att @ v).permute(0, 2, 1, 3).reshape(x.shape[0], 
                                                  -1, self.n_values * self.n_heads)
        print(x.shape)
        x = self.out(x)
        print(x.shape)
        return x

class FeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config['d_model'], 2048)
        self.activation = SwiGLU()
        self.fc2 = nn.Linear(2048, config['d_model'])
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.rms_norm = RMSNorm(config)
    
    def forward(self, x):
        x_norm = self.rms_norm(x) # first normalization
        x = self.multi_head_attention(x_norm) + x
        x_norm = self.rms_norm(x) # second normalization
        x = self.feed_forward(x_norm) + x
        return x
  

if __name__ == "__main__":
   
    config = {
        "d_model": 512, # embedding size, number of features in the input, output
        "eps": 1e-6, # epsilon value for normalization
        "n_queries": 512, 
        "n_keys": 512,
        "n_values": 512,
        "n_heads": 8 
    }
    mblock = TransformerBlock(config)
    x = torch.randn(1, config['d_model'], config['d_model'])
    output = mblock(x)
    print(output)
