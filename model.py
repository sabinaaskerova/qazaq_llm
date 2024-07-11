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
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.scale


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
        v = v.permute(0, 1, 3, 2)
        print(q.shape, k.shape, v.shape)

        att = F.softmax((q @ k) / math.sqrt(self.n_keys), dim=-1)
        print("att", att.shape)
        print("q@k", (q @ k).shape)
        x = (att @ v).permute(0, 2, 1, 3).reshape(x.shape[0], 
                                                  -1, self.n_values * self.n_heads)
        x = self.out(x)
        return x
    
class SimpleAttention(nn.Module):
    # scaled dot product attention
    def __init__(self, config) -> None:
        super().__init__()
        self.d_model = config['d_model'] # d_model = d_k = d_v = d_q in this case
        self.query = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # print(q.shape, k.shape, v.shape)
        # print(k.transpose(1, 2).shape)
        att = F.softmax((q @ k.transpose(1, 2)) / math.sqrt(self.d_model), dim=-1)
        x = att @ v
        x = self.out(x)
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
        # self.simple_attention = SimpleAttention(config)
        self.feed_forward = FeedForward(config)
        self.rms_norm = RMSNorm(config)
    
    def forward(self, x):
        x_norm = self.rms_norm(x) # first normalization
        # x = self.simple_attention(x_norm) + x
        x = self.multi_head_attention(x_norm) + x
        x_norm = self.rms_norm(x) # second normalization
        x = self.feed_forward(x_norm) + x
        return x
    
class RotaryEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        self.rotary_pos = nn.Parameter(torch.randn(self.n_layers, self.n_heads, self.d_model // self.n_heads))
        self.rotary_pos = self.rotary_pos.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x):
        return x + self.rotary_pos

class Transformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_layers = config['n_layers']
        self.d_model = config['d_model']
        self.embedding = nn.Embedding(self.d_model, self.d_model)
        self.blocks = nn.ModuleList()

        for _ in range(self.n_layers):
            self.blocks.append(TransformerBlock(config))

        self.rms_norm = RMSNorm(config)
        self.out = nn.Linear(self.d_model, self.d_model)        
    
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.rms_norm(x)
        x = self.out(x)
        return x

    def generate(self, x, max_new_tokens=50):
        generated_tokens = []

        for _ in range(max_new_tokens):
            logits = self(x)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            x = torch.cat((x, next_token), dim=1)
            generated_tokens.append(next_token)

        return torch.cat(generated_tokens, dim=1)


if __name__ == "__main__":
   
    config = {
        "d_model": 512, # embedding size, number of features in the input, output
        "eps": 1e-6, # epsilon value for normalization
        "n_queries": 512, 
        "n_keys": 512,
        "n_values": 512,
        "n_heads": 8,
        "n_layers": 6
    }

    model = Transformer(config)
    x = torch.randint(0, config['d_model'], (1, 1), dtype=torch.long) 
    # x = torch.randint(0, config['d_model'], (1, 512), dtype=torch.long)
    print("x",x)
    print("x.shape", x.shape)
    generated_tokens = model.generate(x, max_new_tokens=50)
    print(generated_tokens)

