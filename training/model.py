import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from tokenization.tokenizer import Tokenizer
from project_config.data_config import *

class SwiGLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x) + x

class RMSNorm(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.eps = config['eps']
        self.scale = nn.Parameter(torch.ones(config['n_embd']))
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.scale

def compute_rotary_matrices(seq_len, head_dim, device:str, theta = 10000):
    assert head_dim % 2 == 0, "head_dim must be divisible by 2"
    theta_series = torch.arange(0, head_dim, 2).float()
    values = 1 / (theta ** (theta_series / head_dim)).to(device)
    positions = torch.arange(seq_len, device=device).float()
    # seq_len x head_dim/2 -> (seq_len, head_dim/2)
    frequencies = torch.outer(positions, values).float()
    # (seq_len, head_dim/2) -> (seq_len, head_dim / 2)
    complex_frequencies = torch.polar(torch.ones_like(frequencies), frequencies)
    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    complex_frequencies = complex_frequencies.unsqueeze(0).unsqueeze(2)
    return complex_frequencies


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config['n_embd']
        self.n_heads = config['n_heads']
        self.head_dim = self.n_embd // self.n_heads # for one head, 1024/8 = 128
        self.max_len = config['max_seq_length']
        self.device = config['device']

    # x is already a vector from division to heads in multi-head attention
    def forward(self, x: torch.Tensor, freqs_complex):
        seq_len = x.size(1)
        complex_frequencies = freqs_complex[:, :seq_len, :, :].to(self.device)
        # (batch_size, seq_len, n_heads, head_dim) -> (seq_len, batch_size, n_heads, head_dim / 2)
        complex_x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # (batch_size, seq_len, n_heads, head_dim / 2) * (1, seq_len, 1, head_dim / 2) -> (batch_size, seq_len, n_heads, head_dim / 2)
        x_rotated = complex_x * complex_frequencies
        # (batch_size, seq_len, n_heads, head_dim / 2) -> (batch_size, seq_len, n_heads, head_dim / 2, 2)
        x_rotated = torch.view_as_real(x_rotated)
        # (batch_size, seq_len, n_heads, head_dim / 2, 2) -> (batch_size, seq_len, n_heads, head_dim)
        x_rotated = x_rotated.reshape(*x.shape)
        return x_rotated.type_as(x).to(self.device)


class Attention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_embd = config['n_embd']
        self.n_kv_heads = config['n_kv_heads'] if 'n_kv_heads' in config else config['n_heads']
        self.n_heads = config['n_heads']
        self.head_dim = self.n_embd // self.n_heads
        self.max_batch_size = config['max_batch_size']
        self.max_seq_len = config['max_seq_length']
        self.device = config['device']

        self.wq = nn.Linear(self.n_embd, self.head_dim * self.n_heads, bias=False)
        self.wk = nn.Linear(self.n_embd, self.head_dim * self.n_kv_heads, bias=False)
        self.wv = nn.Linear(self.n_embd, self.head_dim * self.n_kv_heads, bias=False)
        self.out = nn.Linear(self.head_dim * self.n_heads, self.n_embd)
        self.rotary_positional_encoding = RotaryPositionalEncoding(config)

        self.k_cache = torch.zeros((self.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim))
        self.v_cache = torch.zeros((self.max_batch_size, self.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos, freqs_complex):
        batch_size, seq_len, _ = x.size() 

        # (batch_size, seq_len, n_heads * head_fim) -> (batch_size, seq_len, n_heads, head_dim)
        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim).to(self.device)
        k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).to(self.device)
        v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).to(self.device)

        q = self.rotary_positional_encoding(q, freqs_complex)
        k = self.rotary_positional_encoding(k, freqs_complex)   

        # storing keys and values in cache
        self.k_cache[batch_size:, start_pos:start_pos+seq_len] = k
        self.v_cache[batch_size:, start_pos:start_pos+seq_len] = v

        keys = self.k_cache[:batch_size, : start_pos + seq_len].to(self.device)
        values = self.v_cache[:batch_size, : start_pos + seq_len].to(self.device)

        if self.n_heads // self.n_kv_heads > 1:
            keys = (keys[:, :, :, None, :]
                    .expand(batch_size, seq_len, self.n_kv_heads, self.n_heads//self.n_kv_heads, self.head_dim)
                    .reshape(batch_size, seq_len, self.n_kv_heads * (self.n_heads//self.n_kv_heads, self.head_dim), self.head_dim)
                    .to(self.device))
            values = (values[:, :, :, None, :]
                    .expand(batch_size, seq_len, self.n_kv_heads, self.n_heads//self.n_kv_heads, self.head_dim)
                    .reshape(batch_size, seq_len, self.n_kv_heads * (self.n_heads//self.n_kv_heads, self.head_dim), self.head_dim)
                    .to(self.device))
        
        q = q.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        qk = torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        att = F.softmax(qk, dim=-1)
        att_output = torch.matmul(att, values)
        att_output = att_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        x = self.out(att_output)
        
        return x

class FeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config['n_embd'], 4096)
        self.activation = SwiGLU()
        self.fc2 = nn.Linear(4096, config['n_embd'])
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.rms_norm = RMSNorm(config)
    
    def forward(self, x, start_pos, freqs_complex):
        x_norm = self.rms_norm(x) # first normalization
        x = self.attention(x_norm, start_pos, freqs_complex) + x
        x_norm = self.rms_norm(x) # second normalization
        x = self.feed_forward(x_norm) + x
        return x
    
    
class LanguageModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_embd = config['n_embd']
        self.vocab_size = config['vocab_size']
        self.n_layers = config['n_layers']
        self.max_seq_len = config['max_seq_length']
        self.device = config['device']
        self.n_heads = config['n_heads']
        self.max_batch_size = config['max_batch_size']

        self.embedding = nn.Embedding(self.vocab_size, self.n_embd)
        self.rms_norm = RMSNorm(config)
        self.out = nn.Linear(self.n_embd, self.vocab_size)
        self.blocks = nn.ModuleList()

        for _ in range(self.n_layers):
            self.blocks.append(TransformerBlock(config))

        self.freqs_complex = compute_rotary_matrices(self.max_seq_len * 2, self.n_embd//self.n_heads, device=self.device).to(self.device)

    
    def forward(self, x, start_pos):
        x = self.embedding(x).to(self.device)
        
        for block in self.blocks:
            x = block(x, start_pos, self.freqs_complex)
        x = self.rms_norm(x)
        x = self.out(x)
        return x

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, do_sample=True, top_k=None):
        generated_tokens = []

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:].to(self.device)
            start_pos = idx_cond.size(1) - 1

            logits = self(idx_cond, start_pos)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            idx = torch.cat((idx, idx_next), dim=1)
            generated_tokens.append(idx_next)

        return torch.cat(generated_tokens, dim=1)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer_path = TOKENIZER_PATH
    num_unique_tokens = torch.load(tokenizer_path+"num_unique_tokens.pt", map_location=device)
    tensor_text = torch.load(tokenizer_path+"tensor_text.pt", map_location=device)

    tokenizer = Tokenizer(tokenizer_path+"m.model")

    test_tokens = tensor_text[0, :1]
    tensor_test_tokens = test_tokens.clone().detach().unsqueeze(0).to(device)

    print("test_tokens", test_tokens)
    print("tensor_test_tokens.shape", tensor_test_tokens.shape)

    config = {
        "n_embd":  1024, # embedding size, number of features in the input, output
        "eps": 1e-6, # epsilon value for normalization
        "n_heads": 8, # n_embd should be divisible by n_heads
        "n_layers": 6,
        "max_seq_length": 1000,
        "vocab_size": num_unique_tokens,
        "max_batch_size": 64,
        "device" : device
        # d_k, d_v, d_q = n_embd // n_heads
    }

    model = LanguageModel(config).to(config['device'])
    model.eval()
    print(tensor_test_tokens.size())

    generated_text = model.generate(tensor_test_tokens)
    print(generated_text)
    print(tokenizer.decode(generated_text.tolist()))