import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from tokenizer import Tokenizer
# Swish-Gated Linear Unit from "GLU Variants Improve Transformer" paper
class SwiGLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x) + x

# "Root Mean Square Layer Normalization" paper
class RMSNorm(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.eps = config['eps']
        self.scale = nn.Parameter(torch.ones(config['d_model']))
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.scale


# Relative Positional Encoding from "Self-Attention with Relative Position Representations" paper
class RelativePositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.head_dim = self.d_model // self.n_heads # for one head
        self.max_len = config['n_positions']
        self.rel_emb = nn.Parameter(torch.randn(self.max_len * 2 - 1, self.head_dim))

    def forward(self, length):
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        range_mat = range_mat + self.max_len - 1
        return self.rel_emb[range_mat]



# Multi-Head Attention from "Attention is All You Need" paper
# using relative positional encoding
class MultiHeadAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_queries = config['n_queries']
        self.n_keys = config['n_keys']
        self.n_values = config['n_values']

        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.head_dim = self.d_model // self.n_heads # d_k = d_v = d_q = d_model // n_heads

        assert self.n_keys % self.n_heads == 0, "n_keys must be divisible by n_heads"
        assert self.n_queries % self.n_heads == 0, "n_queries must be divisible by n_heads"
        assert self.n_values % self.n_heads == 0, "n_values must be divisible by n_heads"

        self.query = nn.Linear(self.d_model, self.d_model) 
        self.key = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)

        self.relative_positional_encoding = RelativePositionalEncoding(config)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # we project the queries, keys, and values into n_heads
        k = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        relative_positions = self.relative_positional_encoding(seq_len)
        qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        rel_scores = torch.einsum('bhld,lrd->bhlr', q, relative_positions)
        qk += rel_scores

        # att = F.softmax(qk / math.sqrt(self.n_keys), dim=-1)
        att = F.softmax(qk, dim=-1)
        att_output = torch.matmul(att, v)
        
        att_output = att_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        x = self.out(att_output)
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
        self.feed_forward = FeedForward(config)
        self.rms_norm = RMSNorm(config)
    
    def forward(self, x):
        x_norm = self.rms_norm(x) # first normalization
        x = self.multi_head_attention(x_norm) + x
        x_norm = self.rms_norm(x) # second normalization
        x = self.feed_forward(x_norm) + x
        return x
    

    
class LanguageModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.d_model = config['d_model']
        self.vocab_size = config['vocab_size']
        self.n_layers = config['n_layers']
        self.seq_len = config['n_positions']

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.rms_norm = RMSNorm(config)
        self.out = nn.Linear(self.d_model, self.vocab_size)
        self.blocks = nn.ModuleList()

        for _ in range(self.n_layers):
            self.blocks.append(TransformerBlock(config))
    
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.rms_norm(x)
        x = self.out(x)
        return x

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, do_sample=False, top_k=None):
        generated_tokens = []

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.seq_len else idx[:, -self.seq_len:]
            logits = self(idx_cond)
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

    
    num_unique_tokens = torch.load("num_unique_tokens.pt")
    tensor_text = torch.load("tensor_text.pt")
    tokenizer = Tokenizer("m.model")

    test_tokens = tensor_text[0, :1]
    tensor_test_tokens = test_tokens.clone().detach().unsqueeze(0)

    # print("tensor_test_tokens", tensor_test_tokens)
    print("test_tokens", test_tokens)
    # print("tensor_text.shape", tensor_text.shape)
    print("tensor_test_tokens.shape", tensor_test_tokens.shape)

   
    config = {
        "d_model": 512, # embedding size, number of features in the input, output
        "eps": 1e-6, # epsilon value for normalization
        "n_queries": 128, # = length of the input sequence
        "n_keys": 128,
        "n_values": 128,
        "n_heads": 8,
        "n_layers": 6,
        "n_positions": 1000,
        "vocab_size": num_unique_tokens
        # d_k, d_v, d_q = d_model // n_heads
    }

    model = LanguageModel(config)
    model.eval()


    generated_text = model.generate(tensor_test_tokens)
    print(generated_text)
    print(tokenizer.decode(generated_text.tolist()))