import torch
from tokenizer import Tokenizer



with open('kazakh_corpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()

tokenizer = Tokenizer("m.model")

text = tokenizer.encode(text)
unique_tokens = set(text)
num_unique_tokens = len(unique_tokens)
tensor_text = torch.tensor(text).unsqueeze(0)
print("tensor_text.shape", tensor_text.shape)
torch.save(tensor_text, "tensor_text.pt")
torch.save(num_unique_tokens, "num_unique_tokens.pt")
