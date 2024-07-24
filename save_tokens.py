import torch
from tokenizer import Tokenizer

data_path = "data/"

with open(data_path+'kazakh_corpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()

tokenizer_path = "tokenizer/"
tokenizer = Tokenizer(tokenizer_path+"m.model")

text = tokenizer.encode(text)
unique_tokens = set(text)
num_unique_tokens = len(unique_tokens)
tensor_text = torch.tensor(text).unsqueeze(0)
print("tensor_text.shape", tensor_text.shape)
torch.save(tensor_text, tokenizer_path+"tensor_text.pt")
torch.save(num_unique_tokens, tokenizer_path+"num_unique_tokens.pt")
