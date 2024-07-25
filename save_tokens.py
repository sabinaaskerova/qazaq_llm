import torch
from tokenizer import Tokenizer

data_path = "data/"

def data_generator(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield line.strip()

tokenizer_path = "tokenizer/"
tokenizer = Tokenizer(tokenizer_path+"m.model")

tokenized_text = []

for text in data_generator(data_path+'kazakh_corpus.txt'):
     tokenized = tokenizer.encode(text)
     tokenized_text.extend(tokenized)


unique_tokens = set(tokenized_text)
num_unique_tokens = len(unique_tokens)
print(num_unique_tokens)

tensor_text = torch.tensor(tokenized).unsqueeze(0)
print("tensor_text.shape", tensor_text.shape)
torch.save(tensor_text, tokenizer_path+"tensor_text.pt")
torch.save(num_unique_tokens, tokenizer_path+"num_unique_tokens.pt")
