
from model import Transformer, RMSNorm, SwiGLU, SimpleAttention
from tokenizer import Tokenizer
import torch

if __name__ == "__main__":
    config = {
        "d_model": 512, # embedding size, number of features in the input, output
        "eps": 1e-6, # epsilon value for normalization
        "n_queries": 512, 
        "n_keys": 512,
        "n_values": 512,
        "n_heads": 8,
        "n_layers": 6,
        "n_positions": 512
    }
    with open('kazakh_corpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    
    model = Transformer(config)

    model.eval()

    tokenizer = Tokenizer("m.model")
    text = tokenizer.encode(text)
    print(len(text))
    tensor_text = torch.tensor(text).unsqueeze(0)
    print(len(tensor_text))
    print(tensor_text)
    print(tensor_text.shape)
    print(tensor_text[0])

    test_tokens = tensor_text[0, :1]
    tensor_test_tokens = test_tokens.clone().detach().unsqueeze(0)
    print(len(test_tokens))
    print("test_tokens", test_tokens)
    print("tensor_test_tokens", tensor_test_tokens)

    generated_text = model.generate(tensor_test_tokens)
    print(generated_text)
    print(tokenizer.decode(generated_text.tolist()))
    