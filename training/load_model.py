import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
import os
import sentencepiece as spm
from project_config.data_config import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer_path = TOKENIZER_PATH
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(tokenizer_path + "m.model")
tensor_text = torch.load(tokenizer_path+"tensor_text.pt", map_location=device) # we will pretrain the model on the same dataset as the tokenizer

# example state_dict
model = torch.load('/content/drive/MyDrive/QazaqLLM/model_checkpoint_epoch1_batch78892.pth')

start_tokens = tensor_text[0][:20].squeeze(0).unsqueeze(0).to(device)
generated_text = model.generate(start_tokens, max_new_tokens=400, temperature=1.0)
print(generated_text)
print(tokenizer.decode(generated_text.tolist()[0]))