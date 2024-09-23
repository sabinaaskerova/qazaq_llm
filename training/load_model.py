import torch
torch.autograd.set_detect_anomaly(True)
import os
import sentencepiece as spm
from project_config.data_config import *
from model import LanguageModel

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer_path = TOKENIZER_PATH
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(tokenizer_path + "m.model")

tensor_text = torch.load(tokenizer_path+"tensor_text.pt", map_location=device) # we will pretrain the model on the same dataset as the tokenizer

batch_size = 32
vocab_size = tokenizer.get_piece_size()
vocab_tokens = [tokenizer.id_to_piece(i) for i in range(vocab_size)]


config = {
    "n_embd": 1024,
    "eps": 1e-6,
    "n_heads": 8,
    "n_layers": 6,
    "max_seq_length": 1000,
    "vocab_size": vocab_size,
    "max_batch_size": batch_size,
    "device": device,
    "multiple": 64
}

model = LanguageModel(config).to(device)

# Load the checkpoint
checkpoints = [f for f in os.listdir(MODEL_STATES_PATH) if f.startswith('checkpoint') and f.endswith('.pth')]
if checkpoints:
    checkpoints.sort(key=lambda x: int(x.split('_')[2].split('batch')[1].split('.')[0]))
    checkpoint_path = MODEL_STATES_PATH + checkpoints[-1] 
elif os.path.exists(COLAB_PATH):
    checkpoints = [f for f in os.listdir(COLAB_PATH) if f.startswith('checkpoint') and f.endswith('.pth')]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('_')[2].split('batch')[1].split('.')[0]))
        checkpoint_path = COLAB_PATH + checkpoints[-1]

checkpoint = torch.load(checkpoint_path, map_location=device)
model_state_dict = checkpoint['model_state_dict']
model.load_state_dict(model_state_dict)


start_tokens = tensor_text[0][:20].squeeze(0).unsqueeze(0).to(device)
generated_text = model.generate(start_tokens, max_new_tokens=400, temperature=1.0)
print(generated_text)
print(tokenizer.decode(generated_text.tolist()[0]))