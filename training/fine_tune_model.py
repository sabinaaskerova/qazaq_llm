import os
import torch
import pandas as pd
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from model import LanguageModel
from project_config.data_config import *
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# 1. Setup and Definitions
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = spm.SentencePieceProcessor()
tokenizer.load(TOKENIZER_PATH + "m.model")

special_tokens = {
    '<instruction>': tokenizer.vocab_size() + 1,
    '</instruction>': tokenizer.vocab_size() + 2,
    '<context>': tokenizer.vocab_size() + 3,
    '</context>': tokenizer.vocab_size() + 4,
    '<response>': tokenizer.vocab_size() + 5,
    '</response>': tokenizer.vocab_size() + 6
}

# 2. Tokenization Function
def tokenize_with_special_tokens(text):
    tokens = []
    for word in text.split():
        if word in special_tokens:
            tokens.append(special_tokens[word])
        else:
            tokens.extend(tokenizer.encode(word))
    return tokens

# 3. Data Preparation and Tokenization
def prepare_and_tokenize_data(csv_path):
    df = pd.read_csv(csv_path)
    tokenized_data = []
    for _, row in df.iterrows():
        instruction = f"<s><instruction>{row['instruction_kz']}</instruction>"
        context = f"<context>{row['context_kz']}</context>" if 'context' in row and pd.notna(row['context']) else ""
        response = f"<response>{row['response_kz']}</response></s>"
        full_text = f"{instruction}{context}{response}"
        tokenized = tokenize_with_special_tokens(full_text)
        tokenized_data.append(torch.tensor(tokenized))
    return tokenized_data

csv_path = INSTRUCTION_DATA_PATH + "dolly_kz.csv"
tokenized_data = prepare_and_tokenize_data(csv_path)

# 4. Save tokenized
torch.save(tokenized_data, TOKENIZER_PATH + "finetuning_tokenized_data.pt")
print(f"Tokenized data saved to {TOKENIZER_PATH}finetuning_tokenized_data.pt")

# 5. Dataset and DataLoader
class FineTuningDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item[:-1], item[1:]  # input, target
    
# Custom collate function
def collate_fn(batch):
    inputs, targets = zip(*batch)  # Unzip the inputs and targets
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)  # Pad inputs
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)  # Pad targets
    return inputs_padded, targets_padded

tokenized_data = torch.load(TOKENIZER_PATH + "finetuning_tokenized_data.pt")
dataset = FineTuningDataset(tokenized_data)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# 6. Model and Training Setup
config = {
    "n_embd": 1024,
    "eps": 1e-6,
    "n_heads": 8,
    "n_layers": 6,
    "max_seq_length": 1000,
    "vocab_size": tokenizer.vocab_size() + len(special_tokens),
    "max_batch_size": 32,
    "device": device,
    "multiple": 64
}

model = LanguageModel(config).to(device)

##################### Load the checkpoint
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
#########################


new_vocab_size = tokenizer.vocab_size() + len(special_tokens)
model.resize_token_embeddings(new_vocab_size)

old_embedding = model.get_input_embeddings().weight.data
if old_embedding.size(0) < new_vocab_size:
    new_embedding = torch.mean(old_embedding, dim=0)
    model.get_input_embeddings().weight.data[-len(special_tokens):] = new_embedding.unsqueeze(0).repeat(len(special_tokens), 1)

# Load the state dict excluding the embedding and output layers
filtered_state_dict = {k: v for k, v in model_state_dict.items() if 'embedding' not in k and 'out' not in k}
model.load_state_dict(filtered_state_dict, strict=False)

# 7 Fine-Tuning Loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        start_pos = 0  # Initialize start_pos to 0 for each new batch

        outputs = model(inputs, start_pos)  # Pass start_pos to the model
        outputs = outputs.squeeze(0)  # to remove batch dimension
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Batch Loss: {loss.item()}")

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss}")

    # Save the model after each epoch
    model_save_path = f"{MODEL_STATES_PATH}finetuned_model_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

print("Fine-tuning completed.")
