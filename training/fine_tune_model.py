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
        instruction = f"<s><instruction> {row['instruction_kz']} </instruction>"
        context = f"<context> {row['context_kz']} </context>" if 'context' in row and pd.notna(row['context']) else ""
        response = f"<response> {row['response_kz']} </response></s>"
        full_text = f"{instruction}{context}{response}"
        tokenized = tokenize_with_special_tokens(full_text)
        tokenized_data.append(torch.tensor(tokenized))
    return tokenized_data

csv_path = INSTRUCTION_DATA_PATH + "dolly_kz.csv"
tokenized_data = prepare_and_tokenize_data(csv_path) # list of tokenized tensors

#4 Save the tokenized data
torch.save(tokenized_data, TOKENIZER_PATH + "finetuning_tokenized_data.pt")
print(f"Tokenized data saved to {TOKENIZER_PATH}finetuning_tokenized_data.pt")


# 5. Dataset and DataLoader
class QADataset(Dataset):
    def __init__(self, tokenized_data, eos_token_id):
        self.tokenized_data = tokenized_data
        self.eos_token_id = eos_token_id

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

    def collate_fn(self, batch):
        # Find the maximum length in the batch
        max_len = max(len(seq) for seq in batch)
        # Pad sequences to the maximum length
        pad_token_id = tokenizer.pad_id()  # Ensure your SentencePiece model has a padding token
        padded_batch = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)

        for i, seq in enumerate(batch):
            padded_batch[i, :len(seq)] = seq
        return padded_batch, padded_batch  # Return inputs and targets

# Load the tokenized data
tokenized_data = torch.load(TOKENIZER_PATH + "finetuning_tokenized_data.pt")

eos_token_id = tokenizer.eos_id()
dataset = QADataset(tokenized_data, eos_token_id)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

print(f"Number of batches: {len(dataloader)}")
print(f"Number of samples: {len(dataloader) * batch_size}")
print(f"Batch size: {batch_size}")
    
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

# Function to save checkpoints
def save_checkpoint(model, optimizer, epoch, batch_idx, best_loss, epochs_no_improve):
    checkpoint_path = f'{MODEL_STATES_PATH}finetuning_checkpoint_epoch{epoch}_batch{batch_idx}.pth'
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'epochs_no_improve': epochs_no_improve
    }
    torch.save(checkpoint, checkpoint_path)
    if os.path.exists(COLAB_PATH):
        torch.save(checkpoint, f'{COLAB_PATH}finetuning_checkpoint_epoch{epoch}_batch{batch_idx}.pth')
    print(f'Finetuning Checkpoint saved at {checkpoint_path}')

    # Delete previous checkpoints
    checkpoints = [f for f in os.listdir(MODEL_STATES_PATH) if f.startswith('finetuning_checkpoint') and f.endswith('.pth')]
    if checkpoints:
        if len(checkpoints) > 2:
            checkpoints.sort(key=lambda x: int(x.split('_')[2].split('batch')[1].split('.')[0]))
            for old_checkpoint in checkpoints[:-2]: # Keep the last two checkpoints
                os.remove(MODEL_STATES_PATH + old_checkpoint)
             
    elif os.path.exists(COLAB_PATH):
        checkpoints = [f for f in os.listdir(COLAB_PATH) if f.startswith('finetuning_checkpoint') and f.endswith('.pth')]
        if checkpoints:
            if len(checkpoints) > 2:
                checkpoints.sort(key=lambda x: int(x.split('_')[2].split('batch')[1].split('.')[0]))
                for old_checkpoint in checkpoints[:-2]:
                    os.remove(COLAB_PATH + old_checkpoint)

# Load checkpoint if exists
def load_checkpoint():
    checkpoints = [f for f in os.listdir(MODEL_STATES_PATH) if f.startswith('finetuning_checkpoint') and f.endswith('.pth')]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('_')[2].split('batch')[1].split('.')[0]))
        checkpoint_path = MODEL_STATES_PATH + checkpoints[-1] 
    elif os.path.exists(COLAB_PATH):
        checkpoints = [f for f in os.listdir(COLAB_PATH) if f.startswith('finetuning_checkpoint') and f.endswith('.pth')]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[2].split('batch')[1].split('.')[0]))
            checkpoint_path = COLAB_PATH + checkpoints[-1]
    else:
        return None, None, 0, 0, float('inf'), 0
        
    if checkpoints:
        torch.load(checkpoint_path, map_location=device, weights_only=True)


    model = LanguageModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    batch_idx = checkpoint['batch_idx']
    best_loss = checkpoint['best_loss']
    if 'epochs_no_improve' in checkpoint:
        epochs_no_improve = checkpoint['epochs_no_improve']
    else:
        epochs_no_improve = 0
    print(f'Resuming from checkpoint: {checkpoint_path}')
    return model, optimizer, epoch, batch_idx, best_loss, epochs_no_improve

##################### Load the pretraining checkpoint
finetuning_checkpoints = [f for f in os.listdir(MODEL_STATES_PATH) if f.startswith('finetuning_checkpoint') and f.endswith('.pth')]
if os.path.exists(COLAB_PATH):
    finetuning_checkpoints = [f for f in os.listdir(COLAB_PATH) if f.startswith('finetuning_checkpoint') and f.endswith('.pth')]
if not finetuning_checkpoints: # if there is no finetuning checkpoint, load the pretraining checkpoint
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
    new_vocab_size = tokenizer.vocab_size() + len(special_tokens)
    model.resize_token_embeddings(new_vocab_size)

    old_embedding = model.get_input_embeddings().weight.data
    if old_embedding.size(0) < new_vocab_size:
        new_embedding = torch.mean(old_embedding, dim=0)
        model.get_input_embeddings().weight.data[-len(special_tokens):] = new_embedding.unsqueeze(0).repeat(len(special_tokens), 1)

    # Load the state dict excluding the embedding and output layers
    filtered_state_dict = {k: v for k, v in model_state_dict.items() if 'embedding' not in k and 'out' not in k}
    model.load_state_dict(filtered_state_dict, strict=False)

    epochs_no_improve = 0
    start_epoch = 0
    start_batch = 0
    best_loss = float('inf')
    epochs_no_improve = 0

else: # if resuming from a finetuning checkpoint
    model, optimizer, start_epoch, start_batch, best_loss, epochs_no_improve = load_checkpoint()


# 7 Fine-Tuning
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
patience = 5 # Early stopping patience
criterion = nn.CrossEntropyLoss()
checkpoint_interval = 15000  # save model every n batches
# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)

##################### Fine-tuning loop
num_epochs = 100
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # inputs = inputs.unsqueeze(0)  # add batch dimension
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        start_pos = 0  # Initialize start_pos to 0 for each new batch
        print("input.shape", inputs.size())
        outputs = model(inputs, start_pos)  # Pass start_pos to the model
        # outputs = outputs.squeeze(0)  # to remove batch dimension
        # loss = criterion(outputs, targets)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))  # Flatten for CrossEntropyLoss
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
