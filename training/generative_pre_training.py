import os
import torch
from model import LanguageModel
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
import sentencepiece as spm
from project_config.data_config import *

################# Environment setup #################
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################# Data preparation #################
tokenizer_path = TOKENIZER_PATH
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(tokenizer_path + "m.model")
tensor_text = torch.load(tokenizer_path + "tensor_text.pt", map_location=device) # we will pretrain the model on the same dataset as the tokenizer

vocab_size = tokenizer.get_piece_size()
vocab_tokens = [tokenizer.id_to_piece(i) for i in range(vocab_size)]
print(f"First 10 tokens in vocabulary: {vocab_tokens[:100]}")

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_text):
        self.tensor_text = tensor_text

    def __len__(self):
        return self.tensor_text.shape[1] - 1  # Last token has no next token

    def __getitem__(self, idx):
        return self.tensor_text[0][idx], self.tensor_text[0][idx + 1]

dataset = TextDataset(tensor_text)

train_size = len(dataset)
print(f"Train size: {train_size}")
train_dataset = dataset

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f"Number of batches: {len(train_loader)}")
print(f"Number of samples: {len(train_loader) * batch_size}")
print(f"Batch size: {batch_size}")

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

# Function to save checkpoints
def save_checkpoint(model, optimizer, epoch, batch_idx, best_loss, epochs_no_improve):
    checkpoint_path = f'{MODEL_STATES_PATH}checkpoint_epoch{epoch}_batch{batch_idx}.pth'
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
        torch.save(checkpoint, f'{COLAB_PATH}checkpoint_epoch{epoch}_batch{batch_idx}.pth')
    print(f'Checkpoint saved at {checkpoint_path}')

    # Delete previous checkpoints
    checkpoints = [f for f in os.listdir(MODEL_STATES_PATH) if f.startswith('checkpoint') and f.endswith('.pth')]
    if checkpoints:
        if len(checkpoints) > 2:
            checkpoints.sort(key=lambda x: int(x.split('_')[2].split('batch')[1].split('.')[0]))
            for old_checkpoint in checkpoints[:-2]: # Keep the last two checkpoints
                os.remove(MODEL_STATES_PATH + old_checkpoint)
             
    elif os.path.exists(COLAB_PATH):
        checkpoints = [f for f in os.listdir(COLAB_PATH) if f.startswith('checkpoint') and f.endswith('.pth')]
        if checkpoints:
            if len(checkpoints) > 2:
                checkpoints.sort(key=lambda x: int(x.split('_')[2].split('batch')[1].split('.')[0]))
                for old_checkpoint in checkpoints[:-2]:
                    os.remove(COLAB_PATH + old_checkpoint)

# Load checkpoint if exists
def load_checkpoint():
    checkpoints = [f for f in os.listdir(MODEL_STATES_PATH) if f.startswith('checkpoint') and f.endswith('.pth')]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('_')[2].split('batch')[1].split('.')[0]))
        checkpoint_path = MODEL_STATES_PATH + checkpoints[-1] 
    elif os.path.exists(COLAB_PATH):
        checkpoints = [f for f in os.listdir(COLAB_PATH) if f.startswith('checkpoint') and f.endswith('.pth')]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[2].split('batch')[1].split('.')[0]))
            checkpoint_path = COLAB_PATH + checkpoints[-1]
    else:
        return None, None, 0, 0, float('inf'), 0
        
    if checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)

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

# Attempt to resume from checkpoint 
model, optimizer, start_epoch, start_batch, best_loss, epochs_no_improve = load_checkpoint()
if model is None:
    model = LanguageModel(config).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs_no_improve = 0

patience = 5 # Early stopping patience
criterion = nn.CrossEntropyLoss()
checkpoint_interval = 15000  # save model every n batches
# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)

##################### Training loop
num_epochs = 500
for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Skip batches if resuming
        if epoch == start_epoch and batch_idx < start_batch:
            continue

        inputs = inputs.unsqueeze(0)  # add batch dimension
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        start_pos = 0  # Initialize start_pos to 0 for each new batch

        outputs = model(inputs, start_pos)  # Pass start_pos to the model
        outputs = outputs.squeeze(0)  # to remove batch dimension
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % checkpoint_interval == 0 or (batch_idx + 1) == len(train_loader):
            save_checkpoint(model, optimizer, epoch, batch_idx, best_loss, epochs_no_improve)

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Batch Loss: {loss.item()}")

    save_checkpoint(model, optimizer, epoch, batch_idx, best_loss, epochs_no_improve)

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss}")

    # Learning rate scheduler step
    scheduler.step(average_loss)
    print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

    model.eval()
    start_tokens = tensor_text[0][:20].squeeze(0).unsqueeze(0).to(device)
    generated_text = model.generate(start_tokens, max_new_tokens=400, temperature=0.5)
    resulting_text = tokenizer.decode(generated_text.tolist()[0])
    generated_text_path = GENERATED_TEXT_TEST
    separator = "------------------------"
    with open(generated_text_path, "a") as file:
        file.write(f"\n{separator}\nEpoch {epoch+1}\n{resulting_text}\n")

    # Early stopping check
    if average_loss < best_loss:
        best_loss = average_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

################# Save model and optimizer state at the end of the training #################
languagemodel_path = MODEL_STATES_PATH
if not os.path.exists(languagemodel_path):
    os.makedirs(languagemodel_path)

model_save_path = languagemodel_path + "language_model_state_dict.pth"
torch.save(model.state_dict(), model_save_path)
if os.path.exists(COLAB_PATH):
    torch.save(model.state_dict(), f'{COLAB_PATH}language_model_state_dict.pth')

optimizer_save_path = languagemodel_path + "optimizer.pth"
torch.save(optimizer.state_dict(), optimizer_save_path)
if os.path.exists(COLAB_PATH):
    torch.save(optimizer.state_dict(), f'{COLAB_PATH}optimizer.pth')

print(f"Model and optimizer state saved to {model_save_path} and {optimizer_save_path}, respectively.")
