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
tensor_text = torch.load(tokenizer_path + "finetuning_tokenized_data.pt", map_location=device) # we will finetune the model on qna data

vocab_size = tokenizer.get_piece_size()
vocab_tokens = [tokenizer.id_to_piece(i) for i in range(vocab_size)]
print(f"First 100 tokens in vocabulary: {vocab_tokens[:100]}")

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

model = LanguageModel(config).to(device)


special_tokens = [
    "<s>", "</s>",
    "<instruction>", "</instruction>",
    "<context>", "</context>",
    "<response>", "</response>"
]


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


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
patience = 5 # Early stopping patience
criterion = nn.CrossEntropyLoss()
checkpoint_interval = 15000  # save model every n batches
# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)

##################### Training loop
num_epochs = 100
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
    # Save the model after each epoch
    model_save_path = f"{MODEL_STATES_PATH}finetuned_model_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Learning rate scheduler step
    scheduler.step(average_loss)
    print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

    model.eval()

    # Early stopping check
    if average_loss < best_loss:
        best_loss = average_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

