import torch
from model import LanguageModel
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
tensor_text = torch.load(tokenizer_path+"tensor_text.pt", map_location=device)

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

model = LanguageModel(config).to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

checkpoint_interval = 1000

# Early stopping parameters
patience = 1
best_loss = float('inf')
epochs_no_improve = 0

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)

# Training loop
# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.unsqueeze(0)  # Add batch dimension
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        start_pos = 0  # Initialize start_pos to 0 for each new batch
        
        outputs = model(inputs, start_pos)  # Pass start_pos to the model
        outputs = outputs.squeeze(0)  # Remove batch dimension
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % checkpoint_interval == 0 or (batch_idx + 1) == len(train_loader):
            torch.save(model.state_dict(), f'model_checkpoint_epoch{epoch+1}_batch{batch_idx+1}.pth')

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Batch Loss: {loss.item()}")
            
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss}")

    # Learning rate scheduler step
    scheduler.step(average_loss)
    print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

    # Early stopping check
    if average_loss < best_loss:
        best_loss = average_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break


languagemodel_path = MODEL_STATES_PATH
if not os.path.exists(languagemodel_path):
    os.makedirs(languagemodel_path)

model_save_path = languagemodel_path+"language_model_state_dict.pth"
torch.save(model.state_dict(), model_save_path)

optimizer_save_path = languagemodel_path+"optimizer.pth"
torch.save(optimizer.state_dict(), optimizer_save_path)

print(f"Model and optimizer state saved to {model_save_path} and {optimizer_save_path}, respectively.")

model.eval()
start_tokens = tensor_text[0][:20].squeeze(0).unsqueeze(0).to(device)
generated_text = model.generate(start_tokens, max_new_tokens=100, temperature=1.0)
print(generated_text)
print(tokenizer.decode(generated_text.tolist()[0]))