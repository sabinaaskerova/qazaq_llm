import os
import torch
from model import LanguageModel
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
import sentencepiece as spm
from project_config.data_config import *
from tokenization.special_tokenizer import SpecialTokenizerWrapper

################# Environment setup #################
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################# Data preparation #################
tokenizer_path = TOKENIZER_PATH
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(tokenizer_path + "m.model")
tensor_text = torch.load(tokenizer_path + "finetuning_tokenized_data.pt", map_location=device)

special_tokens = [
    "<s>", "</s>",
    "<instruction>", "</instruction>",
    "<context>", "</context>",
    "<response>", "</response>"
]

vocab_size = tokenizer.get_piece_size()
new_vocab_size = vocab_size + len(special_tokens)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_text):
        self.tensor_text = tensor_text
        
    def __len__(self):
        return self.tensor_text.shape[1] - 1
        
    def __getitem__(self, idx):
        # Find the start of the current sequence
        current_seq = self.tensor_text[0][idx:idx + 1024]  # Limit sequence length
        
        # Find positions of special tokens
        instruction_start = -1
        response_start = -1
        
        for i in range(len(current_seq) - 1):
            if current_seq[i] == tokenizer.piece_to_id("<instruction>"):
                instruction_start = i
            elif current_seq[i] == tokenizer.piece_to_id("<response>"):
                response_start = i
                break
        
        # If we found both markers, separate input and target
        if instruction_start != -1 and response_start != -1:
            input_seq = current_seq[instruction_start:response_start]
            target_seq = current_seq[response_start:response_start + 1]  # Only predict first response token
        else:
            # Fallback to original behavior if markers not found
            input_seq = current_seq[:-1]
            target_seq = current_seq[1:]
            
        return input_seq, target_seq

train_dataset = TextDataset(tensor_text)

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    collate_fn=lambda x: pad_sequences(x, vocab_size)  # New padding function
)

def pad_sequences(batch, vocab_size):
    # Separate inputs and targets
    inputs, targets = zip(*batch)
    
    max_input_len = max(len(seq) for seq in inputs)
    max_target_len = max(len(seq) for seq in targets)
    
    # Pad sequences
    padded_inputs = torch.full((len(batch), max_input_len), vocab_size-1)  # Use last token as padding
    padded_targets = torch.full((len(batch), max_target_len), vocab_size-1)
    
    for i, (input_seq, target_seq) in enumerate(zip(inputs, targets)):
        padded_inputs[i, :len(input_seq)] = input_seq
        padded_targets[i, :len(target_seq)] = target_seq
    
    return padded_inputs, padded_targets

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

def save_checkpoint(model, optimizer, epoch, batch_idx, best_loss, epochs_no_improve):
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'epochs_no_improve': epochs_no_improve
    }
    checkpoint_path = f'{COLAB_PATH}finetuning_checkpoint_epoch{epoch}_batch{batch_idx}.pth'
    torch.save(checkpoint, checkpoint_path) # I always train in colab so no choice now lol
    print(f'Finetuning Checkpoint saved at {checkpoint_path}')

    # Keep only last two checkpoints
    checkpoints = [f for f in os.listdir(COLAB_PATH) if f.startswith('finetuning_checkpoint') and f.endswith('.pth')]
    if len(checkpoints) > 2:
        checkpoints.sort(key=lambda x: int(x.split('_')[3].split('batch')[1].split('.')[0]))
        for old_checkpoint in checkpoints[:-2]:
            os.remove(checkpoint_path + old_checkpoint)

def load_checkpoint():
    checkpoint_path = None
    checkpoints = [f for f in os.listdir(COLAB_PATH) if f.startswith('finetuning_checkpoint') and f.endswith('.pth')]
   
    if checkpoints:
        config["vocab_size"] = new_vocab_size
        model = LanguageModel(config).to(device)
        checkpoints.sort(key=lambda x: int(x.split('_')[3].split('batch')[1].split('.')[0]))
        checkpoint_path = COLAB_PATH + checkpoints[-1]
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    else: # no fine-tuning checkpoints, we will use the pre-training checkpoint
        # I assume there is always a pre-training checkpoint, otherwise why would you be fine-tuning?
        checkpoints = [f for f in os.listdir(COLAB_PATH) if f.startswith('checkpoint') and f.endswith('.pth')]
        assert checkpoints, "No pre-training checkpoints found"
        checkpoints.sort(key=lambda x: int(x.split('_')[2].split('batch')[1].split('.')[0]))
        checkpoint_path = COLAB_PATH + checkpoints[-1]
        model = LanguageModel(config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.resize_token_embeddings(new_vocab_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return (
        model,
        optimizer,
        checkpoint['epoch'],
        checkpoint['batch_idx'],
        checkpoint['best_loss'],
        checkpoint.get('epochs_no_improve', 0)
    )
##################### Load checkpoint or initialize model
model, optimizer, start_epoch, start_batch, best_loss, epochs_no_improve = load_checkpoint()

# Training configuration
criterion = nn.CrossEntropyLoss(ignore_index=vocab_size-1)  # Ignore padding token
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
checkpoint_interval = 15000 # Save checkpoint every n batches

##################### Training loop
num_epochs = 100
accumulation_steps = 4  # Gradient accumulation steps

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if epoch == start_epoch and batch_idx < start_batch:
            continue
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs, start_pos=0)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        
        # Gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        
        if (batch_idx + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, batch_idx, best_loss, epochs_no_improve)
            
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item() * accumulation_steps}")
    
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss}")
    
    # Checkpoint saving and early stopping
    save_checkpoint(model, optimizer, epoch, batch_idx, best_loss, epochs_no_improve)
    scheduler.step(average_loss)
    
    if average_loss < best_loss:
        best_loss = average_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= 5:  # Early stopping patience
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

# Add generation function for inference
def generate_response(model, instruction, tokenizer, max_length=100):
    model.eval()
    with torch.no_grad():
        # Format input
        input_text = f"<instruction>{instruction}</instruction><response>"
        input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0).to(device)
        
        # Generate response
        output_ids = []
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token = outputs[:, -1, :].argmax(dim=-1)
            
            if next_token.item() == tokenizer.piece_to_id("</response>"):
                break
                
            output_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        # Decode response
        response = tokenizer.decode(output_ids)
        
        # Clean up response
        if "</response>" in response:
            response = response.split("</response>")[0]
        
        return response.strip()