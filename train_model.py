import torch
from model import LanguageModel
from tokenizer import Tokenizer
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


tokenizer = Tokenizer("m.model")
tensor_text = torch.load("tensor_text.pt")
num_unique_tokens = torch.load("num_unique_tokens.pt")

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_text):
        self.tensor_text = tensor_text

    def __len__(self):
        return self.tensor_text.shape[1]

    def __getitem__(self, idx):
        return self.tensor_text[0][idx], self.tensor_text[0][idx + 1]

dataset = TextDataset(tensor_text)

train_size = len(dataset)
train_dataset = dataset

batch_size = 256
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f"Number of batches: {len(train_loader)}")
print(f"Number of samples: {len(train_loader) * batch_size}")
print(f"Number of unique tokens: {num_unique_tokens}")
print(f"Number of tokens: {len(tensor_text[0])}")
print(f"Batch size: {batch_size}")
config = {
    "d_model": 512,
    "eps": 1e-6,
    "n_queries": 128,
    "n_keys": 128,
    "n_values": 128,
    "n_heads": 8,
    "n_layers": 6,
    "n_positions": 1000,
    "vocab_size": num_unique_tokens
}

model = LanguageModel(config)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.unsqueeze(0)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.squeeze(0)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Print every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Batch Loss: {loss.item()}")
    
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss}")


model.eval()

start_tokens = torch.tensor([tensor_text[0][:20]], dtype=torch.long).to(device) # Starting with the first token
generated_text = model.generate(start_tokens, max_new_tokens=100, temperature=1.0)
print(tokenizer.decode(generated_text.tolist()[0]))