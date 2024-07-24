import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import PennTreebank
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torchtext
import os

torchtext.disable_torchtext_deprecation_warning()

# Hyperparameters
batch_size = 512
epochs = 10
learning_rate = 0.01
val_split = 0.1
test_split = 0.1
model_dim = 512
num_heads = 8
num_layers = 6
context_size = 30
checkpoint_dir = './checkpoints'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Tokenizer and vocabulary
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

train_iter = PennTreebank(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

# Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, data, vocab, context_size):
        self.data = [torch.tensor(vocab(tokenizer(text)), dtype=torch.long) for text in data]
        self.context_size = context_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        x = text[:-1]
        y = text[1:]
        return x[:self.context_size], y[:self.context_size]

train_iter, val_iter, test_iter = PennTreebank(split=('train', 'valid', 'test'))

train_dataset = TextDataset(train_iter, vocab, context_size)
val_dataset = TextDataset(val_iter, vocab, context_size)
test_dataset = TextDataset(test_iter, vocab, context_size)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=vocab["<pad>"])
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=vocab["<pad>"])
    return inputs_padded, targets_padded

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)

# Transformer model for language modeling
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, num_layers, context_size):
        super(TransformerLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, context_size, model_dim))
        self.transformer = nn.Transformer(model_dim, num_heads, num_layers)
        self.fc = nn.Linear(model_dim, vocab_size)

    def forward(self, src, tgt_mask=None):
        src = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        src = src.permute(1, 0, 2)
        output = self.transformer(src, src, tgt_mask=tgt_mask)
        output = output.permute(1, 0, 2)
        output = self.fc(output)
        return output

# Model instance, loss function, and optimizer
model = TransformerLM(len(vocab), model_dim, num_heads, num_layers, context_size)
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Accuracy calculation
def calculate_accuracy(outputs, targets, ignore_index):
    _, predicted = torch.max(outputs, -1)
    mask = targets != ignore_index
    correct = (predicted == targets) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy

# Training loop with teacher forcing
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    train_pbar = tqdm(loader, desc="Training", leave=False, colour="blue")
    for inputs, targets in train_pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Use teacher forcing
        outputs = model(inputs)
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_accuracy += calculate_accuracy(outputs, targets, vocab["<pad>"]) * inputs.size(0)
        train_pbar.set_postfix(loss=running_loss / len(loader.dataset), accuracy=running_accuracy / len(loader.dataset))
    return running_loss / len(loader.dataset), running_accuracy / len(loader.dataset)

# Validation and testing loop without teacher forcing
def evaluate_epoch(model, loader, criterion, device, phase="Validating"):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    val_pbar = tqdm(loader, desc=phase, leave=False, colour="green" if phase == "Validating" else "red")
    with torch.no_grad():
        for inputs, targets in val_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            # Autoregression: feed inputs one by one
            batch_size, seq_len = inputs.size()
            outputs = torch.zeros(batch_size, seq_len, len(vocab)).to(device)
            for t in range(seq_len):
                output = model(inputs[:, :t+1])
                outputs[:, t, :] = output[:, t, :]
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            running_accuracy += calculate_accuracy(outputs, targets, vocab["<pad>"]) * inputs.size(0)
            val_pbar.set_postfix(loss=running_loss / len(loader.dataset), accuracy=running_accuracy / len(loader.dataset))
    return running_loss / len(loader.dataset), running_accuracy / len(loader.dataset)

# Main training and evaluation loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = evaluate_epoch(model, val_loader, criterion, device, phase="Validating")
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
    }, checkpoint_path)

# Test the model
def test_model(model, loader, criterion, device):
    test_loss, test_accuracy = evaluate_epoch(model, loader, criterion, device, phase="Testing")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Evaluate on test set
test_model(model, test_loader, criterion, device)
