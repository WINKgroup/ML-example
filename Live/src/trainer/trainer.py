from torch.utils.data import DataLoader
from torch import Tensor
from models.SimpleCNN import SimpleCNN
from torch.nn import Module
from torch.optim import Optimizer
import torch
from rich.progress import Progress

def train_loop(
    epochs: int, 
    train_loader: DataLoader, 
    val_loader: DataLoader,  # Add validation loader
    model: SimpleCNN, 
    criterion: Module,
    optimizer: Optimizer,
    device: torch.device,
    pbar: Progress, 
    pbar_tasks: dict
):
    
    model.to(device)
    
    pbar.start()
    
    pbar.reset(pbar_tasks["epoch"])

    batch_losses = []
    batch_accuracies = []
    epoch_losses = []
    epoch_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):

        pbar.reset(pbar_tasks["train"])

        epoch_loss = 0.0
        correct = 0
        total = 0

        model.train()  # Set the model to training mode
        for train_batch in train_loader:

            x: Tensor = train_batch['image'].to(device)
            y: Tensor = train_batch['label'].to(device)

            x = x.reshape(-1, 1, 28, 28)

            y_logits: Tensor = model.forward(x)
            
            y = y.squeeze()

            loss: Tensor = criterion(y_logits, y)

            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()

            pbar.update(pbar_tasks["train"], advance=1)

            batch_losses.append(loss.item())
            _, predicted = torch.max(y_logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            batch_accuracy = 100 * (predicted == y).sum().item() / y.size(0)
            batch_accuracies.append(batch_accuracy)

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        epoch_accuracy = 100 * correct / total

        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        pbar.console.print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Validation step
        val_batch_losses, val_batch_accuracies, val_epoch_loss, val_epoch_accuracy = validation_loop(
            val_loader, model, criterion, device, pbar, pbar_tasks, "val"
        )

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

        pbar.console.print(f"Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.2f}%")
        
        pbar.update(pbar_tasks["epoch"], advance=1)

    return batch_losses, batch_accuracies, epoch_losses, epoch_accuracies, val_batch_losses, val_batch_accuracies, val_losses, val_accuracies


def validation_loop(
    val_loader: DataLoader, 
    model: SimpleCNN, 
    criterion: Module,
    device: torch.device,
    pbar: Progress, 
    pbar_tasks: dict,
    val_or_test: str
):
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    pbar.start()
    
    pbar.reset(pbar_tasks[f"{val_or_test}"])

    batch_losses = []
    batch_accuracies = []
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for val_batch in val_loader:

            x: Tensor = val_batch['image'].to(device)
            y: Tensor = val_batch['label'].to(device)

            x = x.reshape(-1, 1, 28, 28)

            y_logits: Tensor = model.forward(x)
            
            y = y.squeeze()

            loss: Tensor = criterion(y_logits, y)

            pbar.update(pbar_tasks[f"{val_or_test}"], advance=1)

            batch_losses.append(loss.item())
            _, predicted = torch.max(y_logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            batch_accuracy = 100 * (predicted == y).sum().item() / y.size(0)
            batch_accuracies.append(batch_accuracy)

            total_loss += loss.item()

        total_loss /= len(val_loader)
        accuracy = 100 * correct / total

        pbar.console.print(f"{val_or_test.capitalize()} Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return batch_losses, batch_accuracies, total_loss, accuracy
