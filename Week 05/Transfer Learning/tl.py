import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Hyperparameters
num_epochs = 2
batch_size = 128
learning_rate = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 dataset
train_dataset_A = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset_A = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# CIFAR-100 dataset
train_dataset_B = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset_B = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Data loaders
train_loader_A = DataLoader(dataset=train_dataset_A, batch_size=batch_size, shuffle=True)
test_loader_A = DataLoader(dataset=test_dataset_A, batch_size=batch_size, shuffle=False)
train_loader_B = DataLoader(dataset=train_dataset_B, batch_size=batch_size, shuffle=True)
test_loader_B = DataLoader(dataset=test_dataset_B, batch_size=batch_size, shuffle=False)

# Simple CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training function
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        running_loss = 0.0
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

# Testing function
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    return accuracy

# Train on Dataset A (CIFAR-10)
model_A = CNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_A.parameters(), lr=learning_rate)

print("Training model from scratch on CIFAR-10")
train(model_A, train_loader_A, criterion, optimizer, num_epochs)
print("Testing model trained on CIFAR-10")
accuracy_A = test(model_A, test_loader_A)

# Fine-tune on Dataset B (CIFAR-100)
model_A.classifier[2] = nn.Linear(256, 100)  # Adjust classifier for 100 classes
model_A = model_A.to(device)
optimizer = optim.Adam(model_A.parameters(), lr=learning_rate)

print("Fine-tuning model on CIFAR-100")
train(model_A, train_loader_B, criterion, optimizer, num_epochs)
print("Testing fine-tuned model on CIFAR-100")
accuracy_fine_tune_B = test(model_A, test_loader_B)

# Train from scratch on Dataset B (CIFAR-100)
model_B = CNN(num_classes=100).to(device)
optimizer = optim.Adam(model_B.parameters(), lr=learning_rate)

print("Training model from scratch on CIFAR-100")
train(model_B, train_loader_B, criterion, optimizer, num_epochs)
print("Testing model trained from scratch on CIFAR-100")
accuracy_scratch_B = test(model_B, test_loader_B)

print(f"Accuracy on CIFAR-10: {accuracy_A:.2f}%")
print(f"Accuracy on CIFAR-100 (Fine-tuned): {accuracy_fine_tune_B:.2f}%")
print(f"Accuracy on CIFAR-100 (From scratch): {accuracy_scratch_B:.2f}%")
