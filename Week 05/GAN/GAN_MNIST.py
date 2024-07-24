import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

# Hyperparameters
latent_size = 64
hidden_size = 256
image_size = 784  # 28x28
num_epochs = 10
batch_size = 2000
learning_rate = 0.0002
checkpoint_interval = 1  # Save checkpoint every 10 epochs

# Create directory for checkpoints
os.makedirs('checkpoints', exist_ok=True)

# DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Split the training set into train and validation sets
train_size = int(0.8 * len(mnist))
val_size = len(mnist) - train_size
train_dataset, val_dataset = random_split(mnist, [train_size, val_size])

# Create DataLoaders for each set
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Initialize models
D = Discriminator()
G = Generator()

# Loss and optimizer
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)

# Device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = D.to(device)
G = G.to(device)

# Training
epoch_progress = tqdm(range(num_epochs), desc="Epoch Progress", colour='blue')
for epoch in epoch_progress:
    batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Batch Progress", colour='green')
    for i, (images, _) in enumerate(batch_progress):
        images = images.reshape(batch_size, -1).to(device)

        # Create labels for real and fake data
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs.mean().item()

        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs.mean().item()

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)

        g_loss = criterion(outputs, real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        batch_progress.set_postfix(
            D_loss=f"{d_loss.item():.4f}",
            G_loss=f"{g_loss.item():.4f}",
            D_x=f"{real_score:.2f}",
            D_G_z=f"{fake_score:.2f}"
        )

    # Validation
    D.eval()
    G.eval()
    val_d_loss, val_g_loss = 0, 0
    val_batch_progress = tqdm(val_loader, desc=f"Validation {epoch+1}/{num_epochs} Batch Progress", colour='red')
    with torch.no_grad():
        for i, (images, _) in enumerate(val_batch_progress):
            images = images.reshape(batch_size, -1).to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Validate Discriminator
            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs.mean().item()

            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs.mean().item()
            d_loss = d_loss_real + d_loss_fake

            # Validate Generator
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
            g_loss = criterion(outputs, real_labels)

            val_d_loss += d_loss.item()
            val_g_loss += g_loss.item()

            val_batch_progress.set_postfix(
                Val_D_loss=f"{val_d_loss/len(val_loader):.4f}",
                Val_G_loss=f"{val_g_loss/len(val_loader):.4f}",
                Val_D_x=f"{real_score:.2f}",
                Val_D_G_z=f"{fake_score:.2f}"
            )
    D.train()
    G.train()

    # Save checkpoints
    if (epoch + 1) % checkpoint_interval == 0:
        torch.save(G.state_dict(), f'checkpoints/generator_epoch_{epoch+1}.pth')
        torch.save(D.state_dict(), f'checkpoints/discriminator_epoch_{epoch+1}.pth')
        print(f"Checkpoint saved for epoch {epoch+1}")

# Save final models
torch.save(G.state_dict(), 'checkpoints/generator_final.pth')
torch.save(D.state_dict(), 'checkpoints/discriminator_final.pth')

# Test
D.eval()
G.eval()
test_d_loss, test_g_loss = 0, 0
test_batch_progress = tqdm(test_loader, desc="Test Progress", colour='yellow')
with torch.no_grad():
    for i, (images, _) in enumerate(test_batch_progress):
        images = images.reshape(batch_size, -1).to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Test Discriminator
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs.mean().item()

        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs.mean().item()
        d_loss = d_loss_real + d_loss_fake

        # Test Generator
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)

        test_d_loss += d_loss.item()
        test_g_loss += g_loss.item()

        test_batch_progress.set_postfix(
            Test_D_loss=f"{test_d_loss/len(test_loader):.4f}",
            Test_G_loss=f"{test_g_loss/len(test_loader):.4f}",
            Test_D_x=f"{real_score:.2f}",
            Test_D_G_z=f"{fake_score:.2f}"
        )
