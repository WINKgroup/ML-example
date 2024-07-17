from rich import print

import h5py

DATASET_PATH = "./data/MNISTdata.hdf5"

MNIST_data = h5py.File(DATASET_PATH, 'r')

X_train = MNIST_data['x_train'][:]
Y_train = MNIST_data['y_train'][:]

X_val = X_train[50000:60000]
Y_val = Y_train[50000:60000]

X_train = X_train[0:50000]
Y_train = Y_train[0:50000]

X_test = MNIST_data['x_test'][:]
Y_test = MNIST_data['y_test'][:]

print(f"X_train.shape: {X_train.shape}")
print(f"Y_train.shape: {Y_train.shape}")
print(f"X_val.shape  : {X_val.shape}")
print(f"Y_val.shape  : {Y_val.shape}")
print(f"X_test.shape : {X_test.shape}")
print(f"Y_test.shape : {Y_test.shape}")

from data.MNISTDataset import MNISTDataset

ds_train = MNISTDataset(X_data=X_train, Y_data=Y_train)
ds_val = MNISTDataset(X_data=X_val, Y_data=Y_val)
ds_test = MNISTDataset(X_data=X_test, Y_data=Y_test)

from torch.utils.data import DataLoader

BATCH_SIZE = 4096

train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

from models.SimpleCNN import SimpleCNN
from torch import Tensor

model = SimpleCNN()

EPOCHS = 3

from utils.progress import get_pbar, get_tasks
from rich.console import Console

console = Console()
pbar = get_pbar(console)
pbar_tasks = get_tasks(
    pbar, EPOCHS, len(train_loader), len(val_loader), len(test_loader)
)

from torch.nn import CrossEntropyLoss

criterion = CrossEntropyLoss()

# from torch.optim.sgd import SGD
from torch.optim.adam import Adam

LR = 0.01
# optimizer = SGD(model.parameters(), lr=LR)
optimizer = Adam(model.parameters(), lr=LR)

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

from trainer.trainer import train_loop

train_loop(EPOCHS, train_loader, val_loader, model, criterion, optimizer, device, pbar, pbar_tasks)

from trainer.trainer import validation_loop

checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),

}
torch.save(checkpoint, "./checkpoints/checkpoint.pth")

#### begin deployment

model = SimpleCNN()

ckpt = torch.load("./checkpoints/checkpoint.pth")

model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])

validation_loop(test_loader, model, criterion, device, pbar, pbar_tasks, "test")