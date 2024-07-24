import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):

    def __init__(self, X_data, Y_data):

        assert len(X_data) == len(Y_data), f"error, len(X_data) = {len(X_data)} != {len(Y_data)} = len(Y_data)"
        
        self.X_data = X_data
        self.Y_data = Y_data


    def __len__(self):
        
        assert len(self.X_data) == len(self.Y_data), f"error, len(X_data) = {len(self.X_data)} != {len(self.Y_data)} = len(Y_data)"
        
        return len(self.X_data)


    def __getitem__(self, idx):
        
        return {
            "image": torch.tensor(self.X_data[idx]),
            "label": torch.tensor(self.Y_data[idx], dtype=torch.long)
        }
