import torch
from torch.utils.data import Dataset
import random

class RotationDataset(Dataset):
    def __init__(self, base_dataset, label, transform=None):
        self.base_dataset = base_dataset
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        
        image = torch.rot90(image, k=random.randint(1, 3), dims=(1, 2))

        return image, self.label