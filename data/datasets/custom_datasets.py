from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate
import torchvision
import os

class SingleLabelDataset(Dataset):
    # defining values in the constructor
    def __init__(self, label, dataset, transform = None, lim=None):
        self.data = [dataset[i][0] for i in range(len(dataset) if lim is None else min(lim, len(dataset)))]
        
        self.label = label

    # Getting the data samples
    def __getitem__(self, idx):
        image = self.data[idx]
        return image, self.label

    # Getting data size/length
    def __len__(self):
        return len(self.data)

class DummyDataset(Dataset):
    def __init__(self, label, pattern, pattern_args={}, transform=None):
        num_samples = pattern_args.get('num_samples', 1000)
        size = pattern_args.get('size', 32)
        
        if pattern == 'gaussian':
            self.data = torch.randn((num_samples, 3, size, size))
        elif pattern == 'blank':
            color = pattern_args.get('color', 0)
            self.data = torch.ones((num_samples, 3, size, size)) * color
        elif pattern == 'uniform':
            low = pattern_args.get('low', 0)
            high = pattern_args.get('high', 1)
            self.data = torch.rand((num_samples, 3, size, size)) * (high - low) + low
        else:
            raise ValueError('Invalid pattern')
        self.label = label
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.to_pil(self.data[idx])
        return self.transform(sample), self.label
    