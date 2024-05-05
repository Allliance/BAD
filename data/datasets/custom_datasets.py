from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate
import torchvision
from torchvision.datasets import ImageFolder
import os
import random
import numpy as np


class MixedDataset(Dataset):
    def __init__(self, datasets, label, length, transform=None, datasets_probs=None):
        '''
        prob_dist is a probability distribution, according to which, a sample from datasets is selected
        '''
        self.datasets = datasets
        self.transform = transform
        self.length = length
        self.label = label
        
        self.datasets_probs = None
        if datasets_probs is not None:
            assert len(datasets_probs) == len(datasets)
            self.datasets_probs = datasets_probs

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        target_dataset = self.datasets[np.random.choice(len(self.datasets), p=self.datasets_probs)]
        sample_idx = np.random.randint(len(target_dataset))
        sample, _ = target_dataset[sample_idx]
        
        if self.transform is not None:
                
            to_pil = transforms.ToPILImage()
            sample = to_pil(sample)
    
            sample = self.transform(sample)
        
        return sample, self.label

class SingleLabelDataset(Dataset):
    # defining values in the constructor
    def __init__(self, label, dataset, transform = None, lim=None):
        self.dataset = dataset
        self.len = len(dataset)
        if lim is not None:
            self.len = min(lim, self.len)

        self.label = label

    # Getting the data samples
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]

        return image, self.label

    # Getting data size/length
    def __len__(self):
        return self.len

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
    
