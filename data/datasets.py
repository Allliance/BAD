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


class GaussianDataset(Dataset):
    def __init__(self, label, num_samples=3000, size=32):
        self.data = torch.randn((num_samples,3,size,size))
        self.label = label
        self.transform = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.to_pil(self.data[idx])
        return self.transform(sample), self.label

class BlankDataset(Dataset):
    def __init__(self, label, num_samples=3000, size=32, color=0):
        self.data = torch.ones((num_samples,3,size,size)) * color
        self.label = label
        self.transform = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.to_pil(self.data[idx])
        return self.transform(sample), self.label