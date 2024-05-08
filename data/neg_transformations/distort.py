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
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DistortDataset(Dataset):
    def __init__(self, base_dataset, label, transform=None, num_steps=10, distort_limit=1, p=1.0,**kwargs):
        self.base_dataset = base_dataset
        self.label = label
        self.transform = transform
        self.distort = A.Compose([A.GridDistortion(num_steps=10, distort_limit=1, p=1.0),
                    ToTensorV2()])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        
        return self.distort(image=image.permute(1, 2, 0).numpy())['image'], self.label
