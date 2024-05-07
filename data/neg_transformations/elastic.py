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

class ElasticDataset(Dataset):
    def __init__(self, base_dataset, imagenet_root, label, transform=None, alpha_affine=100, sigma=50, alpha=100, p=1.0,**kwargs):
        self.base_dataset = base_dataset
        self.label = label
        self.transform = transform
        self.elastic = A.Compose([A.ElasticTransform(alpha=alpha, p=p, sigma=sigma, alpha_affine=alpha_affine),
                      ToTensorV2()])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        
        return self.elastic(image=image.permute(1, 2, 0).numpy())['image'], self.label
