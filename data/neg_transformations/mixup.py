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

class MixupDataset(Dataset):
    def __init__(self, base_dataset, imagenet_root, label, transform=None, mixup_alpha=0.5, **kwargs):
        self.base_dataset = base_dataset
        self.imagenet_dataset = ImageFolder(imagenet_root,
                               transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
        self.mixup_alpha = mixup_alpha
        self.label = label
        self.transform = transform
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Imagenet normalization
#         ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        base_img, _ = self.base_dataset[idx]

        imagenet_idx = np.random.randint(len(self.imagenet_dataset))
        imagenet_img, _ = self.imagenet_dataset[imagenet_idx]

        mixed_img = (1 - self.mixup_alpha) * base_img + self.mixup_alpha * imagenet_img

        # if self.transform:
        #     mixed_img = self.transform(mixed_img)
        
        return mixed_img, self.label
