from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate
import torchvision
import os
import random

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

class MixedDataset(Dataset):
    def __init__(self, base_dataset, imagenet_root, label, transform=None, mixup_alpha=0.2):
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

        if self.transform:
            mixed_img = self.transform(mixed_img)
        
        return mixed_img, self.label

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
    