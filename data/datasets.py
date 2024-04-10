from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import torchvision
import os

class ModelDataset(Dataset):
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.classes = ['cleans', 'bads']
        self.model_paths = []
        self.labels = []
        for cls in self.classes:
            class_folder = os.path.join(root_folder, cls)
            label = 0 if cls == 'cleans' else 1
            for filename in os.listdir(class_folder):
                if filename.endswith('.pt'):
                    self.model_paths.append(os.path.join(class_folder, filename))
                    self.labels.append(label)
        self.data = [feature_extractor(path) for path in model_paths]

    def __len__(self):
        return len(self.model_paths)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

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
