from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import skimage.io
import os
import numpy as np
import torch

def image_loader(fn):
    img = skimage.io.imread(fn)
    img = img.astype(dtype=np.float32)

    # perform center crop to what the CNN is expecting 224x224
    h, w, c = img.shape
    dx = int((w - 224) / 2)
    dy = int((w - 224) / 2)
    img = img[dy:dy + 224, dx:dx + 224, :]

    # convert to CHW dimension ordering
    img = np.transpose(img, (2, 0, 1))
    # convert to NCHW dimension ordering
    img = np.expand_dims(img, 0)
    # normalize the image matching pytorch.transforms.ToTensor()
    img = img / 255.0

    # convert image to a gpu tensor
    return torch.from_numpy(img)[0]

class ExampleDataset(Dataset):
    def __init__(self, root_dir, size=224, use_bgr=False):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        self.bgr = use_bgr
        
        images_paths = [x for x in os.listdir(root_dir) if x.endswith('.png')]
        sample_data = pd.read_csv(os.path.join(root_dir, 'data.csv'))
        labels_dict = {}
        for index, row in sample_data.iterrows():
            if row['true_label'] != row['train_label']:
                raise ValueError("The provided folder contains poisonous data!")
            labels_dict[row['file']] = row['true_label']
            
        self.labels = [labels_dict[sample_name] for sample_name in images_paths]
        self.data = [image_loader(os.path.join(root_dir, image_path)) for image_path in images_paths]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.bgr:
            image = image[[2, 1, 0], :, :]
        if self.transform is not None:
            image = self.transform(image) 
        return image, self.labels[idx]