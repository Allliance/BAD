from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import skimage.io
from BAD.benchmarks.trojai.utils import image_loader

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