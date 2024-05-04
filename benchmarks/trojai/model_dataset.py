import random
import torch
import pandas as pd

import warnings
import os

warnings.simplefilter("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TrojAIDataset(Dataset):
    def __init__(self, root_dir, rnd, model_loader, shuffle=True, data_csv=None,
                 size=224, use_bgr=False, sample=False, sample_portion=0.2, custom_arch=None):
        names = [x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]        
        self.bgr = rnd in [0, 1]
        self.model_loader = model_loader
        self.root_dir = root_dir
        self.round = rnd
        
        self.model_data = []
        
        if isinstance(data_csv, str):
            data_csv = pd.read_csv(data_csv)
        
        data_csv.set_index('model_name', inplace=True)

        data = data_csv.to_dict(orient='index')
        
        for name in names:
            if custom_arch and data[name]['model_architecture'] != custom_arch:
                continue
            model_path = os.path.join(root_dir, name, 'model.pt')
            self.model_data.append({
                'name': name,
                'bgr': self.bgr,
                'label': 1 - int(data[name]['ground_truth']),
                'model_path': model_path,
                'num_classes': data[name]['number_classes'],
                'arch': data[name]['model_architecture'],
                'data': data,
            })
        
        random.shuffle(self.model_data)
        if sample:
            self.model_data = random.sample(self.model_data, int(sample_portion * len(self.model_data)))
        
    def __len__(self):
        return len(self.model_data)

    def __getitem__(self, idx):
        model = self.model_loader(self.model_data[idx], meta_data=self.model_data[idx], normalize=False)
        return model, model.meta_data['label']
    
