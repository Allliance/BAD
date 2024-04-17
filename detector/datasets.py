import os, random, pathlib
from torch.utils.data import Dataset
import re
CLEAN_LABEL = 'clean'

def extract_models_paths(root_dir):
    paths = []
    for root, _, files in os.walk(root_dir):
        new_paths = [os.path.join(root, file) for file in files if file.endswith('.pt')]
        paths += new_paths
    return paths

def extract_target(filepath):
    match = re.search(r'target(\d)', filepath)
    if match:
        return int(match.group(1))
    raise ValueError("The target for the given filepath is not identified.")

def extract_info_from_filepath(filepath):
    target = extract_target(filepath)
    return {
        'path': filepath,
        'target': target
    }

class ModelDataset(Dataset):
    def __init__(self, cleans_folder, bads_folder, model_loader, sample=False, sample_k=5):
        self.data = []
        self.loader = model_loader
        
        attack_folders = [x for x in os.listdir(bads_folder) if os.path.isdir(os.path.join(bads_folder, x))]
        for attack_folder in attack_folders:
            bad_data = [extract_info_from_filepath(model_path) for model_path in \
                extract_models_paths(os.path.join(bads_folder, attack_folder))]
            
            for i in range(len(bad_data)):
                model_data[i]['attack'] = attack_folder

            if sample:
                bad_data = random.sample(bad_data, sample_k)
            self.data += bad_data
        
        cleans_data = [extract_info_from_filepath(model_path) for model_path in extract_models_paths(cleans_folder)]
        
        if sample:
            cleans_data = random.sample(clean_checkpoints, sample_k*4)
        self.data += cleans_data
        
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.loader(self.data[idx]['path']),
    int(self.data[idx].get('attack') is not None),
    self.data[idx]