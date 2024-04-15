import os, random, pathlib
from torch.utils.data import Dataset
CLEAN_LABEL = 'clean'

def extract_attack_name(model_path):
    return 'bad'

class ModelDataset(Dataset):
    def __init__(self, cleans_folder, bads_folder, model_loader, sample=False, sample_k=5):
        self.model_paths = []
        self.labels = []
        self.loader = model_loader
        
        bad_checkpoints = [str(x) for x in pathlib.Path(bads_folder).iterdir() if str(x).endswith('.pt')]
        if sample:
            clean_checkpoints = random.sample(clean_checkpoints, sample_k*4)
        
        self.model_paths += bad_checkpoints
        self.labels += [extract_attack_name(x) for x in bad_checkpoints]
            
        
        clean_checkpoints = [str(x) for x in pathlib.Path(cleans_folder).iterdir() if str(x).endswith('.pt')]
        if sample:
            clean_checkpoints = random.sample(clean_checkpoints, sample_k*4)
            
        self.model_paths += clean_checkpoints
        self.labels += [CLEAN_LABEL] * len(clean_checkpoints)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.loader(self.model_paths[idx]), self.labels[idx] == CLEAN_LABEL