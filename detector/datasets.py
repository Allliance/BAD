import os
from torch.utils.data import Dataset
CLEAN_LABEL = 'clean'

class ModelDataset(Dataset):
    def __init__(self, cleans_folder, bad_folders, model_loader, sample=False, sample_k=5):
        self.model_paths = []
        self.labels = []
        self.loader = model_loader
        for attack_folder in bad_folders:
            ck_folder = os.path.join(bads_folder, attack_folder)
            ck_files = [os.path.join(ck_folder, ck_file) for ck_file in os.listdir(ck_folder)]
            
            if sample:
                ck_files = random.sample(ck_files, k=sample_k)
                
            self.model_paths += ck_files
            self.labels += [attack_folder] * len(ck_files)
            
        
        clean_checkpoints = [os.path.join(CLEAN_ROOT, x) for x in os.listdir(CLEAN_ROOT)]
        if sample:
            clean_checkpoints = random.sample(clean_checkpoints, sample_k*4)
            
        self.model_paths += clean_checkpoints
        self.labels += [CLEAN_LABEL] * len(clean_checkpoints)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.loader(self.model_paths[idx]), self.labels[idx] == CLEAN_LABEL
    