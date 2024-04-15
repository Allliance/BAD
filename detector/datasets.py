import os, random, pathlib
from torch.utils.data import Dataset
CLEAN_LABEL = 'clean'

def extract_models_paths(root_dir):
    paths = []
    for root, _, files in os.walk(root_dir):
        new_paths = [os.path.join(root, file) for file in files if file.endswith('.pt')]
        paths += new_paths
    return paths

class ModelDataset(Dataset):
    def __init__(self, cleans_folder, bads_folder, model_loader, sample=False, sample_k=5):
        self.model_paths = []
        self.labels = []
        self.loader = model_loader
        
        attack_folders = [x for x in os.listdir(bads_folder) if os.path.isdir(os.path.join(bads_folder, x))]
        for attack_folder in attack_folders:
            bad_checkpoints = extract_models_paths(os.path.join(bads_folder, attack_folder))
            if sample:
                bad_checkpoints = random.sample(bad_checkpoints, sample_k)
            self.model_paths += bad_checkpoints
            self.labels += [attack_folder] * len(bad_checkpoints)
        
        clean_checkpoints = extract_models_paths(cleans_folder)
        if sample:
            clean_checkpoints = random.sample(clean_checkpoints, sample_k*4)
        self.model_paths += clean_checkpoints
        
        random.shuffle(self.model_paths)
        all_labels = attack_folders + [CLEAN_LABEL]
        self.labels = [[x for x in all_labels if x in model_path][0] for model_path in self.model_paths]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        print(self.model_paths[idx], self.labels[idx]) 
        return self.loader(self.model_paths[idx]), self.labels[idx] == CLEAN_LABEL, self.labels[idx]