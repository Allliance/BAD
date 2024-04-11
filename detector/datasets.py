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