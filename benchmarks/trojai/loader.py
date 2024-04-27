import random
import torch

from BAD.models.base_model import BaseModel as Model
from BAD.benchmarks.trojai.dataset import ExampleDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TrojAIDataset(Dataset):
    def __init__(self, root_dir, round, model_loader, shuffle=True, data_csv=None, size=224, use_bgr=False):
        names = [x for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]        
        self.bgr = round in [0, 1]
        self.model_loader = model_loader
        self.root_dir = root_dir
        self.round = round
        
        self.model_data = []
        for name in names:
            model_path = os.path.join(root_dir, name, 'model.pt')
            self.model_data.append({
                'label': 1 - int(data[name]['ground_truth']),
                'model_path': model_path,
                'num_classes': data[name]['number_classes'],
                'arch': data[name]['model_architecture'],
                'data': data,
            })
        
        random.shuffle(self.model_data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        model = self.model_loader(self.model_data[idx])
        dataset = ExampleDataset(root_dir=os.path.join(self.root_dir,
                                                       self.model_data[idx]['name'], 'example_data'), use_bgr=self.bgr)
        return model, dataset
    

def load_model(model_data):
    model_path = model_data['model_path']
    arch = model_data['arch']
    num_classes = model_data['num_classes']
    
    net = torch.load(model_path, map_location=device)
    
    if arch == 'inceptionv3':
        new_net = inception_v3(num_classes=num_classes)
        new_net.load_state_dict(deepcopy(net.state_dict()), strict=False)
        net = new_net
    
    feature_extractor = torch.nn.Sequential(*list(net.children())[:-1])
    
    model = Model(net, feature_extractor=feature_extractor, **model_kwargs)
    model.to(device)
    model.eval()
    
    return model