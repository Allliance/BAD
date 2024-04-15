import copy
import pickle 
from torchvision import transforms
import torch
from .base_model import BaseModel as Model
from .preact import PreActResNet18

def load_preact(record_path, device, num_classes=10):
    
    load_file = torch.load(record_path)

    new_dict = copy.deepcopy(load_file['model'])
    for k, v in load_file['model'].items():
        if k.startswith('module.'):
            del new_dict[k]
            new_dict[k[7:]] = v

    load_file['model'] = new_dict
    
    net = PreActResNet18(num_classes=num_classes)
    net.load_state_dict(load_file['model'])
    
    model = Model(net, device)
    model.to(device)
    model.eval()
    
    return model