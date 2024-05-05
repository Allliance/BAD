import random
import torch
import pandas as pd
import warnings
import os
from copy import deepcopy

from torchvision import transforms
from torchvision.models import inception_v3
from BAD.models.base_model import BaseModel as Model
from torch.utils.data import DataLoader
from BAD.benchmarks.trojai.dataset import ExampleDataset
from BAD.data.loaders import get_ood_loader
from BAD.data.utils import sample_dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

archs_batch_sizes = {
    'resnet50': 32,
    'densenet121': 32,
    'inceptionv3': 64,
    'default': 32,
    'wideresnet101': 16,
}

def load_model(model_data, **model_kwargs):
    model_path = model_data['model_path']
    arch = model_data['arch']
    num_classes = model_data['num_classes']
    try:
        net = torch.load(model_path, map_location=device)
    except Exception as e:
        print("facing problems in while loading this model")
    
    
    if arch == 'inceptionv3':
        new_net = inception_v3(num_classes=num_classes)
        new_net.load_state_dict(deepcopy(net.state_dict()), strict=False)
        net = new_net
    
    feature_extractor = torch.nn.Sequential(*list(net.children())[:-1])
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    model = Model(net, feature_extractor=feature_extractor, **model_kwargs)
    model.to(device)
    model.eval()
    
    return model


def get_dataset_trojai(model):
    return ExampleDataset(root_dir=os.path.join(os.path.dirname(model.meta_data['model_path'])
                                                   , 'example_data'), use_bgr=model.meta_data['bgr'])


def get_sanityloader_trojai(model, batch_size=None):
    if batch_size is None:
        arch = model.meta_data['arch']
        if arch not in archs_batch_sizes:
            arch = 'default'
        batch_size = archs_batch_sizes[arch]
    return DataLoader(get_dataset_trojai(model), shuffle=True, batch_size=batch_size)


def get_oodloader_trojai(model, out_dataset, sample_num=None, batch_size=None):
    if batch_size is None:
        arch = model.meta_data['arch']
        if arch not in archs_batch_sizes:
            arch = 'default'
        batch_size = archs_batch_sizes[arch]
    dataset = get_dataset_trojai(model)
    if sample_num:
        dataset = sample_dataset(dataset, portion=sample_num)
    
    return get_ood_loader(custom_in_dataset=dataset,
                          out_dataset=out_dataset,
                          out_transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
                          batch_size=batch_size, sample=False)
