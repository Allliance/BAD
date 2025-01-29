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
from BAD.trojai.dataset import ExampleDataset
from BAD.data.loaders import get_ood_loader
from BAD.data.utils import sample_dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

archs_batch_sizes = {
    'squeezenetv1_1': 128,
    'squeezenetv1_0': 128,
    'shufflenet1_0': 128,
    'shufflenet1_5': 128,
    'shufflenet2_0': 128,
    # 'googlenet': 128,
    
    'default': 64,
    'resnet18': 128,
    
    'resnet50': 32,
    'resnet101': 32,
    'densenet121': 32,
    'inceptionv3': 64,
    'vgg19bn': 32,
    'vgg16bn': 32,
    'vgg13bn': 16,
    'vgg11bn': 32,
    'wideresnet101': 16,
    'wideresnet50': 32,
    'resnet152': 16,
    'densenet201': 16,
    'densenet161': 8,
    'densenet169':16,
}

def load_model(model_data, **model_kwargs):
    model_path = model_data['model_path']
    arch = model_data['arch']
    num_classes = model_data['num_classes']
    print("Loading a", arch)
    
    try:
        net = torch.load(model_path, map_location=device)
    except Exception as e:
        print("facing problems while loading this model", str(e))
        return None
    
    
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
    rnd = model.meta_data['rnd']
    # if rnd < 3:
    #     example_data_path = 'example_data'
    # elif rnd == 4:
    #     example_data_path = 'clean_example_data'
    # else:
    #     example_data_path = 'clean-example-data'
    if rnd == 1:
        example_data_path = 'clean-example-data'
    else:
        example_data_path = 'example_data'
        
    return ExampleDataset(root_dir=os.path.join(os.path.dirname(model.meta_data['model_path']),
                          example_data_path), use_bgr=model.meta_data['bgr'], rnd=rnd)


def get_sanityloader_trojai(model, batch_size=None):
    if batch_size is None:
        arch = model.meta_data['arch']
        if arch not in archs_batch_sizes:
            arch = 'default'
        batch_size = archs_batch_sizes[arch]
    return DataLoader(get_dataset_trojai(model), shuffle=True, batch_size=batch_size)


def get_oodloader_trojai(model, out_dataset, sample_num=None, batch_size=None, **kwargs):
    if batch_size is None:
        arch = model.meta_data['arch']
        if arch not in archs_batch_sizes:
            arch = 'default'
        batch_size = archs_batch_sizes[arch]
    dataset = get_dataset_trojai(model)
    if sample_num:
        sample_num = min(sample_num, len(dataset))
        dataset = sample_dataset(dataset, portion=sample_num)
    
    return get_ood_loader(custom_in_dataset=dataset,
                          out_dataset=out_dataset,
                          out_transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
                          batch_size=batch_size, **kwargs)
