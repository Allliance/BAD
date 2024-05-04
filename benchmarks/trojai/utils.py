import random
import torch
import pandas as pd
import warnings
import os

from torchvision.models import inception_v3
from BAD.models.base_model import BaseModel as Model
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

archs_batch_sizes = {
    'resnet50': 32,
    'densenet121': 32,
    'inceptionv3': 64,
}

def load_model(model_data, **model_kwargs):
    model_path = model_data['model_path']
    arch = model_data['arch']
    num_classes = model_data['num_classes']
    
    net = torch.load(model_path, map_location=device)
    
    if arch == 'inceptionv3':
        new_net = inception_v3(num_classes=num_classes)
        new_net.load_state_dict(deepcopy(net.state_dict()), strict=False)
        net = new_net
    
    feature_extractor = torch.nn.Sequential(*list(net.children())[:-1])
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = torch.nn.DataParallel(net)
#     net = torch.nn.DataParallel(net, device_ids = [0,1]).to(device)
    model = Model(net, feature_extractor=feature_extractor, **model_kwargs)
    model.to(device)
    model.eval()
    
    return model

def image_loader(fn):
    img = skimage.io.imread(fn)
    img = img.astype(dtype=np.float32)

    # perform center crop to what the CNN is expecting 224x224
    h, w, c = img.shape
    dx = int((w - 224) / 2)
    dy = int((w - 224) / 2)
    img = img[dy:dy + 224, dx:dx + 224, :]

    # convert to CHW dimension ordering
    img = np.transpose(img, (2, 0, 1))
    # convert to NCHW dimension ordering
    img = np.expand_dims(img, 0)
    # normalize the image matching pytorch.transforms.ToTensor()
    img = img / 255.0

    # convert image to a gpu tensor
    return torch.from_numpy(img)[0]


def get_dataset_trojai(model):
    return ExampleDataset(root_dir=os.path.join(os.path.dirname(model.meta_data['model_path'])
                                                   , 'example_data'), use_bgr=model.meta_data['bgr'])


def get_sanityloader_trojai(model, batch_size=None):
    return DataLoader(get_dataset_trojai(model), shuffle=True, batch_size=batch_size)


def get_oodloader_trojai(model, batch_size=None):
    if batch_size is None:
        batch_size = archs_batch_sizes[model.meta_data['arch']]
    dataset = get_dataset_trojai(model)
    return get_ood_loader(custom_in_dataset=dataset, out_dataset=out_dataset, shuffle=True, batch_size=batch_size, sample=False)
