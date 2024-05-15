import random
import torch
import torchvision
from torchvision import transforms
from BAD.data.datasets.custom_datasets import MixedDataset, SingleLabelDataset, DummyDataset, NegativeDataset
from BAD.data.datasets.gtsrb import GTSRB
from BAD.data.datasets.pubfig import PubFig
from torch.utils.data import Subset
from collections import defaultdict
from copy import deepcopy
from torchvision.datasets import ImageFolder
from BAD.data.transforms import *
from BAD.data.utils import sample_dataset, filter_labels
from BAD.constants import OUT_LABEL, IN_LABEL
import os

ROOT = '~/data'
negatives = ['rot', 'mixup', 'cutpaste', 'distort', 'elastic']

def get_transform(dataset):
  if dataset in ['cifar10', 'cifar100', 'gtsrb', 'SVHN']:
      return normal_transform
  elif dataset in ['fmnist', 'mnist']:
      return bw_transform
  elif dataset in ['gaussian', 'blank']:
      return None
  elif dataset in ['pubfig']:
      return hr_transform
  else:
      raise NotImplementedError

def get_dataset(name, transform=None, train=False,
                dummy_params={}, download=False, in_dataset=None, **kwargs):
    '''
    Available datasets:
    - 'cifar10'
    - 'cifar100'
    - 'gtsrb'
    - 'mnist'
    - 'pubfig'
    - 'fmnist'
    - 'SVHN'
    - 'gaussian'
    - 'blank'
    - 'uniform'
    '''
    
    if transform is None:
        transform = get_transform(name)
        
        # Make sure ID samples and OOD samples have the same size
        if in_dataset is not None:
            id_sample = in_dataset[0][0]
            size = id_sample.size()[-1]
            channels = id_sample.size()[0]
            new_transforms = [transforms.ToPILImage()]
            print(channels, size)
            
            if channels == 1:
                new_transforms.append(transforms.Grayscale())
            elif channels == 3 and name in ['mnist', 'fmnist']:
                new_transforms.append(transforms.Grayscale(3))
            new_transforms.append(transforms.Resize((size, size)))
            new_transforms.append(transforms.ToTensor())
            
            transform = transforms.Compose([transform, transforms.Compose(new_transforms)])
            print(transform)
    try:
        if name == 'SVHN':
            return torchvision.datasets.SVHN(root=ROOT, split='train' if train else 'test', download=download, transform=transform)
        elif name == 'mnist':
            return torchvision.datasets.MNIST(root=ROOT, train=train, download=download, transform=transform)
        elif name == 'fmnist':
            return torchvision.datasets.FashionMNIST(root=ROOT, train=train, download=download, transform=transform)
        elif name == 'cifar10':
            return torchvision.datasets.CIFAR10(root=ROOT, train=train,transform=transform, download=download)
        elif name =='cifar100':
            return torchvision.datasets.CIFAR100(root=ROOT, train=train, download=download, transform=transform)
        elif name == 'gtsrb':
            return GTSRB(train=train,transform=transform, download=download)
        elif name == 'pubfig':
            return PubFig(train=train, transform=transform)
        elif name in ['gaussian', 'blank', 'uniform']:
            label = dummy_params.get('label', OUT_LABEL)
            return DummyDataset(pattern=name, label=label, pattern_args=dummy_params)
        elif os.path.isdir(name):
            return ImageFolder(name, transform=hr_transform)
        else:
            raise NotImplementedError
    except Exception as e:
        if not download:
            return get_dataset(name, transform, train, dummy_params, download=True)
        else:
            raise e

def get_ood_loader(in_dataset=None, out_dataset=None,
                   sample_num=None, batch_size=256,
                   in_transform=None, out_transform=None,
                   custom_ood_dataset=None, custom_in_dataset=None,
                   out_in_ratio=1, final_ood_trans=None, **kwargs):
    assert in_dataset is not None or custom_in_dataset is not None or custom_ood_dataset is not None or out_dataset is not None
    
    # In-Distribution Dataset
    if custom_in_dataset is not None:
        in_dataset = custom_in_dataset
    else:
        in_dataset = get_dataset(in_dataset, in_transform, trian=True, **kwargs)
    
    in_transform = in_dataset.transform
    
    # Sampling - ID
    if in_dataset is not None and sample_num is not None:
        in_dataset = sample_dataset(in_dataset, portion=sample_num)

    # Labeling - ID
    if in_dataset is not None:
        in_dataset = SingleLabelDataset(IN_LABEL, in_dataset)

    # Out-Distribution Dataset
    if custom_ood_dataset is None:
        if isinstance(out_dataset, str):
            out_dataset = [out_dataset]
        all_out_datasets = []
        neg_datasets = [item for item in out_dataset if item in negatives]
        if neg_datasets:
            all_out_datasets.append(NegativeDataset(base_dataset=in_dataset, label=OUT_LABEL,
                                        neg_transformations=neg_datasets, **kwargs))
        for out in out_dataset:
            if out not in negatives:
                all_out_datasets.append(get_dataset(out, out_transform, train=True,
                                                    in_dataset=in_dataset, in_transform=in_transform, **kwargs))
        length = int(out_in_ratio * len(in_dataset))
        out_dataset = MixedDataset(all_out_datasets, label=OUT_LABEL, length=length, transform=out_transform)
    else:
        out_dataset = custom_ood_dataset
        
    if out_dataset and final_ood_trans:
        out_dataset = NegativeDataset(base_dataset=out_dataset, label=OUT_LABEL,
                                          neg_transformations=[final_ood_trans], **kwargs)

    # Labeling - OOD
    if out_dataset is not None:
        out_dataset = SingleLabelDataset(OUT_LABEL, out_dataset)
    
    if in_dataset is not None and out_dataset is not None:
        final_dataset = torch.utils.data.ConcatDataset([in_dataset, out_dataset])
    elif in_dataset is not None:
        final_dataset = in_dataset
    elif out_dataset is not None:
        final_dataset = out_dataset
    else:
        raise ValueError("Empty dataset error occured")
    
    testloader = torch.utils.data.DataLoader(final_dataset, batch_size=batch_size,
                                         shuffle=True)
    
    # Sanity Check
    next(iter(testloader))
    
    return testloader

def get_cls_loader(dataset, train=False, sample_portion=0.2, batch_size=256, transforms_list=None):
    transform = None
    if transforms_list:
        transform = transforms.Compose(transforms_list)
    if isinstance(dataset, str):
        test_dataset = get_dataset(dataset, transform, train)
    else:
        test_dataset = dataset
    if sample_portion < 1:
        test_dataset = sample_dataset(test_dataset, portion=sample_portion)
    
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # Sanity Check
    next(iter(testloader))

    return testloader
