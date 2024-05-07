import random
import torch
import torchvision
from torchvision import transforms
from BAD.data.datasets.custom_datasets import MixedDataset, SingleLabelDataset, DummyDataset
from BAD.data.neg_transformations.rotation import RotationDataset
from BAD.data.neg_transformations.mixup import MixupDataset
from BAD.data.neg_transformations.cutpaste import CutPasteDataset
from BAD.data.neg_transformations.distort import DistortDataset
from BAD.data.neg_transformations.elastic import ElasticDataset
from BAD.data.datasets.gtsrb import GTSRB
from torch.utils.data import Subset
from collections import defaultdict
from copy import deepcopy
from torchvision.transforms.functional import rotate
from BAD.data.transforms import normal_transform, bw_transform
from BAD.data.utils import sample_dataset, filter_labels

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

def get_dataset(name, transform=None, train=False, dummy_params={}, download=False):
    '''
    Available datasets:
    - 'cifar10'
    - 'cifar100'
    - 'gtsrb'
    - 'mnist'
    - 'fmnist'
    - 'SVHN'
    - 'gaussian'
    - 'blank'
    - 'uniform'
    '''
    if transform is None:
        transform = get_transform(name)
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
        elif name in ['gaussina', 'blank', 'uniform']:
            return DummyDataset(pattern=name, label=dummy_params['label'], pattern_args=dummy_params)
        else:
            raise NotImplementedError
    except Exception as _:
        if not download:
            return get_dataset(name, transform, train, dummy_params, download=True)
        else:
            raise ValueError("Error occured during loading datasets")

def get_negative_augmentation(name, dataset, label, transform=None, **kwargs):
    if name == 'rot':
        return RotationDataset(dataset, label=label, transform=transform, **kwargs)
    elif name == 'mixup':
        return MixupDataset(dataset, label=label, transform=transform, **kwargs)
    elif name == 'cutpaste':
        return CutPasteDataset(dataset, label=label, transform=transform, **kwargs)
    elif name == 'distort':
        return DistortDataset(dataset, label=label, transform=transform, **kwargs)
    elif name == 'elastic':
        return ElasticDataset(dataset, label=label, transform=transform, **kwargs)
    else:
        raise NotImplementedError

def get_ood_loader(in_dataset=None, out_dataset=None, sample=True, sample_num=2000, in_label=1,
                   out_label=0, batch_size=256, in_source='train', out_filter_labels=[],
                   in_transform=None, out_transform=None, custom_ood_dataset=None, custom_in_dataset=None,
                    balanced=True, balanced_sample=False, out_portion=1, **kwargs):
    assert in_label != out_label
    assert out_label is not None
    assert in_source in ['train', 'test', None]
    assert in_dataset is not None or custom_in_dataset is not None or custom_ood_dataset is not None or out_dataset is not None
    
    # In-Distribution Dataset
    if custom_in_dataset is not None:
        in_dataset = custom_in_dataset
    else:
        if in_source is not None:
            in_dataset = get_dataset(in_dataset, in_transform, in_source == 'train', **kwargs)
    

    # Sampling - ID
    if in_dataset is not None and sample:
        in_dataset = sample_dataset(in_dataset, portion=sample_num, balanced=balanced_sample)

    # Out-Distribution Dataset
    if custom_ood_dataset is None:
        if isinstance(out_dataset, list):
            out_datasets = []
            for out in out_dataset:
                if out in negatives:
                    out_datasets.append(get_negative_augmentation(out, in_dataset, out_label, transform=out_transform, **kwargs))
                else:
                    out_datasets.append(get_dataset(out, out_transform, train=False, **kwargs))
            length = int(out_portion * len(in_dataset))
            out_dataset = MixedDataset(out_datasets, label=out_label, length=length,transform=out_transform)
        elif out_dataset in negatives:
            out_dataset = get_negative_augmentation(out_dataset, in_dataset, out_label, transform=out_transform, **kwargs)
        else:
            out_dataset = get_dataset(out_dataset, out_transform, in_dataset, **kwargs)
    else:
        out_dataset = custom_ood_dataset

    # Labeling - ID
    if in_label is not None and in_dataset is not None:
        in_dataset = SingleLabelDataset(in_label, in_dataset)

    # Labeling - OOD
    if out_dataset is not None and out_label is not None:
        out_dataset = SingleLabelDataset(out_label, out_dataset)
    
    # Sampling
        
    if out_filter_labels:
        out_dataset = filter_labels(out_dataset, out_filter_labels)

    if sample:
        out_dataset = sample_dataset(out_dataset, portion=sample_num, balanced=balanced_sample)
    if balanced and len(out_dataset) > len(in_dataset):
        out_dataset = sample_dataset(out_dataset, portion=len(in_dataset)/len(out_dataset), balanced=balanced_sample)
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
