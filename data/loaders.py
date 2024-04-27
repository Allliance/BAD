import random
import torch
import torchvision
from torchvision import transforms
from BAD.data.datasets import SingleLabelDataset, GaussianDataset, BlankDataset
from BAD.data.gtsrb import GTSRB
from torch.utils.data import Subset
from collections import defaultdict
from copy import deepcopy
from torchvision.transforms.functional import rotate
from BAD.data.transforms import normal_transform, bw_transform

ROOT = '~/data'

def sample_dataset(dataset, portion=0.1, balanced=True):
    if portion>1:
        portion = portion / len(dataset)
    if not balanced:
        indices = random.sample(range(len(dataset)), int(portion * len(dataset)))
    # It is assumed that the dataset has labels
    else:
        indices = []
        labels = [y for _, y in dataset]
        unique_labels = list(set(labels))
        labels_indices = defaultdict(lambda : [])
        
        for i, label in enumerate(labels):
            labels_indices[label].append(i)
            
        for label in unique_labels:
            indices += random.sample(labels_indices[label], int(portion * len(labels_indices[label])))
        
    return Subset(dataset, indices)

def filter_labels(dataset, labels):
    indices = [i for i, (_, y) in enumerate(dataset) if y not in labels]
    return Subset(dataset, indices)

def get_transform(dataset):
  if dataset in ['cifar10', 'cifar100', 'gtsrb', 'SVHN']:
      return normal_transform
  elif dataset in ['fmnist', 'mnist']:
      return bw_transform
  elif dataset in ['gaussian', 'blank']:
      return None
  else:
      raise NotImplementedError

def get_dataset(name, transform=None, train=False, custom_label=None, sample_num=3000):
    if transform is None:
        transform = get_transform(dataset)

    if dataset == 'SVHN':
        return torchvision.datasets.SVHN(root=ROOT, split='train' if train else 'test', download=True, transform=transform)
    elif dataset == 'mnist':
        return torchvision.datasets.MNIST(root=ROOT, train=train, download=True, transform=transform)
    elif dataset == 'fmnist':
        return torchvision.datasets.FashionMNIST(root=ROOT, train=train, download=True, transform=transform)
    elif dataset == 'cifar10':
        in_dataset = torchvision.datasets.CIFAR10(root=ROOT, train=train,transform=transform, download=True)
    elif dataset =='cifar100':
        return torchvision.datasets.CIFAR100(root=ROOT, train=train, download=True, transform=transform)
    elif dataset == 'gtsrb':
        in_dataset = GTSRB(train=train,transform=transform, download=True)
    elif dataset == 'gaussian':
        return GaussianDataset(out_label, num_samples=sample_num)
    elif dataset == 'blank':
        return BlankDataset(out_label, color=0, num_samples=sample_num)
    else:
        raise NotImplementedError

def get_ood_loader(in_dataset='cifar10', out_dataset='cifar100', sample=True, sample_num=2000, in_label=1,
                   out_label=0, batch_size=256, in_source='train', out_filter_labels=[],
                   custom_in_dataset=None, custom_ood_dataset=None, in_transform=None, out_transform=None):
    assert in_label != out_label
    assert out_label is not None
    assert in_source in ['train', 'test', None]
    
    # In-Distribution Dataset
    if custom_in_dataset is not None:
        in_dataset = custom_in_dataset
    else:
        if in_source is not None:
            in_dataset = get_dataset(in_dataset, in_transform, in_source == 'train', custom_label=in_label)
    
    # Out-Distribution Dataset
    if custom_ood_dataset is None:
        if out_dataset == 'rot':
            out_dataset = deepcopy(in_dataset)
            out_dataset.transform = transforms.Compose([lambda x: rotate(x, 90), out_dataset.transform])    
        else:
            out_dataset = get_ood_dataset(out_dataset, out_transform,
            in_dataset, out_label=out_label, sample_num=sample_num)
    else:
        out_dataset = custom_ood_dataset

    # Sampling

    if in_label is not None:
        in_dataset = SingleLabelDataset(in_label, in_dataset)
        
    if out_filter_labels:
        out_dataset = filter_labels(out_dataset, out_filter_labels)

    out_dataset = SingleLabelDataset(out_label, out_dataset)

    if sample:
        out_dataset = sample_dataset(out_dataset, portion=sample_num)
    
    final_dataset = out_dataset
    
    # Concat Dataset
    if in_source is not None:
        if sample:
            in_dataset = sample_dataset(in_dataset, portion=len(out_dataset))
        
        final_dataset = torch.utils.data.ConcatDataset([in_dataset, out_dataset])
    
    testloader = torch.utils.data.DataLoader(final_dataset, batch_size=batch_size,
                                         shuffle=True)
    
    # Sanity Check
    next(iter(testloader))
    
    return testloader

def get_cls_loader(dataset, train=False, sample_portion=0.2, batch_size=256, transforms_list=None):
    transform = None
    if transforms_list:
        transform = transforms.Compose(transforms_list)
    if type(dataset) == str:
        test_dataset = get_dataset(dataset, transform, train)
    else:
        test_dataset = dataset
    if sample_portion < 1:
        test_dataset = sample_dataset(test_dataset, portion=sample_portion)
    
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Sanity Check
    next(iter(testloader))

    return testloader
