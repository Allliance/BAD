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
    if dataset in ['cifar10', 'cifar100', 'gtsrb']:
        return normal_transform
    elif dataset == 'mnist':
        return bw_transform
    else:
        raise NotImplementedError


def get_ood_loader(in_dataset, out='cifar100', sample=True, sample_num=2000, in_label=1,
                   out_label=0, batch_size=256, in_source='train', out_filter_labels=[],
                   custom_ood_dataset=None):
    assert in_label != out_label
    assert out_label is not None
    assert in_source in ['train', 'test', None]
    
    # In-Distribution Dataset
    if in_source is not None:
        transform = get_transform(in_dataset)
        use_train = in_source == 'train'
        if in_dataset == 'cifar10':
            in_dataset = torchvision.datasets.CIFAR10(root=ROOT, train=use_train,transform=transform, download=True)
        elif in_dataset == 'cifar100':
            in_dataset = torchvision.datasets.CIFAR100(root=ROOT, train=use_train,transform=transform, download=True)
        elif in_dataset == 'gtsrb':
            in_dataset = GTSRB(train=use_train,transform=transform, download=True)
        elif in_dataset == 'mnist':
            in_dataset = torchvision.datasets.MNIST(root=ROOT, train=use_train, download=True, transform=transform)
        else:
            raise NotImplementedError("In Distribution Dataset not implemented")
    
    # Out-Distribution Dataset
    if custom_ood_dataset is not None:
        out_dataset = custom_ood_dataset
    elif out == 'SVHN':
        out_dataset = torchvision.datasets.SVHN(root=ROOT, split='test', download=True, transform=normal_transform)
    elif out == 'mnist':
        out_dataset = torchvision.datasets.MNIST(root=ROOT, train=False, download=True, transform=bw_transform)
    elif out=='cifar100':
        out_dataset = torchvision.datasets.CIFAR100(root=ROOT, train=False, download=True, transform=normal_transform)
    elif out == 'gaussian':
        out_dataset = GaussianDataset(out_label, num_samples=sample_num)
    elif out == 'blank':
        out_dataset = BlankDataset(out_label, color=0, num_samples=sample_num)
    elif out == 'fmnist':
        out_dataset = torchvision.datasets.FashionMNIST(root=ROOT, train=False, download=True, transform=bw_transform)
    elif out == 'rot':
        out_dataset = deepcopy(in_dataset)
        out_dataset.transform = transforms.Compose([lambda x: rotate(x, 90), out_dataset.transform])
    else:
        raise NotImplementedError

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
    if transforms_list:
        transform = transforms.Compose(transforms_list)
    else:
        transform = get_transform(dataset)
    if dataset == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root=ROOT, train=train, download=True, transform=transform)
    elif dataset == 'mnist':
        test_dataset = torchvision.datasets.MNIST(root=ROOT, train=train, download=True, transform=transform)
    elif dataset == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100(root=ROOT, train=train, download=True, transform=transform)
    elif dataset == 'gtsrb':
        test_dataset = GTSRB(train=train,transform=transform, download=True)
    else:
        raise NotImplementedError

    if sample_portion < 1:
        test_dataset = sample_dataset(test_dataset, portion=sample_portion)
    
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Sanity Check
    next(iter(testloader))

    return testloader
