import random
import torch
import torchvision
from torchvision import transforms
from BAD.data.datasets import SingleLabelDataset, GaussianDataset, BlankDataset
from torch.utils.data import Subset
from collections import defaultdict
from copy import deepcopy

ROOT = '~/data'

# Transforms
normal_transform = transforms.Compose(
    [
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    ])
bw_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
    ])

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


def get_ood_loader(in_dataset, out='cifar100', sample=True, sample_num=2000, in_label=1,
                   out_label=0, batch_size=256, in_source='train', out_filter_labels=[]):
    assert in_label != out_label
    assert out_label is not None
    assert in_source in ['train', 'test', None]
    
    # In-Distribution Dataset
    if in_source is not None:
        use_train = in_source == 'train'
        if in_dataset == 'cifar10':
            in_dataset = torchvision.datasets.CIFAR10(root=ROOT, train=use_train,transform=normal_transform, download=True)
        elif in_dataset == 'mnist':
            in_dataset = torchvision.datasets.MNIST(root=ROOT, train=use_train, download=True, transform=bw_transform)
        else:
            raise NotImplementedError("In Distribution Dataset not implemented")
        if in_label is not None:
            in_dataset = SingleLabelDataset(in_label, in_dataset)
    
    
    # Out-Distribution Dataset
    if out == 'SVHN':
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
        out_dataset.transform = transforms.Compose([transforms.RandomRotation(90), out_dataset.transform])
    else:
        raise NotImplementedError

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

def get_transform(dataset):
    if dataset == 'cifar10':
        return normal_transform
    elif dataset == 'mnist':
        return bw_transform
    else:
        raise NotImplementedError

def get_cls_loader(dataset='cifar10', train=False, sample_portion=0.2, batch_size=256, transforms_list=None):
    if transforms_list:
        transform = transforms.Compose(transforms_list)
    else:
        transform = get_transform(dataset)
    if dataset == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root=ROOT, train=train, download=True, transform=transform)
    elif dataset == 'mnist':
        test_dataset = torchvision.datasets.MNIST(root=ROOT, train=train, download=True, transform=transform)
    else:
        raise NotImplementedError

    if sample_portion < 1:
        test_dataset = sample_dataset(test_dataset, portion=sample_portion)
    
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Sanity Check
    next(iter(testloader))

    return testloader
