import random
import torch
import torchvision
from torchvision import transforms
from .datasets import SingleLabelDataset, GaussianDataset
from torch.utils.data import Subset
from collections import defaultdict

ROOT = '~/data'

def sample_dataset(dataset, portion=0.1, balanced=True):
    
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



def get_ood_loader(out='cifar100', sample=True, in_label=1, out_label=0, batch_size=256):
    assert in_label != out_label
    assert out_label is not None
    transform = transforms.Compose(
        [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])
    in_dataset = torchvision.datasets.CIFAR10(root=ROOT, train=False,transform=transform, download=True)
    if in_label is not None:
        in_dataset = SingleLabelDataset(1, in_dataset)
    if out == 'SVHN':
        out_dataset = SingleLabelDataset(0, torchvision.datasets.SVHN(root=ROOT, split='test', download=True, transform=transform))
    elif out == 'mnist':
        transform_out = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
        ])
        out_dataset = torchvision.datasets.MNIST(root=ROOT, train=False, download=True, transform=transform_out)
    elif out=='cifar100':
        out_dataset = torchvision.datasets.CIFAR100(root=ROOT, train=False, download=True, transform=transform)
    elif out == 'gaussian':
        out_dataset = GaussianDataset(out_label)
    elif out == 'fmnist':
        transform_out = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor() 
        ])
        out_dataset = torchvision.datasets.FashionMNIST(root=ROOT, train=False, download=True, transform=transform_out)
    else:
        raise NotImplementedError

    out_dataset = SingleLabelDataset(out_label, out_dataset)

    if sample:
        in_dataset = sample_dataset(in_dataset)
        out_dataset = sample_dataset(out_dataset)

    merged_dataset = torch.utils.data.ConcatDataset([in_dataset, out_dataset])
    
    
    testloader = torch.utils.data.DataLoader(merged_dataset, batch_size=batch_size,
                                         shuffle=True)
    
    # Sanity Check
    next(iter(testloader))
    
    return testloader

def get_transform(dataset):
    if dataset == 'cifar10':
        transform = transforms.Compose(
            [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            ])
    else:
        raise NotImplementedError
    return transform

def get_cls_loader(dataset='cifar10', sample_portion=0.2, batch_size=256, transforms_list=None):
    if transforms_list:
        transform = transforms.Compose(transforms_list)
    else:
        transform = get_transform(dataset)
    if dataset == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(root=ROOT, train=False, download=True, transform=transform)
    else:
        raise NotImplementedError

    if sample_portion < 1:
        test_dataset = sample_dataset(test_dataset, portion=sample_portion)
    
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Sanity Check
    next(iter(testloader))

    return testloader