import random
import torch
import torchvision
from torchvision import transforms
from .datasets import SingleLabelDataset, GaussianDataset

ROOT = '~/data'

def sample_dataset(dataset, lim=2000):
    indices = random.sample(list(range(len(dataset))), k=lim)
    data = [dataset.data[x] for x in indices]
    dataset.data = data
    return dataset


def get_ood_loader(out='cifar100', sample=True):
    transform = transforms.Compose(
        [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])
    
    InDataset = SingleLabelDataset(1, 
                      torchvision.datasets.CIFAR10(root=ROOT, train=False,transform=transform, download=True))
    if out == 'SVHN':
        OutDataset = SingleLabelDataset(0, torchvision.datasets.SVHN(root=ROOT, split='test', download=True, transform=transform))
    elif out == 'mnist':
        transform_out = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
        ])
        OutDataset = SingleLabelDataset(0, torchvision.datasets.MNIST(root=ROOT, train=False, download=True, transform=transform_out))

    elif out=='cifar100':
        OutDataset = SingleLabelDataset(0, torchvision.datasets.CIFAR100(root=ROOT, train=False, download=True, transform=transform))
    elif out == 'gaussian':
        OutDataset = GaussianDataset(0)
    elif out == 'fmnist':
        transform_out = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor() 
        ])
        OutDataset = SingleLabelDataset(0, torchvision.datasets.FashionMNIST(root=ROOT, train=False, download=True, transform=transform_out))
    else:
        raise NotImplementedError

    if sample:
        InDataset = sample_dataset(InDataset)
        OutDataset = sample_dataset(OutDataset)

    merged_dataset = torch.utils.data.ConcatDataset([InDataset, OutDataset])
    testloader = torch.utils.data.DataLoader(merged_dataset, batch_size=256,
                                         shuffle=True)
    
    # Sanity Check
    next(iter(get_ood_loader(dataset)))
    
    return testloader

def get_cls_loader(dataset='cifar10', sample=False):
    if dataset == 'cifar10':
        transform = transforms.Compose(
            [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            ])
        cifar = torchvision.datasets.CIFAR10(root=ROOT, train=False, download=True, transform=transform)
        if sample:
            cifar = sample_dataset(cifar)
        testloader = torch.utils.data.DataLoader(cifar, batch_size=256,
                                                shuffle=False)
    else:
        raise NotImplementedError
    return testloader