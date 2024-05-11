from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate
import torchvision
from torchvision.datasets import ImageFolder
import os
import random
import numpy as np
from BAD.data.datasets.cutpaste import CutPasteUnion
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import ElasticTransform
from torchvision.transforms import InterpolationMode

def get_mixup(**kwargs):
    mixup_alpha = kwargs.get('mixup_alpha', 0.3)
    imagenet_root = kwargs.get('imagenet_root')
    if imagenet_root is None:
        raise ValueError('imagenet_root must be provided for mixup')
    
    imagenet_dataset = ImageFolder(imagenet_root,
                    transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
    def mixup(image, imagenet_dataset, mixup_alpha):
        imagenet_idx = np.random.randint(len(imagenet_dataset))
        imagenet_img, _ = imagenet_dataset[imagenet_idx]

        mixed_img = (1 - mixup_alpha) * image + mixup_alpha * imagenet_img

        return mixed_img
    return lambda image: mixup(image, imagenet_dataset, mixup_alpha)

def get_elastic(**kwargs):
    p = kwargs.get('p', 1.0)
    alpha = kwargs.get('alpha', 100)
    sigma = kwargs.get('sigma', 50)
    alpha_affine = kwargs.get('alpha_affine', 100)
    interp = kwargs.get('interp', "bilinear") # or nearest
    
    # to_pil = transforms.ToPILImage()
    # elastic = transforms.Compose([ElasticTransform(alpha=alpha, sigma=sigma,
    #                            interpolation=InterpolationMode.BILINEAR if interp == 'bilinear' else InterpolationMode.NEAREST),
    #                               transforms.ToTensor()])
    # return lambda image: elastic(to_pil(image))
    elastic = A.Compose([A.ElasticTransform(alpha=alpha, p=p, sigma=sigma, alpha_affine=alpha_affine),
            ToTensorV2()])
    
    return lambda image: elastic(image=image.permute(1, 2, 0).numpy())['image']
    

def get_cutpaste(**kwargs):
    cutpaste = CutPasteUnion(transform=transforms.Compose([transforms.ToTensor(), ]))
    to_pil = transforms.ToPILImage()
    return lambda image: cutpaste(to_pil(image))

def get_distort(**kwargs):
    p = kwargs.get('p', 1.0)
    num_steps = kwargs.get('num_steps', 10)
    distort_limit = kwargs.get('distort_limit', 1)
    
    distort = A.Compose([A.GridDistortion(num_steps=num_steps, distort_limit=distort_limit, p=p),
        ToTensorV2()])
    return lambda image: distort(image=image.permute(1, 2, 0).numpy())['image']

def get_rot(**kwargs):
    return lambda image: torch.rot90(image, k=random.randint(1, 3), dims=(1, 2))