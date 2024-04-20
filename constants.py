CLEAN_ROOT_DICT = {
'cifar10': {
    'resnet': '/kaggle/input/clean-resnet18-120models-dataset',
    'preact': '/kaggle/input/cleanset-preact',
},
'mnist': {
    'resnet': '/kaggle/input/clean-resnet18-120models-dataset',
    'preact': '/kaggle/input/clean-preactresnet18-mnist-120models-dataset',
}}

BAD_ROOT_DICT = {
    'cifar10': {
            'resnet': '/kaggle/input/backdoored-resnet18-120models-6attack-dataset',
            'preact': '/kaggle/input/badset-preact',
            },
    'mnist': {
            'resnet': '/kaggle/input/backdoored-resnet18-120models-6attack-dataset',
            'preact': '/kaggle/input/backdoored-preactresnet18-mnist-100models-5attack',
            }
           }


# Normalizations
NORM_MEAN = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'mnist': (0.5, 0.5, 0.5),
}

NORM_STD = {
    'cifar10': (0.247, 0.243, 0.261),
    'mnist': (0.5, 0.5, 0.5),
}

