# For OOD Data: 1 is for ID, 0 is for OOD
OUT_LABEL = 0
IN_LABEL = 1
# For Model Dataset: 1 is for Clean, 0 is for Bad
CLEAN_LABEL = 1
BACKDOOR_LABEL = 0

TINY_IMAGENET_ROOT = "/kaggle/input/tinyimagenet/"

SMALL_ARCHS = ['squeezenetv1_1', 'squeezenetv1_0', 'shufflenet1_0',
                'shufflenet1_5', 'shufflenet2_0', 'googlenet']

TROJAI_ROOT_DICT = {
    0: {
    'train': ['/kaggle/input/trojai-r0-train-p0/root/models'],
    'test': ['/kaggle/input/trojai-r0-train-p1/models'],
    },
    1: {
        'train': 
        {
        'folders': [f'/kaggle/input/trojai-r1-all-train-p{i}/models' for i in range(11)],
        'metadata': '/kaggle/input/trojai-r1-all-train-p0/metadata.csv',
        },
        'test': {
            'folders': ['/kaggle/input/trojai-r1-test/models'],
            'metadata': '/kaggle/input/trojai-r1-test/models/metadata.csv',
        }
    },
    2: {
    'train': 
        {
        'folders': [f'/kaggle/input/trojai-r2-all-train-part{i}/models' for i in range(12)],
        'metadata': '/kaggle/input/trojai-r2-all-train-part0/metadata.csv',
        },
    'test': {
        'folders': ['/kaggle/input/trojai-r2-test-p0/models', '/kaggle/input/trojai-r2-train-p1/models'],
        'metadata': '/kaggle/input/trojai-r2-test-p0/models/metadata.csv',
    }
    },
    3: {
    'train': 
        {
        'folders': [f'/kaggle/input/trojai-r3-all-train-part{i}/models' for i in range(11)],
        'metadata': '/kaggle/input/trojai-r3-all-train-part0/metadata.csv',
        },
    'test': {
        'folders': ['/kaggle/input/trojai-r3-test-p0/models', '/kaggle/input/trojai-r3-train-p1/models'],
        'metadata': '/kaggle/input/trojai-r3-test-p0/models/metadata.csv',
    }
    },
    4: {
    'train': 
        {
        'folders': [f'/kaggle/input/trojai-r4-all-train-part{i}/models' for i in range(11)],
        'metadata': '/kaggle/input/trojai-r4-all-train-part0/metadata.csv',
        },
    'test': {
        'folders': ['/kaggle/input/trojai-r4-test-p0/models', '/kaggle/input/trojai-r4-train-p1/models'],
        'metadata': '/kaggle/input/trojai-r4-test-p0/models/metadata.csv',
    },
    11: {
    'train':
        {
        'folders': ['/kaggle/input/trojai-round11-train-p1/models', '/kaggle/input/trojai-round11-all-train-part0/models'],
        'metadata': '/kaggle/input/trojai-round11-all-train-part0/models/metadata.csv',
        },
    'test': {
        'folders': ['/kaggle/input/trojai-r11-test-p0/models', '/kaggle/input/trojai-r11-train-p1/models'],
        'metadata': '/kaggle/input/trojai-r11-test-p0/models/metadata.csv',
    },
    }
    },
}

CLEAN_ROOT_DICT = {
'a2o':{
    'cifar10': {
    'resnet': '/kaggle/input/clean-resnet18-120models-dataset',
    'preact': '/kaggle/input/cleanset-preact',
    'vit': '/kaggle/input/vitb16-cifar10-allmodels/models/clean',
    },
    'cifar100': {
        'resnet': '/kaggle/input/cifar100-renset18-all-models/models/clean',
        'preact': '/kaggle/input/cifar100-preactresnet18-allmodels/models/clean',
    },
    'mnist': {
        'resnet': '/kaggle/input/clean-testset-resnet-mnist/models',
        'preact': '/kaggle/input/clean-preactresnet18-mnist-120models-dataset',
    },
    'gtsrb': {
        'resnet': '/kaggle/input/gtsrb-renset18-all-models/models/clean',
        'preact': '/kaggle/input/gtsrb-preactresnet18-cleans-bads/models/clean',
    },
    'celeba': {
        'resnet': '',
        'preact': '',
    },
    'pubfig': {
        'resnet': '/kaggle/input/pubfig-resnet-allmodels/models/clean',
        'preact': '/kaggle/input/pubfig-preact-allmodels/models/clean',
    }
},
'a2a':{
    'cifar10': {
        
    }
}
}

CLEAN_ADV_ROOT_DICT = {
    
}

BAD_ROOT_DICT = {
'a2o':{
    'cifar10': {
    'resnet': '/kaggle/input/backdoored-resnet18-120models-6attack-dataset',
    'preact': '/kaggle/input/badset-preact',
    'vit': '/kaggle/input/vitb16-cifar10-allmodels/models'
    },
    'mnist': {
        'resnet': '/kaggle/input/bad-testset-resnet-mnist/models',
        'preact': '/kaggle/input/backdoored-preactresnet18-mnist-100models-5attack',
    },
    'cifar100': {
        'resnet': '/kaggle/input/cifar100-renset18-all-models/models',
        'preact': '/kaggle/input/cifar100-preactresnet18-allmodels/models',
    },
    'gtsrb': {
        'resnet': '/kaggle/input/gtsrb-renset18-all-models/models',
        'preact': '/kaggle/input/gtsrb-preactresnet18-cleans-bads/models',
    },
    'celeba': {
        'resnet': '',
        'preact': '',
    },
    'pubfig': {
        'resnet': '/kaggle/input/pubfig-resnet-allmodels/models',
        'preact': '/kaggle/input/pubfig-preact-allmodels/models',
    }
},
'a2a':{
    
}
}

BAD_ADV_ROOT_DICT = {
    
}

# Number of classes
num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist': 10,
    'pubfig': 50,
    'fmnist': 10,
    'gtsrb': 43,
    'celeba': 8,
}

# Normalizations
NORM_MEAN = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4865, 0.4409),
    'mnist': (0.5, 0.5, 0.5),
    'pubfig': (0.485, 0.456, 0.406),
    'gtsrb': (0, 0, 0),
    'celeba': (0, 0, 0),
}

NORM_STD = {
    'cifar10': (0.247, 0.243, 0.261),
    'cifar100': (0.2673, 0.2564, 0.2762),
    'mnist': (0.5, 0.5, 0.5),
    'pubfig': (0.229, 0.224, 0.225),
    'gtsrb': (1, 1, 1),
    'celeba': (1, 1, 1),
}

