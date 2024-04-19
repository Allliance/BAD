import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class BaseModel(nn.Module):
    def __init__(self, backbone, normalize=True,
                 mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261],
                 input_scalar=None, feature_extractor=None, normalize_features=False):
        super(BaseModel, self).__init__()
        
        print(mean, std)
        
        mu = torch.tensor(mean).view(3,1,1)
        std = torch.tensor(std).view(3,1,1)
        
        self.backbone = backbone
        
        self.input_scalar = input_scalar
        
        if device:
            mu = mu.to(device)
            std = std.to(device)
        self.do_norm = normalize
        self.norm = lambda x: ( x - mu ) / std
        self.feature_extractor = feature_extractor
        self.normalize_features = normalize_features

    def get_features(self, x):
        if self.input_scalar is not None:
            x = x * self.input_scalar
        if self.do_norm:
            x = self.norm(x)
        features = self.feature_extractor(x)
        if self.normalize_features:
            features = nn.functional.normalize(features, p=2, dim=1)    
        return features

    def forward(self, x):
        if self.input_scalar is not None:
            x = x * self.input_scalar
        if self.do_norm:
            x = self.norm(x)
        out = self.backbone(x)
        return out