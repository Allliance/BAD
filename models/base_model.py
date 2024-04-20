import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class BaseModel(nn.Module):
    def __init__(self, backbone, normalize=True,
                 mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261],
                 input_scalar=None, feature_extractor=None, meta_data=None):
        super(BaseModel, self).__init__()
        
        mu = torch.tensor(mean).view(3,1,1)
        std = torch.tensor(std).view(3,1,1)
        
        self.meta_data = meta_data
        if self.meta_data is None:
            self.meta_data = {}
        
        self.backbone = backbone
        
        self.input_scalar = input_scalar
        
        if device:
            mu = mu.to(device)
            std = std.to(device)
        self.do_norm = normalize
        self.norm = lambda x: ( x - mu ) / std
        self.feature_extractor = feature_extractor

    def get_features(self, x, normalize=False):
        if self.input_scalar is not None:
            x = x * self.input_scalar
        if self.do_norm:
            x = self.norm(x)
        features = self.feature_extractor(x)
        if normalize:
            features = nn.functional.normalize(features)
        return features

    def forward(self, x):
        if self.input_scalar is not None:
            x = x * self.input_scalar
        if self.do_norm:
            x = self.norm(x)
        out = self.backbone(x)
        return out