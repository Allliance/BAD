import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, backbone, device=None, normalize=True, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261], input_scalar=None):
        super(Model, self).__init__()
        
        mu = torch.tensor(mean).view(3,1,1)
        std = torch.tensor(std).view(3,1,1)
        
        self.backbone = backbone
        
        self.input_scalar = input_scalar
        
        if device:
            mu = mu.to(device)
            std = std.to(device)
        self.do_norm = normalize
        self.norm = lambda x: ( x - mu ) / std

    def get_features(self, x):
        if self.input_scalar is not None:
            x = x * self.input_scalar
        if self.do_norm:
            x = self.norm(x)
        features = self.backbone.get_features(x)
        return features

    def forward(self, x):
        if self.input_scalar is not None:
            x = x * self.input_scalar
        if self.do_norm:
            x = self.norm(x)
        out = self.backbone(x)
        return out