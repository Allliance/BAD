import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, backbone, device=None, normalize=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        super(Model, self).__init__()

        mu = torch.tensor(mean).view(3,1,1)
        std = torch.tensor(std).view(3,1,1)
        
        self.backbone = backbone
        
        if device:
            mu = mu.to(device)
            std = std.to(device)
        self.norm = lambda x: ( x - mu ) / std

    def forward(self, x):
        norm_x = self.norm(x)
        out = self.backbone(norm_x)
        return out