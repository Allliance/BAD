import torch

_mean = (0.5, 0.5, 0.5)
_std = (0.5, 0.5, 0.5)

mu = torch.tensor(_mean).view(3,1,1).cuda()
std = torch.tensor(_std).view(3,1,1).cuda()

class Model(nn.Module):
    def __init__(self, backbone, normalize=True):
        super(Model, self).__init__()
        self.backbone = backbone
        self.norm = lambda x: ( x - mu ) / std

    def forward(self, x):
        norm_x = self.norm(x)
        out = self.backbone(norm_x)
        return out