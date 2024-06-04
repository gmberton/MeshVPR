# Code from Optimal Transport Aggregation for Visual Place Recognition https://arxiv.org/abs/2311.15937

import torch
import torchvision.transforms as tfm


class SaladWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("serizba/salad", "dinov2_salad")
        self.resize = tfm.Resize([322, 322], antialias=True)
    def forward(self, images):
        images = self.resize(images)
        return self.model(images)
