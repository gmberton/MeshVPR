# Code from https://github.com/gmberton/VPR-methods-evaluation

import torch

from vpr_models import sfrs, salad, convap, mixvpr, netvlad


def get_model(method):
    if method == "sfrs":
        model = sfrs.SFRSModel()
        model.desc_dim = 4096
    elif method == "netvlad":
        model = netvlad.NetVLAD()
        model.desc_dim = 4096
    elif method == "cosplace":
        model = torch.hub.load("gmberton/cosplace", "get_trained_model",
                               backbone="ResNet50", fc_output_dim=2048)
        model.desc_dim = 2048
    elif method == "mixvpr":
        model = mixvpr.get_mixvpr(descriptors_dimension=4096)
        model.desc_dim = 4096
    elif method == "convap":
        model = convap.get_convap(descriptors_dimension=4096)
        model.desc_dim = 4096
    elif method == "eigenplaces":
        model = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                               backbone="ResNet50", fc_output_dim=2048)
        model.desc_dim = 2048
    elif method == "salad":
        model = salad.SaladWrapper()
        model.desc_dim = 8448
    
    return model

