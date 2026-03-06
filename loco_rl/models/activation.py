import torch.nn as nn

import torch.nn as nn

def get_activation(act_name):
    activations = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "crelu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid()
    }
    
    if act_name in activations:
        return activations[act_name]
    else:
        raise ValueError(f"Invalid activation function: {act_name}")
