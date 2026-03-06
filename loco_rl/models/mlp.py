import torch.nn as nn
from .activation import get_activation

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="elu", final_layer_activation=None):
        super().__init__()
        layers = []
        activation_func = get_activation(activation)
        prev_dim = input_dim
        if hidden_dims is not None:
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(activation_func)
                prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        if final_layer_activation is not None:
            layers.append(get_activation(final_layer_activation))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

    def reset(self, dones=None):
        pass

