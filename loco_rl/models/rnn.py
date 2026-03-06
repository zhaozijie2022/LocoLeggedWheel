import torch.nn as nn
from .memory_module import Memory
from .mlp import MLP


class RNN(nn.Module):
    def __init__(
        self, input_dim, hidden_dims, output_dim, activation="elu",
        rnn_memory_type='gru', rnn_hidden_size=256, rnn_num_layers=1):
        super().__init__()
        self.memory = Memory(rnn_memory_type, input_dim, rnn_hidden_size, rnn_num_layers)
        self.mlp = MLP(rnn_hidden_size, hidden_dims, output_dim, activation)
    
    def forward(self, x, hidden_states=None):
        rnn_out = self.memory(x, hidden_states=hidden_states)
        return self.mlp(rnn_out)
    
    def reset(self, dones=None):
        self.memory.reset(dones=dones)

    def get_hidden_states(self):
        return self.memory.get_hidden_states()
