import torch.nn as nn

class Memory(nn.Module):
    def __init__(self, memory_type, input_dim, hidden_size, num_layers):
        super().__init__()
        rnn_cls = nn.GRU if memory_type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
    
    def forward(self, input, hidden_states=None):
        if len(input.shape) == 3:
            # Batch mode during training
            out, _ = self.rnn(input, hidden_states)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
            out = out.squeeze(0)
        return out
    
    def reset(self, dones=None):
        if self.hidden_states is not None:
            # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
            if dones is None:
                self.hidden_states = None
            else:
                for state in self.hidden_states:
                    state[..., dones, :] = 0.0

    def get_hidden_states(self):
        return self.hidden_states


