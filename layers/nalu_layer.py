from torch.nn import Sequential
from torch import nn
from models.nalu import NALU


class NALULayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers, hidden_shape):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.hidden_shape = hidden_shape
        layers = [ NALU(hidden_shape if n > 0 else in_dim, hidden_shape if n < n_layers - 1 else out_dim) for n in range(n_layers)]
        self.model = Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
