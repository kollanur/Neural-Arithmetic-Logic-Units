import torch
from torch import nn
from models.nac import NAC
from torch.nn import functional as F

class NALU(nn.Module):
    '''
    Class implementing Neural Arithmetic Logic Unit (NALU)
    with a small deviation from the original one described
    here: https://arxiv.org/abs/1808.00508
    '''
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.G = nn.Parameter(torch.Tensor(self.out_dim, self.in_dim))
        nn.init.xavier_normal_(self.G)
        self.nac = NAC(self.in_dim, self.out_dim)
        self.eps = 1e-12
        self.bias = None

    def forward(self, x):
        a = self.nac(x)
        g = torch.sigmoid(F.linear(x, self.G, self.bias))
        m = self.nac(torch.log(torch.abs(x) + self.eps))
        m = torch.exp(m)
        y = (g * a) + (1 - g) * m
        return y
