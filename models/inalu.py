import torch
from torch import nn
from models.nac import NAC
from torch.nn import functional as F

class INALU(nn.Module):
    '''
    Class implementing Improved Neural Arithmetic Logic Unit (INALU)
    '''
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.G = nn.Parameter(torch.Tensor(self.out_dim,self.in_dim))
        nn.init.xavier_normal_(self.G)
        self.nac_a = NAC(self.in_dim, self.out_dim)
        self.nac_m = NAC(self.in_dim, self.out_dim)
        self.eps = 1e-12
        self.omega = 1e-12
        self.bias = None
        
    def forward(self, x):
        a = self.nac_a(x)
        m = self.nac_m(torch.log(torch.max(torch.abs(x), torch.tensor(self.eps))))
        m = torch.exp(torch.min(m, torch.tensor(self.omega)))
        g = torch.sigmoid(F.linear(x, self.G, self.bias))
        y = (g * a) + (1 - g) * m
        return y
        
