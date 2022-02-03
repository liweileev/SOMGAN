import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    """2D Generator."""

    def __init__(self, z_dim=256, h_dim=128):
        super(Generator, self).__init__()
        
        self.l1 = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())
        self.l2 = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
        self.last = nn.Sequential(nn.Linear(h_dim, 2))    

    def forward(self, z):
        z = z.view(z.size(0), -1)
        out=self.l1(z)
        out=self.l2(out)
        out=self.last(out)
        return out

class Discriminator(nn.Module):
    """2D Discriminator."""

    def __init__(self, h_dim=128):
        super(Discriminator, self).__init__()
        
        self.l1 = nn.Sequential(nn.Linear(2, h_dim), nn.ReLU())
        self.last = nn.Sequential(nn.Linear(h_dim, 1), nn.Softplus())           

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.l1(x)
        out=self.last(out)
        
        return out.reshape((-1,))

