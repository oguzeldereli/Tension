


import torch
from torch import nn
from torch.nn import Module

class Tension(Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.gruA = nn.GRUCell(hidden_dim, hidden_dim)
        self.gruB = nn.GRUCell(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, output_dim)

        self.tanh = nn.Tanh()
    
    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        hA = self.gruA(x)
        hB = self.gruB(x)
        h = torch.cat((hA, hB), dim=-1)
        x = self.linear3(h)
        x = self.tanh(x)
        x = self.linear4(x)
        return x

    def establish_poles(self):
        pass
    def compute_tension(self, inputs):
        pass
    def compute_synthesis(self, inputs):
        pass


