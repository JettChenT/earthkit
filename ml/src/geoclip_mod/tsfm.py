import torch
import torch.nn as nn

class Tsfm(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(768, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 512))
    
    def forward(self, x):
        return self.mlp(x)
