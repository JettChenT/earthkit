import torch
import torch.nn as nn
import importlib.util
from pathlib import Path
import os

gclip_spec = importlib.util.find_spec("geoclip")
if gclip_spec is None or gclip_spec.origin is None:
    raise ImportError("GCLIP spec or origin not found")
file_dir = Path(gclip_spec.origin).parent / 'model'

class Tsfm(nn.Module):
    """
    MLP part of GeoCLIP's image encoder
    """
    def __init__(self):
        super().__init__()
        self.weights_folder = os.path.join(file_dir, "weights")
        self.mlp = nn.Sequential(nn.Linear(768, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 512))
    
    def load_weights(self):
        self.mlp.load_state_dict(torch.load(f"{self.weights_folder}/image_encoder_mlp_weights.pth"))

    def forward(self, x):
        return self.mlp(x)
