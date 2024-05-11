import torch
import torch.nn as nn
from transformers import CLIPModel, AutoProcessor
from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.*')

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.CLIP = None
        self.image_processor = None
        self.mlp = nn.Sequential(nn.Linear(768, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 512))

        # Load models concurrently
        with ThreadPoolExecutor() as executor:
            future_clip = executor.submit(CLIPModel.from_pretrained, "openai/clip-vit-large-patch14")
            future_processor = executor.submit(AutoProcessor.from_pretrained, "openai/clip-vit-large-patch14")
            self.CLIP = future_clip.result()
            self.image_processor = future_processor.result()

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        x = self.CLIP.get_image_features(pixel_values=x)
        x = self.mlp(x)
        return x