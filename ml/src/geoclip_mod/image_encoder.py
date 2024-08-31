import torch
import torch.nn as nn
from transformers import CLIPModel, AutoProcessor
from concurrent.futures import ThreadPoolExecutor
from geoclip.model.misc import load_gps_data, file_dir

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
        print(x)
        x = self.mlp(x)
        return x

def _main():
    import os
    import time
    from PIL import Image
    
    with torch.no_grad():
        start_time = time.time()
        
        # Initialize ImageEncoder
        t0 = time.time()
        image_encoder = ImageEncoder()
        print(f"ImageEncoder initialization: {time.time() - t0:.4f} seconds")
        
        # Load weights
        t0 = time.time()
        weights_folder = os.path.join(file_dir, "weights")
        image_encoder.mlp.load_state_dict(torch.load(os.path.join(weights_folder, "image_encoder_mlp_weights.pth")))
        print(f"Weight loading: {time.time() - t0:.4f} seconds")
        
        # Move to CPU
        t0 = time.time()
        image_encoder.to("cpu")
        print(f"Moving to CPU: {time.time() - t0:.4f} seconds")
        
        # Open image
        t0 = time.time()
        tst_im = Image.open('.images/sf_test.png')
        print(f"Image opening: {time.time() - t0:.4f} seconds")
        
        # Preprocess image
        t0 = time.time()
        proc = image_encoder.preprocess_image(tst_im)
        print(f"Image preprocessing: {time.time() - t0:.4f} seconds")
        # Forward pass
        t0 = time.time()
        out = image_encoder(proc)

        print(f"Forward pass: {time.time() - t0:.4f} seconds")
        # final
        t0 = time.time()
        print(f"Output shape: {out.shape}")
        print(f"Output: {out}")
        print(f"Total time: {time.time() - start_time:.4f} seconds")

if __name__ == "__main__":
    _main()