import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .tsfm import Tsfm
from geoclip.model.location_encoder import LocationEncoder
from geoclip.model.misc import load_gps_data, file_dir

from PIL import Image

class GeoCLIP(nn.Module):
    def __init__(self, from_pretrained=True, queue_size=4096):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.device = "cpu"
        self.location_encoder = LocationEncoder().to(self.device)
        self.tsfm = Tsfm().to(self.device)
        self.gps_gallery = load_gps_data(os.path.join(file_dir, "gps_gallery", "coordinates_100K.csv")).to(self.device)
        self._initialize_gps_queue(queue_size)
        self.weights_folder = os.path.join(file_dir, "weights")
        self._load_weights()
        self.location_feats = F.normalize(self.location_encoder(self.gps_gallery), dim=1)
        self.logit_scale_exp = self.logit_scale.exp()
        
    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)

    def _load_weights(self):
        self.tsfm.mlp.load_state_dict(torch.load(f"{self.weights_folder}/image_encoder_mlp_weights.pth"))
        self.location_encoder.load_state_dict(torch.load(f"{self.weights_folder}/location_encoder_weights.pth"))
        self.logit_scale = nn.Parameter(torch.load(f"{self.weights_folder}/logit_scale_weights.pth"))

    def _initialize_gps_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """ Update GPS queue

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)
        
        assert self.queue_size % gps_batch_size == 0, f"Queue size {self.queue_size} should be divisible by batch size {gps_batch_size}"

        # Replace the GPS from ptr to ptr+gps_batch_size (dequeue and enqueue)
        self.gps_queue[:, gps_ptr:gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size  # move pointer
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        return self.gps_queue.t()
                                            
    
    @torch.no_grad()
    def forward_cached(self, img_feats):
        """ Forward pass with cached GPS gallery

        Args:
            image_features (torch.Tensor): Image features of shape (n, 512)

        Returns:
            logits_per_image (torch.Tensor): Logits per image of shape (n, m)
        """
        img_features = self.tsfm(img_feats)
        img_feat_norm = F.normalize(img_features, dim=1)
        logits_per_image = self.logit_scale_exp * (img_feat_norm @ self.location_feats.t())
        return logits_per_image

    @torch.no_grad()
    def predict(self, img_feats, top_k):
        """ Given an image, predict the top k GPS coordinates

        Args:
            image_path (str): Path to the image
            top_k (int): Number of top predictions to return

        Returns:
            top_pred_gps (torch.Tensor): Top k GPS coordinates of shape (k, 2)
            top_pred_prob (torch.Tensor): Top k GPS probabilities of shape (k,)
        """
        logits_per_image = self.forward_cached(img_feats)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Get top k predictions
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob

def _dump_coordembeds():
    gclip = GeoCLIP()
    feats = gclip.location_feats.cpu()
    coords = gclip.gps_gallery.cpu()
    print(feats.shape)

def _tst():
    gclip = GeoCLIP()
    print(gclip.logit_scale_exp)

def _main():
    from dotenv import load_dotenv
    from time import time
    load_dotenv()
    import replicate
    gclip = GeoCLIP()
    src_img = "https://jld59we6hmprlla0.public.blob.vercel-storage.com/1d43cbec-f01c-4e21-bf17-4d455a69974e-l1HNRCEvZQ2nDDgc36cUTmiNGGeCms.png"
    print('Getting embeddings...')
    embedding_start = time()
    src_img_embeddings = replicate.run("andreasjansson/clip-features:75b33f253f7714a281ad3e9b28f63e3232d583716ef6718f2e46641077ea040a", input={"inputs": src_img})
    embs = torch.tensor(src_img_embeddings[0]['embedding']).unsqueeze(0)  # Add batch dimension
    embedding_time = time() - embedding_start
    print(f"Embedding generation time: {embedding_time:.2f} seconds")
    print(f"Embedding shape: {embs.shape}")
    
    print('Running inference...')
    print(embs.shape)
    tst = time()
    res = gclip.predict(embs, 40)
    print(f"Inference time: {time() - tst:.2f} seconds")
    print(res)

if __name__ == "__main__":
    _tst()