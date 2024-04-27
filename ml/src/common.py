import torch

def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    dot_product = vec1 @ vec2.T
    norm_vec1 = torch.norm(vec1)
    norm_vec2 = torch.norm(vec2, dim=1)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity