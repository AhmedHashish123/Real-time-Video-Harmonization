"""
Utility functions
"""

import torch
import cv2
import numpy as np
from torchvision.transforms.functional import resize, to_pil_image

def read_image(path):
    """Returns a PyTorch GPU tensor representing the image in the path"""
    return cv2.imread(path)

def postprocessing_weights(t, sigma, device, h=9, w=9):
    """Returns a spatio-temporal Gaussian kernel of the specified dimensions and standard deviation"""
    window = torch.ones(1, h, w, dtype=torch.float32, device=device)
    window[0, h // 2, w // 2] = 0.0
    kernel = torch.stack(tuple(window + i for i in range(t)), dim=1)
    g = torch.exp(-kernel**2/(2*sigma**2)) / (sigma * torch.sqrt(2*2*torch.acos(torch.zeros(1, device=device))))
    return g / g.sum()

def to_torch(image, resolution, device):
    """Converts an H x W x C image (CPU NumPy array) to a 1 x C x *resolution tensor (GPU PyTorch)"""
    image = torch.from_numpy(image).to(device).permute(2, 0, 1).flip(0).unsqueeze(0) / 255.0
    return resize(image, resolution) if (image.shape[-2], image.shape[-1]) != resolution else image

def to_numpy(tensor):
    """Converts a 1 x C x H x W tensor (GPU PyTorch) to an H x W x C image (CPU NumPy array)"""
    return np.array(to_pil_image(tensor.squeeze().cpu()))[...,::-1]

def compose_tensors(alpha, foreground, background):
    """Composes an image from the foreground and background using alpha as a mask"""
    return alpha * foreground + (1 - alpha) * background
