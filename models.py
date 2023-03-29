"""
Loaders for models and their weights
"""

import os
import torch
from torch.nn import DataParallel
from gdown import cached_download

from rvm import MattingNetwork
from rainnet import RainNet, RAIN
from espcn import ESPCN

def load(model, file_name, url, device):
    """Load pretrained model weights from a file"""
    path = file_name
    if url.startswith('https://drive.google.com/'): cached_download(url, path, quiet=True)
    else: os.system(f'wget -c {url} -q -O {path}')
    model.load_state_dict(torch.load(path))
    return model.to(device).eval()

def get_matting_model(device):
    """Loading the matting model"""
    model = MattingNetwork('resnet50')
    return load(model, 'rvm_resnet50.pth',
                'https://drive.google.com/uc?id=11imcmALlt_zI3oTHySaMBFZdSQargWl3', device)

def get_harmonization_model(device):
    """Loading the harmonization model"""
    model = RainNet(input_nc=3, output_nc=3, ngf=64, norm_layer=RAIN, use_dropout=True)
    return load(model, 'netG_latest.pth',
                'https://drive.google.com/uc?id=1nVJFQ1iAGMeZ-ZybJm9vBQfvYs6tkctZ', device)

def get_super_resolution_model(scale_factor, device):
    """Loading the super resolution model"""
    model = ESPCN(scale_factor=scale_factor)
    return load(model, f'espcn_{scale_factor}x.pth',
                f'https://github.com/Lornatang/ESPCN-PyTorch/releases/download/1.0/espcn_{scale_factor}x.pth', device)
