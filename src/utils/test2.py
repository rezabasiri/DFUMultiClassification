
import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from accelerate import Accelerator
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
# from skimage import exposure
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import csv
import torchvision
import torchvision.models
import math

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features_start=32, attention_channels=32):
        super().__init__()
        print(f"\nInitializing UNetGenerator:")
        
        # Calculate channels at each level
        c1 = features_start      # 32
        c2 = features_start * 2  # 64
        c3 = features_start * 4  # 128
        c4 = features_start * 8  # 256

        print(f"Channel progression: {in_channels} -> {c1} -> {c2} -> {c3} -> {c4}")
        
        # Initial convolution
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.InstanceNorm2d(c1),
            nn.ReLU(True)
        )
        
        # Downsampling path (encoder)
        self.down1 = DownBlock(c1, c2)    # 32 -> 64
        self.down2 = DownBlock(c2, c3)    # 64 -> 128
        self.down3 = DownBlock(c3, c4)    # 128 -> 256
        
        # Bridge
        self.bridge = nn.Sequential(
            SelfAttention(c4, attention_channels),
            nn.Conv2d(c4, c4, 3, padding=1),
            nn.InstanceNorm2d(c4),
            nn.ReLU(True)
        )
        
        # Upsampling path (decoder)
        # Note: input channels are doubled due to skip connections
        self.up3 = UpBlock(c4, c3)     # (256 -> 128) + skip(128) = 256
        self.up2 = UpBlock(c3*2, c2)   # (256 -> 64) + skip(64) = 128
        self.up1 = UpBlock(c2*2, c1)   # (128 -> 32) + skip(32) = 64
        
        # Output convolution
        self.conv_out = nn.Sequential(
            nn.Conv2d(c1*2, c1, 3, padding=1),
            nn.InstanceNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, out_channels, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Debug prints for shape tracking
        print(f"\nInput shape: {x.shape}")
        
        # Encoder path
        x1 = self.conv_in(x)
        print(f"After conv_in: {x1.shape}")
        
        x2 = self.down1(x1)
        print(f"After down1: {x2.shape}")
        
        x3 = self.down2(x2)
        print(f"After down2: {x3.shape}")
        
        x4 = self.down3(x3)
        print(f"After down3: {x4.shape}")
        
        # Bridge
        x4 = self.bridge(x4)
        print(f"After bridge: {x4.shape}")
        
        # Decoder path with skip connections
        x = self.up3(x4)
        print(f"After up3 before skip: {x.shape}")
        print(f"Skip3 shape: {x3.shape}")
        x = torch.cat([x, x3], dim=1)
        print(f"After up3 + skip: {x.shape}")
        
        x = self.up2(x)
        print(f"After up2 before skip: {x.shape}")
        print(f"Skip2 shape: {x2.shape}")
        x = torch.cat([x, x2], dim=1)
        print(f"After up2 + skip: {x.shape}")
        
        x = self.up1(x)
        print(f"After up1 before skip: {x.shape}")
        print(f"Skip1 shape: {x1.shape}")
        x = torch.cat([x, x1], dim=1)
        print(f"After up1 + skip: {x.shape}")
        
        x = self.conv_out(x)
        print(f"Final output: {x.shape}")
        
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Use in_channels as input, out_channels as output for the transposed conv
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        print(f"\nUpBlock input: {x.shape}")
        x = self.up(x)
        print(f"After transpose conv: {x.shape}")
        x = self.conv(x)
        print(f"After conv: {x.shape}")
        return x

class DownBlock(nn.Module):
    """Downsampling block with residual connection"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(True),
            nn.Conv2d(out_features, out_features, 3, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(True)
        )
        self.downsample = nn.Conv2d(in_features, out_features, 1, stride=2)
        
    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv(x)
        return F.avg_pool2d(out, 2) + identity
    
model = UNetGenerator()
test_input = torch.randn(1, 3, 32, 32)
output = model(test_input)