"""
Data loading utilities for wound image dataset

Handles:
- Loading wound images from disk
- Train/validation splitting
- Data augmentation
- Preprocessing for Stable Diffusion training
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from transformers import CLIPTokenizer


class WoundDataset(Dataset):
    """
    Dataset for diabetic foot ulcer wound images

    Loads images from directory structure:
        data_root/modality/phase/*.png
    Example:
        data/DFU_Updated/rgb/I/*.png
        data/DFU_Updated/rgb/R/*.png
    """

    def __init__(
        self,
        data_root: str,
        modality: str,
        phase: str,
        resolution: int,
        tokenizer: CLIPTokenizer,
        prompt: Optional[str] = None,
        augmentation: Optional[Dict] = None,
        cache_latents: bool = False
    ):
        """
        Initialize wound dataset

        Args:
            data_root: Root directory containing wound images
            modality: Imaging modality ('rgb', 'depth_map', 'thermal_map')
            phase: Healing phase ('I', 'P', 'R')
            resolution: Target image resolution (e.g., 128)
            tokenizer: CLIP tokenizer for text prompts
            prompt: Text prompt for conditional generation
            augmentation: Dict of augmentation parameters
            cache_latents: Whether to pre-compute and cache VAE latents (faster but more memory)
        """
        self.data_root = Path(data_root)
        self.modality = modality
        self.phase = phase
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.cache_latents = cache_latents

        # Find all image files for this phase
        image_dir = self.data_root / modality / phase
        if not image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")

        # Load all image paths
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            self.image_paths.extend(list(image_dir.glob(ext)))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        print(f"Found {len(self.image_paths)} images for {modality}/{phase}")

        # Setup data augmentation
        self.setup_augmentation(augmentation)

        # Tokenize prompt once (same for all images in this phase)
        if self.prompt is not None:
            self.encoded_prompt = self.tokenize_prompt(self.prompt)
        else:
            self.encoded_prompt = None

    def setup_augmentation(self, augmentation: Optional[Dict]):
        """
        Setup data augmentation transforms

        Args:
            augmentation: Dict with augmentation parameters
        """
        transforms_list = []

        # Base transforms (always applied)
        transforms_list.append(transforms.Resize(
            self.resolution,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        ))
        transforms_list.append(transforms.CenterCrop(self.resolution))

        # Optional augmentations
        if augmentation is not None and augmentation.get('enabled', False):

            # Random horizontal flip
            if augmentation.get('horizontal_flip', False):
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

            # Random vertical flip
            if augmentation.get('vertical_flip', False):
                transforms_list.append(transforms.RandomVerticalFlip(p=0.5))

            # Random rotation
            rotation_degrees = augmentation.get('rotation_degrees', 0)
            if rotation_degrees > 0:
                transforms_list.append(transforms.RandomRotation(
                    degrees=rotation_degrees,
                    interpolation=transforms.InterpolationMode.BILINEAR
                ))

            # Color jitter
            color_jitter = augmentation.get('color_jitter', None)
            if color_jitter is not None:
                transforms_list.append(transforms.ColorJitter(
                    brightness=color_jitter.get('brightness', 0.0),
                    contrast=color_jitter.get('contrast', 0.0),
                    saturation=color_jitter.get('saturation', 0.0),
                    hue=color_jitter.get('hue', 0.0)
                ))

        # Convert to tensor and normalize to [-1, 1] (SD training range)
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize([0.5], [0.5]))

        self.transform = transforms.Compose(transforms_list)

    def tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """
        Tokenize text prompt using CLIP tokenizer

        Args:
            prompt: Text prompt string

        Returns:
            Encoded prompt tensor
        """
        encoded = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        return encoded.input_ids[0]

    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from dataset

        Args:
            idx: Index of item

        Returns:
            Dictionary with 'pixel_values' and optionally 'input_ids'
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # Apply transformations
        pixel_values = self.transform(image)

        # Prepare output dict
        output = {'pixel_values': pixel_values}

        # Add prompt if available
        if self.encoded_prompt is not None:
            output['input_ids'] = self.encoded_prompt

        return output


def create_dataloaders(
    data_root: str,
    modality: str,
    phase: str,
    resolution: int,
    tokenizer: CLIPTokenizer,
    batch_size: int,
    train_val_split: float = 0.85,
    split_seed: int = 42,
    prompt: Optional[str] = None,
    augmentation: Optional[Dict] = None,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Create train and validation dataloaders

    Args:
        data_root: Root directory containing wound images
        modality: Imaging modality ('rgb', 'depth_map', 'thermal_map')
        phase: Healing phase ('I', 'P', 'R')
        resolution: Target image resolution
        tokenizer: CLIP tokenizer
        batch_size: Batch size
        train_val_split: Ratio of train/total (e.g., 0.85 = 85% train)
        split_seed: Random seed for split
        prompt: Text prompt for conditional generation
        augmentation: Dict of augmentation parameters
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        (train_loader, val_loader, train_size, val_size)
    """
    # Create full dataset
    dataset = WoundDataset(
        data_root=data_root,
        modality=modality,
        phase=phase,
        resolution=resolution,
        tokenizer=tokenizer,
        prompt=prompt,
        augmentation=augmentation
    )

    # Split into train/val
    total_size = len(dataset)
    train_size = int(train_val_split * total_size)
    val_size = total_size - train_size

    # Use random_split with seed for reproducibility
    generator = torch.Generator().manual_seed(split_seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    print(f"Dataset split: {train_size} train, {val_size} validation")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, train_size, val_size


def load_reference_images(
    data_root: str,
    modality: str,
    phase: str,
    resolution: int,
    num_images: Optional[int] = None,
    seed: int = 42
) -> torch.Tensor:
    """
    Load a set of reference images for quality comparison

    Args:
        data_root: Root directory containing wound images
        modality: Imaging modality
        phase: Healing phase
        resolution: Target resolution
        num_images: Number of images to load (None = all)
        seed: Random seed for sampling

    Returns:
        Tensor of reference images [N, C, H, W] in [0, 1] range
    """
    # Find all image files
    image_dir = Path(data_root) / modality / phase
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_paths.extend(list(image_dir.glob(ext)))

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")

    # Sample subset if requested
    if num_images is not None and num_images < len(image_paths):
        random.seed(seed)
        image_paths = random.sample(image_paths, num_images)

    # Load and preprocess images
    transform = transforms.Compose([
        transforms.Resize(resolution, antialias=True),
        transforms.CenterCrop(resolution),
        transforms.ToTensor()  # [0, 1] range
    ])

    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        images.append(image_tensor)

    # Stack into batch
    images_tensor = torch.stack(images)

    print(f"Loaded {len(images)} reference images from {image_dir}")

    return images_tensor


def collate_fn(examples: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for dataloader

    Args:
        examples: List of examples from dataset

    Returns:
        Batched dictionary
    """
    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {'pixel_values': pixel_values}

    # Add prompts if available
    if 'input_ids' in examples[0]:
        input_ids = torch.stack([example['input_ids'] for example in examples])
        batch['input_ids'] = input_ids

    return batch


def get_dataset_statistics(
    data_root: str,
    modality: str,
    phase: str
) -> Dict[str, int]:
    """
    Get statistics about the dataset

    Args:
        data_root: Root directory
        modality: Imaging modality
        phase: Healing phase

    Returns:
        Dictionary with statistics
    """
    image_dir = Path(data_root) / modality / phase

    if not image_dir.exists():
        return {'total_images': 0}

    # Count images
    image_count = 0
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_count += len(list(image_dir.glob(ext)))

    # Get image sizes (sample first 10)
    image_paths = list(image_dir.glob('*.png'))[:10]
    sizes = []
    for img_path in image_paths:
        img = Image.open(img_path)
        sizes.append(img.size)

    # Find most common size
    from collections import Counter
    size_counts = Counter(sizes)
    most_common_size = size_counts.most_common(1)[0][0] if sizes else None

    return {
        'total_images': image_count,
        'most_common_size': most_common_size,
        'directory': str(image_dir)
    }
