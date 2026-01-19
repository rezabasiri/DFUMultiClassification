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

    Use phase='all' to load images from all available phases with phase-specific prompts.
    """

    def __init__(
        self,
        data_root: str,
        modality: str,
        phase: str,
        resolution: int,
        tokenizer: CLIPTokenizer,
        prompt: Optional[str] = None,
        phase_prompts: Optional[Dict[str, str]] = None,
        augmentation: Optional[Dict] = None,
        cache_latents: bool = False
    ):
        """
        Initialize wound dataset

        Args:
            data_root: Root directory containing wound images
            modality: Imaging modality ('rgb', 'depth_map', 'thermal_map')
            phase: Healing phase ('I', 'P', 'R', or 'all' for all phases)
            resolution: Target image resolution (e.g., 128)
            tokenizer: CLIP tokenizer for text prompts
            prompt: Text prompt for conditional generation (used when phase != 'all')
            phase_prompts: Dict mapping phase to prompt (used when phase == 'all')
                Example: {'I': 'inflammatory phase...', 'P': 'proliferative phase...', 'R': 'remodeling phase...'}
            augmentation: Dict of augmentation parameters
            cache_latents: Whether to pre-compute and cache VAE latents (faster but more memory)
        """
        self.data_root = Path(data_root)
        self.modality = modality
        self.phase = phase
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.phase_prompts = phase_prompts
        self.cache_latents = cache_latents

        # Find all image files and track their phases
        self.image_paths = []
        self.image_phases = []  # Track phase for each image (for phase-specific prompts)

        if phase.lower() == 'all':
            # Load from all available phase directories
            modality_dir = self.data_root / modality
            if not modality_dir.exists():
                raise ValueError(f"Modality directory not found: {modality_dir}")

            phase_counts = {}
            for phase_dir in modality_dir.iterdir():
                if phase_dir.is_dir():
                    phase_name = phase_dir.name  # 'I', 'P', or 'R'
                    phase_images = []
                    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                        phase_images.extend(list(phase_dir.glob(ext)))

                    self.image_paths.extend(phase_images)
                    self.image_phases.extend([phase_name] * len(phase_images))
                    phase_counts[phase_name] = len(phase_images)

            if len(self.image_paths) == 0:
                raise ValueError(f"No images found in any phase under {modality_dir}")

            print(f"Found {len(self.image_paths)} images for {modality}/all phases")
            for p, count in sorted(phase_counts.items()):
                print(f"  Phase {p}: {count} images")
        else:
            # Load from specific phase
            image_dir = self.data_root / modality / phase
            if not image_dir.exists():
                raise ValueError(f"Image directory not found: {image_dir}")

            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                self.image_paths.extend(list(image_dir.glob(ext)))

            # All images have the same phase
            self.image_phases = [phase] * len(self.image_paths)

            if len(self.image_paths) == 0:
                raise ValueError(f"No images found in {image_dir}")

            print(f"Found {len(self.image_paths)} images for {modality}/{phase}")

        # Setup data augmentation
        self.setup_augmentation(augmentation)

        # Pre-tokenize prompts for efficiency
        self.encoded_prompts = {}

        if phase.lower() == 'all' and phase_prompts is not None:
            # Tokenize phase-specific prompts
            for p, p_prompt in phase_prompts.items():
                self.encoded_prompts[p] = self.tokenize_prompt(p_prompt)
            print(f"Using phase-specific prompts for phases: {list(phase_prompts.keys())}")
        elif self.prompt is not None:
            # Single prompt for all images
            self.encoded_prompts['default'] = self.tokenize_prompt(self.prompt)

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

        # Add prompt - use phase-specific if available
        image_phase = self.image_phases[idx]
        if image_phase in self.encoded_prompts:
            output['input_ids'] = self.encoded_prompts[image_phase]
        elif 'default' in self.encoded_prompts:
            output['input_ids'] = self.encoded_prompts['default']

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
    phase_prompts: Optional[Dict[str, str]] = None,
    augmentation: Optional[Dict] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Create train and validation dataloaders

    Args:
        data_root: Root directory containing wound images
        modality: Imaging modality ('rgb', 'depth_map', 'thermal_map')
        phase: Healing phase ('I', 'P', 'R', or 'all' for all phases)
        resolution: Target image resolution
        tokenizer: CLIP tokenizer
        batch_size: Batch size
        train_val_split: Ratio of train/total (e.g., 0.85 = 85% train)
        split_seed: Random seed for split
        prompt: Text prompt for conditional generation (single phase)
        phase_prompts: Dict mapping phase to prompt (for multi-phase training)
        augmentation: Dict of augmentation parameters
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for faster GPU transfer
        max_samples: Optional limit on total samples (for testing)

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
        phase_prompts=phase_prompts,
        augmentation=augmentation
    )

    # Limit samples if requested (for testing)
    total_size = len(dataset)
    if max_samples is not None and max_samples < total_size:
        print(f"Limiting dataset from {total_size} to {max_samples} samples (TEST MODE)")
        generator = torch.Generator().manual_seed(split_seed)
        indices = torch.randperm(total_size, generator=generator)[:max_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        total_size = max_samples

    # Split into train/val
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
    seed: int = 42,
    verbose: bool = True
) -> torch.Tensor:
    """
    Load a set of reference images for quality comparison

    Args:
        data_root: Root directory containing wound images
        modality: Imaging modality
        phase: Healing phase ('I', 'P', 'R', or 'all' for all phases)
        resolution: Target resolution
        num_images: Number of images to load (None = all)
        seed: Random seed for sampling
        verbose: Whether to print progress messages

    Returns:
        Tensor of reference images [N, C, H, W] in [0, 1] range
    """
    # Find all image files
    image_paths = []

    if phase.lower() == 'all':
        # Load from all available phase directories
        modality_dir = Path(data_root) / modality
        for phase_dir in modality_dir.iterdir():
            if phase_dir.is_dir():
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                    image_paths.extend(list(phase_dir.glob(ext)))
        image_dir = modality_dir  # For error message
    else:
        image_dir = Path(data_root) / modality / phase
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

    if verbose:
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
