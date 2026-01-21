"""
Data loading utilities for wound image dataset

Handles:
- Loading wound images from disk
- Train/validation splitting
- Data augmentation (including bbox-aware cropping)
- Preprocessing for Stable Diffusion training
"""

import os
import random
import csv
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
        bbox_file: Optional[str] = None,
        cache_latents: bool = False,
        data_percentage: float = 100.0,
        sample_seed: int = 42
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
            augmentation: Dict of augmentation parameters (including bbox_crop_prob)
            bbox_file: Path to bounding box CSV file (optional, for bbox-aware augmentation)
            cache_latents: Whether to pre-compute and cache VAE latents (faster but more memory)
            data_percentage: Percentage of data to use (1-100), sampled equally from each phase
            sample_seed: Random seed for reproducible sampling when data_percentage < 100
        """
        self.data_root = Path(data_root)
        self.modality = modality
        self.phase = phase
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.phase_prompts = phase_prompts
        self.cache_latents = cache_latents
        self.data_percentage = data_percentage
        self.sample_seed = sample_seed
        self.augmentation = augmentation  # Store for __getitem__ access

        # Load bounding boxes if provided
        self.bboxes = {}
        if bbox_file is not None:
            self.bboxes = self._load_bboxes(bbox_file)

        # Find all image files and track their phases
        self.image_paths = []
        self.image_phases = []  # Track phase for each image (for phase-specific prompts)

        if phase.lower() == 'all':
            # Load from all available phase directories
            modality_dir = self.data_root / modality
            if not modality_dir.exists():
                raise ValueError(f"Modality directory not found: {modality_dir}")

            phase_counts = {}
            phase_counts_sampled = {}

            for phase_dir in sorted(modality_dir.iterdir()):  # Sort for deterministic order
                if phase_dir.is_dir():
                    phase_name = phase_dir.name  # 'I', 'P', or 'R'
                    phase_images = []
                    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                        phase_images.extend(list(phase_dir.glob(ext)))

                    # Sort for deterministic ordering
                    phase_images = sorted(phase_images, key=lambda x: x.name)
                    phase_counts[phase_name] = len(phase_images)

                    # Apply data_percentage sampling per phase (balanced)
                    if data_percentage < 100.0:
                        n_samples = max(1, int(len(phase_images) * data_percentage / 100.0))
                        random.seed(sample_seed)
                        phase_images = random.sample(phase_images, n_samples)

                    self.image_paths.extend(phase_images)
                    self.image_phases.extend([phase_name] * len(phase_images))
                    phase_counts_sampled[phase_name] = len(phase_images)

            if len(self.image_paths) == 0:
                raise ValueError(f"No images found in any phase under {modality_dir}")

            total_original = sum(phase_counts.values())
            total_sampled = sum(phase_counts_sampled.values())

            if data_percentage < 100.0:
                print(f"Using {data_percentage}% of data: {total_sampled}/{total_original} images for {modality}/all phases")
            else:
                print(f"Found {total_sampled} images for {modality}/all phases")

            for p in sorted(phase_counts.keys()):
                orig = phase_counts[p]
                sampled = phase_counts_sampled[p]
                if data_percentage < 100.0:
                    print(f"  Phase {p}: {sampled}/{orig} images ({100*sampled/orig:.1f}%)")
                else:
                    print(f"  Phase {p}: {sampled} images")
        else:
            # Load from specific phase
            image_dir = self.data_root / modality / phase
            if not image_dir.exists():
                raise ValueError(f"Image directory not found: {image_dir}")

            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                self.image_paths.extend(list(image_dir.glob(ext)))

            # Sort for deterministic ordering
            self.image_paths = sorted(self.image_paths, key=lambda x: x.name)
            original_count = len(self.image_paths)

            # Apply data_percentage sampling
            if data_percentage < 100.0:
                n_samples = max(1, int(len(self.image_paths) * data_percentage / 100.0))
                random.seed(sample_seed)
                self.image_paths = random.sample(self.image_paths, n_samples)

            # All images have the same phase
            self.image_phases = [phase] * len(self.image_paths)

            if len(self.image_paths) == 0:
                raise ValueError(f"No images found in {image_dir}")

            if data_percentage < 100.0:
                print(f"Using {data_percentage}% of data: {len(self.image_paths)}/{original_count} images for {modality}/{phase}")
            else:
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

    def _load_bboxes(self, bbox_file: str) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Load bounding boxes from CSV file

        Args:
            bbox_file: Path to CSV file with columns: Filename, Xmin, Ymin, Xmax, Ymax

        Returns:
            Dict mapping filename to (xmin, ymin, xmax, ymax)
        """
        bboxes = {}
        bbox_path = Path(bbox_file)

        if not bbox_path.exists():
            print(f"Warning: Bbox file not found: {bbox_file}. Skipping bbox augmentation.")
            return bboxes

        try:
            with open(bbox_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row['Filename']
                    xmin = int(row['Xmin'])
                    ymin = int(row['Ymin'])
                    xmax = int(row['Xmax'])
                    ymax = int(row['Ymax'])
                    bboxes[filename] = (xmin, ymin, xmax, ymax)

            print(f"Loaded {len(bboxes)} bounding boxes from {bbox_file}")
        except Exception as e:
            print(f"Warning: Failed to load bbox file: {e}. Skipping bbox augmentation.")

        return bboxes

    def _apply_bbox_crop(self, image: Image.Image, image_path: Path) -> Image.Image:
        """
        Apply bbox-aware cropping to image

        Args:
            image: PIL Image
            image_path: Path to image file

        Returns:
            Cropped PIL Image (or original if no bbox or not cropping)
        """
        # Check if we should apply bbox crop
        if not self.augmentation or not self.augmentation.get('enabled', False):
            return image

        bbox_crop_prob = self.augmentation.get('bbox_crop_prob', 0.0)
        if bbox_crop_prob == 0.0 or random.random() > bbox_crop_prob:
            return image  # Don't crop this time

        # Get bbox for this image
        filename = image_path.name
        if filename not in self.bboxes:
            return image  # No bbox available

        xmin, ymin, xmax, ymax = self.bboxes[filename]

        # Add random margin
        margin_range = self.augmentation.get('bbox_margin_range', [0.0, 0.0])
        margin_pct = random.uniform(margin_range[0], margin_range[1])

        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        margin_x = int(bbox_width * margin_pct)
        margin_y = int(bbox_height * margin_pct)

        # Expand bbox with margin, clipping to image bounds
        img_width, img_height = image.size
        crop_xmin = max(0, xmin - margin_x)
        crop_ymin = max(0, ymin - margin_y)
        crop_xmax = min(img_width, xmax + margin_x)
        crop_ymax = min(img_height, ymax + margin_y)

        # Crop image
        cropped = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

        return cropped

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

        # Apply bbox crop if enabled (before other transforms)
        image = self._apply_bbox_crop(image, image_path)

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
    bbox_file: Optional[str] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_samples: Optional[int] = None,
    data_percentage: float = 100.0
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
        augmentation: Dict of augmentation parameters (including bbox_crop_prob)
        bbox_file: Path to bounding box CSV file (optional)
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for faster GPU transfer
        max_samples: Optional limit on total samples (for testing)
        data_percentage: Percentage of data to use (1-100), sampled equally from each phase

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
        augmentation=augmentation,
        bbox_file=bbox_file,
        data_percentage=data_percentage,
        sample_seed=split_seed
    )

    # Limit samples if requested (for testing)
    total_size = len(dataset)
    if max_samples is not None and max_samples < total_size:
        print(f"Limiting dataset from {total_size} to {max_samples} samples (TEST MODE)")
        generator = torch.Generator().manual_seed(split_seed)
        indices = torch.randperm(total_size, generator=generator)[:max_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        total_size = max_samples

    # Stratified split: ensure each phase has proportional representation in train/val
    # This is important for fair evaluation across imbalanced classes (I, P, R)
    if phase.lower() == 'all' and hasattr(dataset, 'image_phases'):
        # Get phase labels for stratified splitting
        phase_labels = dataset.image_phases

        # Group indices by phase
        phase_indices = {}
        for idx, phase_label in enumerate(phase_labels):
            if phase_label not in phase_indices:
                phase_indices[phase_label] = []
            phase_indices[phase_label].append(idx)

        train_indices = []
        val_indices = []

        # Split each phase separately to maintain class balance
        random.seed(split_seed)
        for phase_name in sorted(phase_indices.keys()):
            indices = phase_indices[phase_name]
            random.shuffle(indices)

            n_train = int(len(indices) * train_val_split)
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:])

        # Create subsets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        train_size = len(train_indices)
        val_size = len(val_indices)

        # Print stratified split details
        print(f"Stratified split: {train_size} train, {val_size} validation")
        for phase_name in sorted(phase_indices.keys()):
            phase_total = len(phase_indices[phase_name])
            phase_train = int(phase_total * train_val_split)
            phase_val = phase_total - phase_train
            print(f"  Phase {phase_name}: {phase_train} train, {phase_val} val ({100*phase_train/phase_total:.1f}%/{100*phase_val/phase_total:.1f}%)")
    else:
        # Fallback to random split for single-phase datasets
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
        # Load BALANCED samples from each phase directory
        # This ensures consistent comparison across epochs and runs
        modality_dir = Path(data_root) / modality
        phase_dirs = sorted([d for d in modality_dir.iterdir() if d.is_dir()])

        if len(phase_dirs) == 0:
            raise ValueError(f"No phase directories found in {modality_dir}")

        # Calculate samples per phase (equal distribution)
        if num_images is not None:
            samples_per_phase = num_images // len(phase_dirs)
            remainder = num_images % len(phase_dirs)
        else:
            samples_per_phase = None
            remainder = 0

        phase_counts = {}
        for i, phase_dir in enumerate(phase_dirs):
            phase_name = phase_dir.name
            phase_images = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                phase_images.extend(list(phase_dir.glob(ext)))

            # Sort for deterministic ordering before sampling
            phase_images = sorted(phase_images, key=lambda x: x.name)

            # Sample from this phase
            if samples_per_phase is not None:
                # Add one extra to first 'remainder' phases for even distribution
                n_samples = samples_per_phase + (1 if i < remainder else 0)
                n_samples = min(n_samples, len(phase_images))
                random.seed(seed)  # Reset seed for each phase for reproducibility
                phase_images = random.sample(phase_images, n_samples)

            image_paths.extend(phase_images)
            phase_counts[phase_name] = len(phase_images)

        image_dir = modality_dir  # For error message

        if verbose:
            print(f"Loading balanced reference images from {modality_dir}:")
            for p, count in sorted(phase_counts.items()):
                print(f"  Phase {p}: {count} images")
    else:
        image_dir = Path(data_root) / modality / phase
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_paths.extend(list(image_dir.glob(ext)))

        # Sort for deterministic ordering
        image_paths = sorted(image_paths, key=lambda x: x.name)

        # Sample subset if requested
        if num_images is not None and num_images < len(image_paths):
            random.seed(seed)
            image_paths = random.sample(image_paths, num_images)

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")

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
        print(f"Loaded {len(images)} reference images total")

    return images_tensor


def load_reference_images_by_phase(
    data_root: str,
    modality: str,
    resolution: int,
    num_images_per_phase: int = 17,
    seed: int = 42,
    bbox_file: Optional[str] = None,
    bbox_crop_prob: float = 0.5,
    bbox_margin_range: List[float] = [0.05, 0.15],
    verbose: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Load reference images separated by phase for per-phase metrics computation

    Args:
        data_root: Root directory containing wound images
        modality: Imaging modality
        resolution: Target resolution
        num_images_per_phase: Number of images to load per phase
        seed: Random seed for sampling
        bbox_file: Optional path to bbox CSV for bbox-aware cropping
        bbox_crop_prob: Probability of applying bbox crop (0.0-1.0)
        bbox_margin_range: [min, max] margin percentage around bbox
        verbose: Whether to print progress messages

    Returns:
        Dictionary mapping phase name to tensor of images [N, C, H, W] in [0, 1] range
    """
    modality_dir = Path(data_root) / modality
    phase_dirs = sorted([d for d in modality_dir.iterdir() if d.is_dir()])

    if len(phase_dirs) == 0:
        raise ValueError(f"No phase directories found in {modality_dir}")

    # Load bboxes if provided
    bboxes = {}
    if bbox_file is not None:
        bbox_path = Path(bbox_file)
        if bbox_path.exists():
            try:
                with open(bbox_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        filename = row['Filename']
                        xmin = int(row['Xmin'])
                        ymin = int(row['Ymin'])
                        xmax = int(row['Xmax'])
                        ymax = int(row['Ymax'])
                        bboxes[filename] = (xmin, ymin, xmax, ymax)
                if verbose:
                    print(f"Loaded {len(bboxes)} bounding boxes for reference images")
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to load bbox file: {e}")

    transform = transforms.Compose([
        transforms.Resize(resolution, antialias=True),
        transforms.CenterCrop(resolution),
        transforms.ToTensor()  # [0, 1] range
    ])

    phase_images = {}
    random.seed(seed)  # Set seed for reproducible bbox cropping decisions

    for phase_dir in phase_dirs:
        phase_name = phase_dir.name
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_paths.extend(list(phase_dir.glob(ext)))

        if len(image_paths) == 0:
            if verbose:
                print(f"Warning: No images found for phase {phase_name}")
            continue

        # Sort for deterministic ordering before sampling
        image_paths = sorted(image_paths, key=lambda x: x.name)

        # Sample subset
        n_samples = min(num_images_per_phase, len(image_paths))
        random.seed(seed)
        image_paths = random.sample(image_paths, n_samples)

        # Load images with optional bbox cropping
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')

            # Apply bbox crop if enabled and bbox available
            if bbox_crop_prob > 0 and random.random() < bbox_crop_prob:
                filename = image_path.name
                if filename in bboxes:
                    xmin, ymin, xmax, ymax = bboxes[filename]

                    # Add random margin
                    margin_pct = random.uniform(bbox_margin_range[0], bbox_margin_range[1])
                    bbox_width = xmax - xmin
                    bbox_height = ymax - ymin
                    margin_x = int(bbox_width * margin_pct)
                    margin_y = int(bbox_height * margin_pct)

                    # Expand bbox with margin, clipping to image bounds
                    img_width, img_height = image.size
                    crop_xmin = max(0, xmin - margin_x)
                    crop_ymin = max(0, ymin - margin_y)
                    crop_xmax = min(img_width, xmax + margin_x)
                    crop_ymax = min(img_height, ymax + margin_y)

                    # Crop image
                    image = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

            image_tensor = transform(image)
            images.append(image_tensor)

        phase_images[phase_name] = torch.stack(images)

        if verbose:
            print(f"Loaded {n_samples} reference images for phase {phase_name}")

    return phase_images


def save_reference_images_to_disk(
    reference_images_by_phase: Dict[str, torch.Tensor],
    save_dir: str,
    verbose: bool = True
) -> None:
    """
    Save reference images to disk for reuse across training sessions

    Args:
        reference_images_by_phase: Dict mapping phase to tensor [N, C, H, W] in [0, 1] range
        save_dir: Directory to save images
        verbose: Whether to print progress
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save tensors as .pt files
    for phase_name, images in reference_images_by_phase.items():
        tensor_path = save_path / f"reference_{phase_name}.pt"
        torch.save(images.cpu(), tensor_path)
        if verbose:
            print(f"Saved {len(images)} reference images for phase {phase_name} to {tensor_path}")

    # Save combined grid visualization
    try:
        from torchvision.utils import make_grid
        import PIL.Image as PILImage

        all_images = []
        for phase_name in sorted(reference_images_by_phase.keys()):
            all_images.append(reference_images_by_phase[phase_name])

        combined = torch.cat(all_images, dim=0)
        # Create grid (8 images per row)
        grid = make_grid(combined, nrow=8, normalize=False, pad_value=1.0)

        # Convert to PIL and save
        grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        grid_image = PILImage.fromarray(grid_np)
        grid_path = save_path / "reference_images_grid.png"
        grid_image.save(grid_path)

        if verbose:
            print(f"Saved reference image grid to {grid_path}")

        # Save per-phase grids too
        for phase_name, images in reference_images_by_phase.items():
            phase_grid = make_grid(images, nrow=4, normalize=False, pad_value=1.0)
            phase_grid_np = (phase_grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            phase_grid_image = PILImage.fromarray(phase_grid_np)
            phase_grid_path = save_path / f"reference_{phase_name}_grid.png"
            phase_grid_image.save(phase_grid_path)

    except Exception as e:
        if verbose:
            print(f"Warning: Could not save reference image grids: {e}")


def load_reference_images_from_disk(
    save_dir: str,
    phases: Optional[List[str]] = None,
    verbose: bool = True
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load reference images from disk

    Args:
        save_dir: Directory containing saved images
        phases: Optional list of phase names to load (e.g., ['I', 'P', 'R'])
                If None, attempts to load all available phases
        verbose: Whether to print progress

    Returns:
        Dict mapping phase to tensor [N, C, H, W] in [0, 1] range, or None if not found
    """
    save_path = Path(save_dir)

    if not save_path.exists():
        if verbose:
            print(f"Reference image directory not found: {save_dir}")
        return None

    # Auto-detect phases if not provided
    if phases is None:
        phase_files = list(save_path.glob("reference_*.pt"))
        if len(phase_files) == 0:
            if verbose:
                print(f"No reference image files found in {save_dir}")
            return None
        phases = [f.stem.replace("reference_", "") for f in phase_files]

    # Load each phase
    reference_images_by_phase = {}
    for phase_name in phases:
        tensor_path = save_path / f"reference_{phase_name}.pt"
        if not tensor_path.exists():
            if verbose:
                print(f"Warning: Reference images not found for phase {phase_name}")
            continue

        try:
            images = torch.load(tensor_path)
            reference_images_by_phase[phase_name] = images
            if verbose:
                print(f"Loaded {len(images)} reference images for phase {phase_name}")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load reference images for phase {phase_name}: {e}")

    if len(reference_images_by_phase) == 0:
        return None

    return reference_images_by_phase


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
