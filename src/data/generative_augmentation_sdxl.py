"""
SDXL-based Generative Augmentation for Diabetic Foot Ulcer Images

This module provides generative augmentation using a fine-tuned SDXL model.
Replaces the older SD 1.5 LoRA-based generative_augmentation_v2.py

Key differences from v2:
- Uses SDXL 1.0 Base (2.6B params) instead of SD 1.5
- Loads from checkpoint files instead of pretrained LoRA models
- Full fine-tuning (100% params) instead of LoRA (11.5%)
- Trained at 512x512 resolution
- Phase-specific training with minimal prompts
- Uses bbox-cropped wound-only images
"""

import os
import gc
from pathlib import Path
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from diffusers.utils import logging as diffusers_logging
import random
import tensorflow as tf
import traceback
import numpy as np
import threading
from collections import OrderedDict
import yaml

# Limit PyTorch thread usage to prevent resource exhaustion when SDXL is loaded
# SDXL is large (2.6B params) and can exhaust system threads alongside TensorFlow
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

from src.utils.production_config import (
    IMAGE_SIZE,
    USE_GENERATIVE_AUGMENTATION,
    GENERATIVE_AUG_PROB,
    GENERATIVE_AUG_MIX_RATIO,
    GENERATIVE_AUG_INFERENCE_STEPS,
    GENERATIVE_AUG_BATCH_LIMIT,
    GENERATIVE_AUG_MAX_MODELS,
    GENERATIVE_AUG_PHASES
)

# Disable all progress bars comprehensively - MUST come before any diffusers imports
os.environ['TQDM_DISABLE'] = '1'  # Disable tqdm globally

# Monkey-patch tqdm to make it a no-op before diffusers uses it
import sys
class DummyTqdm:
    """Dummy tqdm that does nothing"""
    def __init__(self, *args, **kwargs):
        self.iterable = kwargs.get('iterable', args[0] if args else None)
    def __iter__(self):
        return iter(self.iterable) if self.iterable is not None else iter([])
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def update(self, *args, **kwargs):
        pass
    def close(self):
        pass
    def set_description(self, *args, **kwargs):
        pass

# Replace tqdm in all its import locations
sys.modules['tqdm'] = type(sys)('tqdm')
sys.modules['tqdm'].tqdm = DummyTqdm
sys.modules['tqdm.auto'] = type(sys)('tqdm.auto')
sys.modules['tqdm.auto'].tqdm = DummyTqdm

diffusers_logging.disable_progress_bar()  # Disable diffusers progress bars
diffusers_logging.set_verbosity_error()  # Only show errors from diffusers

# Suppress TensorFlow GeneratorDatasetOp warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings (keep errors)
tf.get_logger().setLevel('ERROR')  # Only show TF errors


class GeneratedImageCounter:
    """Thread-safe counter for tracking generated images with per-class breakdown"""
    def __init__(self):
        self.count = 0
        self.class_counts = {'I': 0, 'P': 0, 'R': 0}  # Per-class counts
        self.lock = threading.Lock()
        self.print_interval = 10  # Print every N images

    def increment(self, batch_size=1, phase=None):
        """Increment counter and optionally print update

        Args:
            batch_size: Number of images generated
            phase: Phase ('I', 'P', 'R') for per-class tracking
        """
        with self.lock:
            self.count += batch_size
            if phase and phase in self.class_counts:
                self.class_counts[phase] += batch_size
            if self.count % self.print_interval == 0 or batch_size > 1:
                print(f"  Generated images: {self.count}", flush=True)

    def reset(self):
        """Reset counter to 0"""
        with self.lock:
            self.count = 0
            self.class_counts = {'I': 0, 'P': 0, 'R': 0}

    def get_count(self):
        """Get current count"""
        with self.lock:
            return self.count

    def get_summary(self):
        """Get summary string with per-class breakdown"""
        with self.lock:
            return f"Total: {self.count} (I={self.class_counts['I']}, P={self.class_counts['P']}, R={self.class_counts['R']})"

    def print_summary(self):
        """Print generation summary"""
        with self.lock:
            if self.count > 0:
                print(f"\n================================================================================")
                print(f"SDXL GENERATION SUMMARY")
                print(f"================================================================================")
                print(f"Total images generated: {self.count}")
                print(f"  Phase I (Inflammatory): {self.class_counts['I']}")
                print(f"  Phase P (Proliferative): {self.class_counts['P']}")
                print(f"  Phase R (Remodeling): {self.class_counts['R']}")
                print(f"================================================================================\n")


# Global counter instance
_gen_image_counter = GeneratedImageCounter()


class AugmentationConfig:
    def __init__(self):
        # Per-modality settings
        # Note: depth_rgb setting acts as master switch for generative augmentation
        # Both depth_rgb and thermal_rgb use the same RGB models (trained on both wound types)
        self.modality_settings = {
            'depth_rgb': {
                'regular_augmentations': {
                    'enabled': True,
                    'prob': 0.6,
                    'brightness': {'enabled': True, 'max_delta': 0.6},
                    'contrast': {'enabled': True, 'range': (0.6, 1.4)},
                    'saturation': {'enabled': True, 'range': (0.6, 1.4)},
                    'gaussian_noise': {'enabled': True, 'stddev': 0.15},
                },
                'generative_augmentations': {
                    'enabled': USE_GENERATIVE_AUGMENTATION  # Master switch from production_config
                }
            },
            'thermal_rgb': {
                'regular_augmentations': {
                    'enabled': True,
                    'prob': 0.6,
                    'brightness': {'enabled': True, 'max_delta': 0.6},
                    'contrast': {'enabled': True, 'range': (0.6, 1.4)},
                    'saturation': {'enabled': True, 'range': (0.6, 1.4)},
                    'gaussian_noise': {'enabled': True, 'stddev': 0.15},
                },
                'generative_augmentations': {
                    'enabled': False  # Controlled by depth_rgb setting
                }
            },
            'thermal_map': {
                'regular_augmentations': {
                    'enabled': True,
                    'prob': 0.6,
                    'brightness': {'enabled': True, 'max_delta': 0.4},
                    'contrast': {'enabled': True, 'range': (0.6, 1.4)},
                    'saturation': {'enabled': False},
                    'gaussian_noise': {'enabled': True, 'stddev': 0.1},
                },
                'generative_augmentations': {
                    'enabled': False  # Map modalities not using generative aug
                }
            },
            'depth_map': {
                'regular_augmentations': {
                    'enabled': True,
                    'prob': 0.6,
                    'brightness': {'enabled': True, 'max_delta': 0.4},
                    'contrast': {'enabled': True, 'range': (0.6, 1.4)},
                    'saturation': {'enabled': False},
                    'gaussian_noise': {'enabled': True, 'stddev': 0.1},
                },
                'generative_augmentations': {
                    'enabled': False  # Map modalities not using generative aug
                }
            }
        }

        # Global generative settings (from production_config)
        self.generative_settings = {
            'output_size': {'height': IMAGE_SIZE, 'width': IMAGE_SIZE},
            'prob': GENERATIVE_AUG_PROB,
            'mix_ratio_range': GENERATIVE_AUG_MIX_RATIO,
            'inference_steps': GENERATIVE_AUG_INFERENCE_STEPS,
            'batch_size_limit': GENERATIVE_AUG_BATCH_LIMIT,
            'max_loaded_models': GENERATIVE_AUG_MAX_MODELS,
        }


def resize_and_pad_generated_images(generated_images, target_shape):
    """
    Resize generated images to match target shape while maintaining aspect ratio and padding

    Args:
        generated_images: Tensor of shape [batch_size, height, width, channels]
        target_shape: Tuple of (height, width) for the target size
    """
    if generated_images.shape[1:3] == target_shape[:2]:
        return generated_images

    batch_size = tf.shape(generated_images)[0]
    resized_images = []

    for i in range(batch_size):
        image = generated_images[i]

        # Calculate aspect ratios
        orig_aspect = tf.shape(image)[1] / tf.shape(image)[0]
        target_aspect = target_shape[1] / target_shape[0]

        if orig_aspect > target_aspect:
            # Image is wider than target
            new_width = target_shape[1]
            new_height = tf.cast(new_width / orig_aspect, tf.int32)
        else:
            # Image is taller than target
            new_height = target_shape[0]
            new_width = tf.cast(new_height * orig_aspect, tf.int32)

        # Resize maintaining aspect ratio
        resized = tf.image.resize(
            image,
            [new_height, new_width],
        )

        # Calculate padding
        pad_height = target_shape[0] - new_height
        pad_width = target_shape[1] - new_width

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Pad to match target size
        padded = tf.pad(
            resized,
            [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            mode='CONSTANT',
            constant_values=0
        )

        resized_images.append(padded)

    return tf.stack(resized_images)


@tf.function(reduce_retracing=True)
def augment_image(image, modality, seed, config):
    """Apply augmentations based on modality type and config"""
    # Early returns
    if not config.modality_settings[modality]['regular_augmentations']['enabled']:
        return image

    if len(tf.shape(image)) != 4 or tf.shape(image)[-1] != 3:
        return image

    # Get seed value
    seed_val = tf.get_static_value(seed)
    if seed_val is None:
        seed_val = 42

    # Try augmentation - run on CPU to avoid deterministic GPU issues with AdjustContrastv2
    try:
        settings = config.modality_settings[modality]['regular_augmentations']
        with tf.device('/CPU:0'):
            if modality in ['depth_rgb', 'thermal_rgb']:
                augmented = tf.map_fn(
                    lambda x: apply_pixel_augmentation_rgb(x, seed_val, settings),
                    image,
                    fn_output_signature=tf.float32
                )
            else:
                augmented = tf.map_fn(
                    lambda x: apply_pixel_augmentation_map(x, seed_val, settings),
                    image,
                    fn_output_signature=tf.float32
                )

        augmented = tf.ensure_shape(augmented, [
            None,
            config.generative_settings['output_size']['height'],
            config.generative_settings['output_size']['width'],
            3
        ])
        return augmented  # Return the augmented image if successful

    except Exception as e:
        print(f"Error in augment_image: {str(e)}")
        return image


@tf.function(reduce_retracing=True)
def apply_pixel_augmentation_rgb(image, seed, settings):
    """Apply RGB-specific augmentations based on settings"""
    if len(tf.shape(image)) != 3:
        return image

    if tf.random.uniform([], seed=seed) < settings['prob']:
        if tf.random.uniform([], seed=seed) < 0.6:
            if settings['brightness']['enabled']:
                image = tf.image.random_brightness(
                    image,
                    settings['brightness']['max_delta'],
                    seed=seed
                )

        if tf.random.uniform([], seed=seed) < 0.6:
            if settings['contrast']['enabled']:
                image = tf.image.random_contrast(
                    image,
                    settings['contrast']['range'][0],
                    settings['contrast']['range'][1],
                    seed=seed+1
                )

        if tf.random.uniform([], seed=seed) < 0.4:
            if settings['saturation']['enabled']:
                image = tf.image.random_saturation(
                    image,
                    settings['saturation']['range'][0],
                    settings['saturation']['range'][1],
                    seed=seed+2
                )

        if tf.random.uniform([], seed=seed) < 0.3:
            if settings['gaussian_noise']['enabled']:
                noise = tf.random.normal(
                    shape=tf.shape(image),
                    mean=0.0,
                    stddev=settings['gaussian_noise']['stddev'],
                    seed=seed+3
                )
                image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return image


@tf.function(reduce_retracing=True)
def apply_pixel_augmentation_map(image, seed, settings):
    """Apply map-specific augmentations based on settings"""
    if len(tf.shape(image)) != 3:
        return image

    if tf.random.uniform([], seed=seed) < settings['prob']:
        if tf.random.uniform([], seed=seed) < 0.6:
            if settings['brightness']['enabled']:
                image = tf.image.random_brightness(
                    image,
                    settings['brightness']['max_delta'],
                    seed=seed
                )

        if tf.random.uniform([], seed=seed) < 0.4:
            if settings['contrast']['enabled']:
                image = tf.image.random_contrast(
                    image,
                    settings['contrast']['range'][0],
                    settings['contrast']['range'][1],
                    seed=seed+1
                )

        if tf.random.uniform([], seed=seed) < 0.3:
            if settings['saturation']['enabled']:
                image = tf.image.random_saturation(
                    image,
                    settings['saturation']['range'][0],
                    settings['saturation']['range'][1],
                    seed=seed+2
                )

        if tf.random.uniform([], seed=seed) < 0.5:
            if settings['gaussian_noise']['enabled']:
                noise = tf.random.normal(
                    shape=tf.shape(image),
                    mean=0.0,
                    stddev=settings['gaussian_noise']['stddev'] * 0.5,  # Reduced intensity
                    seed=seed+3
                )
                image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return image


class SDXLWoundGenerator:
    """
    Manages SDXL checkpoint loading and image generation

    Loads trained SDXL checkpoints and generates wound images for augmentation.
    Supports phase-specific generation (I, P, R) using minimal prompts.
    """

    def __init__(self, checkpoint_path: str, config_path: str, device=None):
        """
        Initialize SDXL generator from checkpoint

        Args:
            checkpoint_path: Path to trained checkpoint (.pt file)
            config_path: Path to training config (full_sdxl_config.yaml)
            device: torch device to use (defaults to CUDA if available)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config
        with open(self.config_path, 'r') as f:
            self.train_config = yaml.safe_load(f)

        # Extract settings
        self.base_model = self.train_config['model']['base_model']
        self.resolution = self.train_config['model']['resolution']
        self.guidance_scale = self.train_config['quality']['guidance_scale']
        self.phase_prompts = self.train_config['prompts']['phase_prompts']
        self.negative_prompt = self.train_config['prompts']['negative']

        # Pipeline will be loaded lazily
        self.pipeline = None
        self.lock = threading.Lock()

    def load_pipeline(self):
        """Load SDXL pipeline from checkpoint"""
        if self.pipeline is not None:
            return self.pipeline

        with self.lock:
            # Double-check after acquiring lock
            if self.pipeline is not None:
                return self.pipeline

            try:
                print(f"Loading SDXL checkpoint from {self.checkpoint_path}")

                # Load checkpoint
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

                # Load base SDXL components
                print(f"Loading base model: {self.base_model}")
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    self.base_model,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True
                )

                # Load trained UNet weights
                print("Loading trained UNet weights...")
                unet_state = checkpoint['unet_lora_state']
                pipeline.unet.load_state_dict(unet_state)

                # Move to device
                pipeline = pipeline.to(self.device)

                # Enable optimizations
                pipeline.enable_attention_slicing()
                if hasattr(pipeline, 'enable_vae_slicing'):
                    pipeline.enable_vae_slicing()

                # Disable progress bar
                pipeline.set_progress_bar_config(disable=True)

                self.pipeline = pipeline
                print(f"SDXL pipeline loaded successfully on {self.device}")

                return self.pipeline

            except Exception as e:
                print(f"Error loading SDXL checkpoint: {str(e)}")
                traceback.print_exc()
                return None

    def generate(self, phase: str, batch_size: int = 1, num_inference_steps: int = 50,
                 target_height: int = None, target_width: int = None):
        """
        Generate wound images for a specific phase

        Args:
            phase: Wound phase ('I', 'P', or 'R')
            batch_size: Number of images to generate
            num_inference_steps: Number of diffusion steps
            target_height: Target height for generated images (default: training resolution)
            target_width: Target width for generated images (default: training resolution)

        Returns:
            tf.Tensor: Generated images [batch_size, height, width, 3] in [0, 1] range
        """
        pipeline = self.load_pipeline()
        if pipeline is None:
            return None

        # Use target size if specified, otherwise use training resolution
        height = target_height if target_height is not None else self.resolution
        width = target_width if target_width is not None else self.resolution

        try:
            # Get phase-specific prompt
            prompt = self.phase_prompts[phase]

            with torch.no_grad():
                output = pipeline(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    num_images_per_prompt=batch_size,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    guidance_scale=self.guidance_scale,
                ).images

            # Convert PIL images to normalized numpy arrays
            tensors = [np.array(img).astype(np.float32) / 255.0 for img in output]
            return tf.convert_to_tensor(np.stack(tensors), dtype=tf.float32)

        except Exception as e:
            print(f"Error generating images for phase {phase}: {str(e)}")
            traceback.print_exc()
            return None

    def cleanup(self):
        """Release pipeline and GPU memory"""
        with self.lock:
            if self.pipeline is not None:
                del self.pipeline
                self.pipeline = None
            torch.cuda.empty_cache()
            gc.collect()


class GenerativeAugmentationManager:
    """
    Manages SDXL-based generative augmentation for wound images

    This replaces the old SD 1.5 LoRA-based manager with SDXL full fine-tuning.
    Maintains the same interface for compatibility with existing code.

    Supports multi-GPU generation for faster augmentation. Each GPU loads its own
    SDXL pipeline (~10GB) and requests are distributed across GPUs.
    """

    def __init__(self, checkpoint_dir: str, config: AugmentationConfig, device=None, num_gpus: int = None):
        """
        Initialize augmentation manager

        Args:
            checkpoint_dir: Directory containing SDXL checkpoint and config
            config: AugmentationConfig instance
            device: torch device to use (ignored if num_gpus > 1)
            num_gpus: Number of GPUs to use for SDXL (default: from production_config)
        """
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.generators = []  # List of generators, one per GPU
        self.generator = None  # For backward compatibility (points to first generator)
        self.lock = threading.Lock()
        self.gpu_counter = 0  # For round-robin GPU selection

        if not self.config.modality_settings['depth_rgb']['generative_augmentations']['enabled']:
            print("Generative augmentation disabled in config")
            return

        # Check if checkpoint directory exists
        if not self.checkpoint_dir.exists():
            print(f"WARNING: Checkpoint directory not found: {checkpoint_dir}")
            print("Generative augmentation will be disabled until checkpoints are available")
            print("Please copy checkpoint files as described in LOCAL_AGENT_MIGRATION_INSTRUCTIONS.md")
            return

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Find checkpoint and config
        checkpoint_path = self.checkpoint_dir / "checkpoint_best.pt"
        if not checkpoint_path.exists():
            # Fallback to latest
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
        if not checkpoint_path.exists():
            # Try to find any checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if checkpoints:
                checkpoint_path = checkpoints[-1]  # Use latest epoch
            else:
                print(f"WARNING: No checkpoint .pt files found in {checkpoint_dir}")
                print("Found files:", list(self.checkpoint_dir.iterdir()))
                print("Generative augmentation will be disabled until checkpoints are available")
                print("Please copy checkpoint_epoch_*.pt files (~10GB each) as described in:")
                print("  LOCAL_AGENT_MIGRATION_INSTRUCTIONS.md")
                return

        config_path = self.checkpoint_dir / "full_sdxl_config.yaml"
        if not config_path.exists():
            # Try parent directory
            config_path = self.checkpoint_dir.parent / "full_sdxl_config.yaml"
        if not config_path.exists():
            # Try src/utils
            config_path = Path(__file__).parent.parent / "utils" / "full_sdxl_config.yaml"
        if not config_path.exists():
            print(f"WARNING: Config file not found near {checkpoint_dir}")
            print("Generative augmentation will be disabled")
            return

        print(f"✓ Found checkpoint: {checkpoint_path}")
        print(f"✓ Found config: {config_path}")

        # Store paths for lazy loading (don't load SDXL yet to save resources)
        self.checkpoint_path = str(checkpoint_path)
        self.config_path = str(config_path)
        self.generator_initialized = False

        # Determine number of GPUs to use for SDXL
        # Import here to avoid circular import
        try:
            from src.utils.production_config import GENERATIVE_AUG_NUM_GPUS
            default_num_gpus = GENERATIVE_AUG_NUM_GPUS
        except (ImportError, AttributeError):
            default_num_gpus = 1

        self.num_gpus = num_gpus if num_gpus is not None else default_num_gpus

        # Limit to available GPUs
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            self.num_gpus = min(self.num_gpus, available_gpus)
        else:
            self.num_gpus = 1

        # Store device for backward compatibility (single GPU case)
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Modality mapping (both thermal_rgb and depth_rgb use same RGB model)
        self.modality_mapping = {
            'thermal_rgb': 'rgb',
            'depth_rgb': 'rgb',
            'thermal_map': 'thermal_map',
            'depth_map': 'depth_map'
        }

        print(f"✓ SDXL paths configured (will load on first use to conserve resources)")
        print(f"✓ Multi-GPU: Will use {self.num_gpus} GPU(s) for SDXL generation")

    def _ensure_generator_loaded(self):
        """Lazy-load SDXL generator(s) on first use to conserve system resources

        For multi-GPU mode, loads one SDXL pipeline per GPU. Each pipeline uses ~10GB VRAM.
        """
        if len(self.generators) > 0 or self.generator_initialized:
            return len(self.generators) > 0

        if not hasattr(self, 'checkpoint_path') or not hasattr(self, 'config_path'):
            return False

        with self.lock:
            # Double-check after acquiring lock
            if len(self.generators) > 0 or self.generator_initialized:
                return len(self.generators) > 0

            try:
                print(f"[MULTI-GPU] Loading SDXL on {self.num_gpus} GPU(s)...")

                for gpu_idx in range(self.num_gpus):
                    device = torch.device(f"cuda:{gpu_idx}")
                    print(f"  Loading SDXL generator on GPU {gpu_idx}...")

                    generator = SDXLWoundGenerator(
                        checkpoint_path=self.checkpoint_path,
                        config_path=self.config_path,
                        device=device
                    )
                    self.generators.append(generator)
                    print(f"  ✓ GPU {gpu_idx}: SDXL loaded ({torch.cuda.get_device_name(gpu_idx)})")

                # Backward compatibility: point to first generator
                self.generator = self.generators[0] if self.generators else None
                self.generator_initialized = True

                print(f"✓ SDXL generators loaded on {len(self.generators)} GPU(s)")
                return True

            except Exception as e:
                print(f"ERROR: Failed to load SDXL generator(s): {str(e)}")
                traceback.print_exc()
                self.generator_initialized = True  # Mark as attempted
                self.generators = []
                self.generator = None
                return False

    def _get_next_generator(self):
        """Get the next generator using round-robin selection"""
        if not self.generators:
            return None
        with self.lock:
            generator = self.generators[self.gpu_counter % len(self.generators)]
            self.gpu_counter += 1
        return generator

    def generate_images(self, modality: str, phase: str, batch_size: int = 1,
                       target_height: int = None, target_width: int = None):
        """
        Generate wound images for augmentation

        Args:
            modality: Input modality (depth_rgb, thermal_rgb, etc.)
            phase: Wound phase ('I', 'P', or 'R')
            batch_size: Number of images to generate
            target_height: Target height for generated images (from main.py IMAGE_SIZE)
            target_width: Target width for generated images (from main.py IMAGE_SIZE)

        Returns:
            tf.Tensor: Generated images in [0, 1] range at target size
        """
        if not self.config.modality_settings['depth_rgb']['generative_augmentations']['enabled']:
            return None

        # Lazy-load generator on first use
        if not self._ensure_generator_loaded():
            return None

        # Limit batch size
        batch_size = min(batch_size, self.config.generative_settings['batch_size_limit'])

        # Use config output size if target size not specified
        if target_height is None:
            target_height = self.config.generative_settings['output_size']['height']
        if target_width is None:
            target_width = self.config.generative_settings['output_size']['width']

        try:
            # Get next generator (round-robin across GPUs)
            generator = self._get_next_generator()
            if generator is None:
                print("WARNING: No SDXL generators available")
                return None

            # Generate images directly at target size to save computation time
            generated = generator.generate(
                phase=phase,
                batch_size=batch_size,
                num_inference_steps=self.config.generative_settings['inference_steps'],
                target_height=target_height,
                target_width=target_width
            )

            if generated is None:
                return None

            # Update counter with per-class tracking
            _gen_image_counter.increment(batch_size, phase=phase)

            return generated

        except Exception as e:
            print(f"Error generating images for {modality}_{phase}: {str(e)}")
            traceback.print_exc()
            return None

    def should_generate(self, modality: str):
        """
        Determine if generative augmentation should be applied

        Args:
            modality: Input modality

        Returns:
            bool: True if should generate, False otherwise
        """
        if not self.config.modality_settings['depth_rgb']['generative_augmentations']['enabled']:
            return False

        # Check if we have valid checkpoint paths configured
        if not hasattr(self, 'checkpoint_path') or not hasattr(self, 'config_path'):
            return False

        # Check if current phase is in the enabled phases list
        if not hasattr(self, 'current_phase'):
            return False

        try:
            phase_str = self.current_phase if isinstance(self.current_phase, str) else self.current_phase.numpy().decode('utf-8')
        except:
            return False

        if phase_str not in GENERATIVE_AUG_PHASES:
            return False

        return (self.config.modality_settings[modality]['generative_augmentations']['enabled'] and
                tf.random.uniform([], 0, 1) < self.config.generative_settings['prob'])

    def cleanup(self):
        """Release all resources and GPU memory from all GPUs"""
        if not self.config.modality_settings['depth_rgb']['generative_augmentations']['enabled']:
            return

        # Print generation summary before cleanup
        _gen_image_counter.print_summary()

        with self.lock:
            # Clean up all generators (multi-GPU support)
            for i, generator in enumerate(self.generators):
                if generator is not None:
                    print(f"  Cleaning up SDXL generator on GPU {i}...")
                    generator.cleanup()
            self.generators = []
            self.generator = None
            torch.cuda.empty_cache()
            gc.collect()

            # TensorFlow cleanup
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for i, _ in enumerate(gpus):
                    tf.config.experimental.reset_memory_stats(f'GPU:{i}')


def create_enhanced_augmentation_fn(gen_manager, config):
    """
    Create TensorFlow augmentation function with generative augmentation

    This maintains compatibility with the existing training pipeline.
    """
    def apply_augmentation(features, label):
        def augment_batch(features_dict, label_tensor):
            output_features = {}

            @tf.function
            def get_label_phase(label):
                index = tf.argmax(label)
                phases = tf.constant(['I', 'P', 'R'], dtype=tf.string)
                return tf.gather(phases, index)

            try:
                current_phase = get_label_phase(label_tensor[0])
            except Exception as e:
                print(f"Error getting phase: {str(e)}")
                current_phase = tf.constant('I', dtype=tf.string)

            for key, value in features_dict.items():
                if '_input' in key and 'metadata' not in key:
                    modality = key.replace('_input', '')

                    def decode_phase(phase_tensor):
                        return phase_tensor.numpy().decode('utf-8')

                    gen_manager.current_phase = tf.py_function(decode_phase, [current_phase], tf.string)

                    # Apply regular augmentations FIRST (before SDXL mixing)
                    # This ensures SDXL-generated images bypass regular augmentations
                    if config.modality_settings[modality]['regular_augmentations']['enabled']:
                        seed = tf.random.uniform([], maxval=1000000, dtype=tf.int32)
                        value = augment_image(value, modality, seed, config)

                    if gen_manager.should_generate(modality):
                        try:
                            def generate_images_wrapper(phase_tensor, modality_str, batch_size_val, target_height, target_width):
                                try:
                                    phase = phase_tensor.numpy().decode('utf-8')
                                    modality_raw = modality_str.numpy().decode('utf-8')
                                    batch_size = int(batch_size_val.numpy())
                                    height = int(target_height.numpy())
                                    width = int(target_width.numpy())

                                    # Generate directly at target size (no resizing needed)
                                    generated = gen_manager.generate_images(
                                        modality_raw, phase,
                                        batch_size=batch_size,
                                        target_height=height,
                                        target_width=width
                                    )

                                    if generated is None:
                                        return np.zeros([batch_size, height, width, 3], dtype=np.float32)

                                    return generated.numpy()

                                except Exception as e:
                                    print(f"Error in generate_images_wrapper: {str(e)}")
                                    return np.zeros([batch_size, height, width, 3], dtype=np.float32)

                            modality_tensor = tf.constant(modality, dtype=tf.string)
                            batch_size = tf.shape(value)[0]
                            height = tf.shape(value)[1]
                            width = tf.shape(value)[2]

                            generated = tf.py_function(
                                func=generate_images_wrapper,
                                inp=[current_phase, modality_tensor, batch_size, height, width],
                                Tout=tf.float32
                            )
                            generated.set_shape([None, value.shape[1], value.shape[2], 3])

                            mix_ratio = tf.random.uniform([],
                                minval=config.generative_settings['mix_ratio_range'][0],
                                maxval=config.generative_settings['mix_ratio_range'][1]
                            )

                            num_to_replace = tf.cast(
                                tf.cast(tf.shape(value)[0], tf.float32) * mix_ratio,
                                tf.int32
                            )

                            indices = tf.random.shuffle(tf.range(tf.shape(value)[0]))[:num_to_replace]
                            updates = tf.gather(generated, tf.range(tf.minimum(num_to_replace, tf.shape(generated)[0])))
                            indices = tf.expand_dims(indices[:tf.shape(updates)[0]], 1)

                            value = tf.tensor_scatter_nd_update(value, indices, updates)

                        except Exception as e:
                            print(f"Error in generative augmentation: {str(e)}")

                output_features[key] = value

            return output_features, label_tensor

        return augment_batch(features, label)

    return apply_augmentation


class GenerativeAugmentationCallback(tf.keras.callbacks.Callback):
    """Callback to manage generative augmentation resources"""
    def __init__(self, gen_manager):
        super().__init__()
        self.gen_manager = gen_manager

    def on_train_end(self, logs=None):
        # Final cleanup
        self.gen_manager.cleanup()
