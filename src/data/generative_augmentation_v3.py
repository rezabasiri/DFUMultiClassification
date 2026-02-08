"""
Generative Augmentation V3 - SDXL Conditional Model

This module implements generative augmentation using a single conditional SDXL model
that was fine-tuned on all DFU wound phases. Unlike V2 which uses separate SD 1.5 models
per phase, V3 uses phase-specific prompts to condition a single SDXL model.

Key differences from V2:
- Single SDXL model instead of multiple SD 1.5 models
- Phase-conditional generation via prompts (PHASE_I, PHASE_P, PHASE_R)
- 512x512 native resolution (can be dynamically adjusted)
- Dual text encoders (CLIP-L + OpenCLIP-bigG)
- Full fine-tuned model (not LoRA)

Usage:
    from src.data.generative_augmentation_v3 import (
        GenerativeAugmentationManagerSDXL,
        GenerativeAugmentationCallback,
        AugmentationConfig,
        create_enhanced_augmentation_fn
    )
"""

import os
import gc
from pathlib import Path
import torch
import torchvision.transforms as transforms
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import logging as diffusers_logging
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
import random
import tensorflow as tf
import traceback
import numpy as np
import threading

from src.utils.production_config import (
    IMAGE_SIZE,
    USE_GENERATIVE_AUGMENTATION,
    USE_GENERAL_AUGMENTATION,
    GENERATIVE_AUG_PROB,
    GENERATIVE_AUG_MIX_RATIO,
    GENERATIVE_AUG_INFERENCE_STEPS,
    GENERATIVE_AUG_BATCH_LIMIT,
    GENERATIVE_AUG_PHASES,
    GENERATIVE_AUG_SDXL_MODEL_PATH,
    GENERATIVE_AUG_SDXL_GUIDANCE_SCALE,
    GENERATIVE_AUG_SDXL_RESOLUTION,
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
    """Thread-safe counter for tracking generated images (total and per-phase)"""
    def __init__(self):
        self.total_count = 0
        self.phase_counts = {'I': 0, 'P': 0, 'R': 0}
        self.lock = threading.Lock()
        self.print_interval = 50  # Print summary every N images

    def increment(self, phase, batch_size=1):
        """Increment counter for a specific phase and optionally print update"""
        with self.lock:
            self.total_count += batch_size
            if phase in self.phase_counts:
                self.phase_counts[phase] += batch_size

            if self.total_count % self.print_interval == 0:
                self._print_summary()

    def _print_summary(self):
        """Print current generation summary (must be called with lock held)"""
        phase_str = ", ".join([f"{p}:{c}" for p, c in self.phase_counts.items()])
        print(f"  [SDXL Gen] Total: {self.total_count} | By phase: {phase_str}", flush=True)

    def reset(self):
        """Reset all counters to 0"""
        with self.lock:
            self.total_count = 0
            self.phase_counts = {'I': 0, 'P': 0, 'R': 0}

    def get_count(self):
        """Get total count"""
        with self.lock:
            return self.total_count

    def get_phase_counts(self):
        """Get counts per phase"""
        with self.lock:
            return self.phase_counts.copy()

    def print_final_summary(self):
        """Print final summary of all generated images"""
        with self.lock:
            print("\n" + "="*60)
            print("SDXL Generative Augmentation Summary")
            print("="*60)
            print(f"  Total generated images: {self.total_count}")
            print(f"  Per-phase breakdown:")
            for phase, count in self.phase_counts.items():
                phase_name = {'I': 'Inflammatory', 'P': 'Proliferative', 'R': 'Remodeling'}[phase]
                pct = (count / self.total_count * 100) if self.total_count > 0 else 0
                print(f"    Phase {phase} ({phase_name}): {count} ({pct:.1f}%)")
            print("="*60 + "\n")

# Global counter instance
_gen_image_counter = GeneratedImageCounter()


def get_generation_stats():
    """Get current generation statistics (for external access)"""
    return {
        'total': _gen_image_counter.get_count(),
        'by_phase': _gen_image_counter.get_phase_counts()
    }


def print_generation_summary():
    """Print final generation summary (call at end of training)"""
    _gen_image_counter.print_final_summary()


def reset_generation_counter():
    """Reset generation counter (call at start of training)"""
    _gen_image_counter.reset()


class AugmentationConfig:
    """Configuration for augmentation settings"""
    def __init__(self):
        # Per-modality settings
        # Note: depth_rgb setting acts as master switch for generative augmentation
        # Both depth_rgb and thermal_rgb use the same SDXL model with phase-specific prompts
        # USE_GENERAL_AUGMENTATION controls regular augmentations (brightness, contrast, etc.)
        # USE_GENERATIVE_AUGMENTATION controls generative augmentations (SDXL model)
        self.modality_settings = {
            'depth_rgb': {
                'regular_augmentations': {
                    'enabled': USE_GENERAL_AUGMENTATION,  # Controlled by production_config
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
                    'enabled': USE_GENERAL_AUGMENTATION,  # Controlled by production_config
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
                    'enabled': USE_GENERAL_AUGMENTATION,  # Controlled by production_config
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
                    'enabled': USE_GENERAL_AUGMENTATION,  # Controlled by production_config
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
            'sdxl_resolution': GENERATIVE_AUG_SDXL_RESOLUTION,
            'guidance_scale': GENERATIVE_AUG_SDXL_GUIDANCE_SCALE,
        }


# Phase-specific prompts for SDXL conditional generation
# CRITICAL: These MUST match exactly what was used during training
SDXL_PHASE_PROMPTS = {
    'I': "PHASE_I, diabetic foot ulcer, inflammatory phase wound",
    'P': "PHASE_P, diabetic foot ulcer, proliferative phase wound",
    'R': "PHASE_R, diabetic foot ulcer, remodeling phase wound",
}

SDXL_NEGATIVE_PROMPT = (
    "blurry, out of focus, low quality, jpeg artifacts, "
    "cartoon, illustration, painting, anime, "
    "text, watermark, logo"
)


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
        return augmented

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
            # Note: Saturation is typically disabled for map images in your config
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


class GenerativeAugmentationManagerSDXL:
    """
    Manager for SDXL-based generative augmentation.

    Unlike V2 which loads separate SD 1.5 models per phase, this manager
    loads a single SDXL model that was fine-tuned on all phases and uses
    phase-specific prompts to condition generation.
    """

    def __init__(self, checkpoint_path, config, device=None):
        """
        Initialize the SDXL generative augmentation manager.

        Args:
            checkpoint_path (str): Path to the SDXL checkpoint file (.pt)
            config (AugmentationConfig): Configuration object for augmentation
            device (torch.device): Device to run the model on, defaults to GPU if available
        """
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.lock = threading.Lock()
        self.current_phase = None

        # Only initialize if generative augmentation is enabled
        if self.config.modality_settings['depth_rgb']['generative_augmentations']['enabled']:
            self._load_model()

    def _load_model(self):
        """Load the SDXL model from checkpoint"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SDXL checkpoint not found: {self.checkpoint_path}")

        print(f"Loading SDXL model from: {self.checkpoint_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)

            # Get base model ID (SDXL 1.0 base)
            base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

            # Determine dtype based on available hardware
            if torch.cuda.is_available():
                weight_dtype = torch.float16  # Use fp16 for inference on GPU
            else:
                weight_dtype = torch.float32

            # Load VAE
            vae = AutoencoderKL.from_pretrained(
                base_model_id,
                subfolder="vae",
                torch_dtype=weight_dtype
            )

            # Load text encoders
            text_encoder = CLIPTextModel.from_pretrained(
                base_model_id,
                subfolder="text_encoder",
                torch_dtype=weight_dtype
            )
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                base_model_id,
                subfolder="text_encoder_2",
                torch_dtype=weight_dtype
            )

            # Load tokenizers
            tokenizer = CLIPTokenizer.from_pretrained(
                base_model_id,
                subfolder="tokenizer"
            )
            tokenizer_2 = CLIPTokenizer.from_pretrained(
                base_model_id,
                subfolder="tokenizer_2"
            )

            # Load UNet and restore fine-tuned weights
            unet = UNet2DConditionModel.from_pretrained(
                base_model_id,
                subfolder="unet",
                torch_dtype=weight_dtype
            )

            # Load fine-tuned UNet weights from checkpoint
            if 'unet_lora_state' in checkpoint:
                unet.load_state_dict(checkpoint['unet_lora_state'], strict=False)
                print("  Loaded fine-tuned UNet weights from checkpoint")
            else:
                print("  Warning: No fine-tuned weights found in checkpoint, using base model")

            # Load noise scheduler
            noise_scheduler = DDPMScheduler.from_pretrained(
                base_model_id,
                subfolder="scheduler"
            )

            # Create pipeline
            self.pipeline = StableDiffusionXLPipeline(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                unet=unet,
                scheduler=noise_scheduler,
            )

            # Move to device and optimize
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.set_progress_bar_config(disable=True)

            # Enable memory optimizations
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()

            print(f"  SDXL model loaded successfully on {self.device}")

        except Exception as e:
            print(f"Error loading SDXL model: {str(e)}")
            traceback.print_exc()
            self.pipeline = None

    def generate_images(self, modality, phase, batch_size=1, target_height=None, target_width=None):
        """
        Generate images using the SDXL model with phase-specific prompts.

        Images are always generated at native SDXL resolution (512x512) for quality,
        then resized to target dimensions if different.

        Args:
            modality (str): The input modality (e.g., 'thermal_rgb', 'depth_rgb')
            phase (str): The wound healing phase ('I', 'P', or 'R')
            batch_size (int): Number of images to generate
            target_height (int): Target height for output images (will resize if different from SDXL native)
            target_width (int): Target width for output images (will resize if different from SDXL native)

        Returns:
            tf.Tensor: Generated images as a TensorFlow tensor [batch_size, target_height, target_width, 3]
        """
        if self.pipeline is None:
            return None

        if not self.config.modality_settings['depth_rgb']['generative_augmentations']['enabled']:
            return None

        batch_size = min(batch_size, self.config.generative_settings['batch_size_limit'])

        # Always generate at native SDXL resolution for quality
        # SDXL was trained at 512x512, generating at smaller sizes produces poor results
        native_resolution = self.config.generative_settings['sdxl_resolution']

        # Determine final output size (default to IMAGE_SIZE from config)
        final_height = target_height if target_height is not None else self.config.generative_settings['output_size']['height']
        final_width = target_width if target_width is not None else self.config.generative_settings['output_size']['width']

        # Get phase-specific prompt
        if phase not in SDXL_PHASE_PROMPTS:
            print(f"Warning: Unknown phase '{phase}', defaulting to 'I'")
            phase = 'I'

        prompt = SDXL_PHASE_PROMPTS[phase]
        negative_prompt = SDXL_NEGATIVE_PROMPT

        try:
            with self.lock:  # Thread-safe generation
                with torch.no_grad():
                    # Use autocast for memory efficiency
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        output = self.pipeline(
                            prompt=[prompt] * batch_size,
                            negative_prompt=[negative_prompt] * batch_size,
                            num_inference_steps=self.config.generative_settings['inference_steps'],
                            guidance_scale=self.config.generative_settings['guidance_scale'],
                            height=native_resolution,
                            width=native_resolution,
                        ).images

            # Update generated image counter with phase info
            _gen_image_counter.increment(phase, batch_size)

            # Convert PIL images to normalized numpy arrays [0, 1]
            tensors = [np.array(img).astype(np.float32) / 255.0 for img in output]
            images_tensor = tf.convert_to_tensor(np.stack(tensors), dtype=tf.float32)

            # Resize to target dimensions if different from native resolution
            if final_height != native_resolution or final_width != native_resolution:
                images_tensor = tf.image.resize(
                    images_tensor,
                    [final_height, final_width],
                    method=tf.image.ResizeMethod.BILINEAR
                )

            return images_tensor

        except Exception as e:
            print(f"Error generating images for phase {phase}: {str(e)}")
            traceback.print_exc()
            return None

    def should_generate(self, modality):
        """
        Determine if generative augmentation should be applied.

        Args:
            modality (str): The modality to check

        Returns:
            bool: True if generative augmentation should be applied
        """
        if self.pipeline is None:
            return False

        if not self.config.modality_settings['depth_rgb']['generative_augmentations']['enabled']:
            return False

        # Check if current phase is in the enabled phases list
        if not hasattr(self, 'current_phase') or self.current_phase is None:
            return False

        # Decode phase string if it's a TensorFlow tensor
        try:
            phase_str = self.current_phase if isinstance(self.current_phase, str) else self.current_phase.numpy().decode('utf-8')
        except:
            return False

        if phase_str not in GENERATIVE_AUG_PHASES:
            return False

        return (self.config.modality_settings[modality]['generative_augmentations']['enabled'] and
                tf.random.uniform([], 0, 1) < self.config.generative_settings['prob'])

    def cleanup(self):
        """Release model and GPU memory"""
        if self.pipeline is not None:
            with self.lock:
                # Move components to CPU first
                if hasattr(self.pipeline, 'unet'):
                    self.pipeline.unet.to('cpu')
                if hasattr(self.pipeline, 'vae'):
                    self.pipeline.vae.to('cpu')
                if hasattr(self.pipeline, 'text_encoder'):
                    self.pipeline.text_encoder.to('cpu')
                if hasattr(self.pipeline, 'text_encoder_2'):
                    self.pipeline.text_encoder_2.to('cpu')

                del self.pipeline
                self.pipeline = None

                torch.cuda.empty_cache()
                gc.collect()

                # TensorFlow-specific cleanup
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for i, _ in enumerate(gpus):
                        try:
                            tf.config.experimental.reset_memory_stats(f'GPU:{i}')
                        except:
                            pass


def create_enhanced_augmentation_fn(gen_manager, config):
    """
    Create augmentation function that applies generative and/or regular augmentations.

    Pipeline flow:
    1. Real images: bbox cropped at load time -> general augmentation here
    2. Generated images: created at target size -> NO general augmentation (already diverse)

    Args:
        gen_manager: GenerativeAugmentationManagerSDXL instance, or None for regular augmentations only
        config: AugmentationConfig instance

    Returns:
        Augmentation function that can be applied to batched datasets
    """
    # Track how many generated sample images have been saved (max 3)
    _saved_gen_samples = [0]

    def apply_augmentation(features, label):
        def augment_batch(features_dict, label_tensor):
            output_features = {}

            @tf.function
            def get_label_phase(label):
                index = tf.argmax(label)
                phases = tf.constant(['I', 'P', 'R'], dtype=tf.string)
                return tf.gather(phases, index)

            # Only compute phase if gen_manager is available
            current_phase = None
            if gen_manager is not None:
                try:
                    current_phase = get_label_phase(label_tensor[0])
                except Exception as e:
                    print(f"Error getting phase: {str(e)}")
                    current_phase = tf.constant('I', dtype=tf.string)

            for key, value in features_dict.items():
                if '_input' in key and 'metadata' not in key:
                    modality = key.replace('_input', '')
                    batch_size = tf.shape(value)[0]

                    # Track which indices have generated images (should NOT be augmented)
                    generated_indices_mask = tf.zeros([batch_size], dtype=tf.bool)

                    # Apply generative augmentation only if gen_manager is available
                    if gen_manager is not None:
                        # All generation logic runs inside tf.py_function for reliable eager execution.
                        # Python if-statements in the outer scope are evaluated during TF graph tracing,
                        # where .numpy() fails on symbolic tensors, causing should_generate() to
                        # silently return False and exclude generation from the graph entirely.
                        def maybe_generate_and_mix(images_batch, phase_tensor, modality_str):
                            """Generate synthetic images and mix into batch (runs eagerly via py_function)"""
                            try:
                                images = images_batch.numpy().copy()
                                modality_raw = modality_str.numpy().decode('utf-8')
                                phase = phase_tensor.numpy().decode('utf-8')
                                bs = images.shape[0]

                                # Set current phase as string for should_generate check
                                gen_manager.current_phase = phase

                                # Create mask (0=real, 1=generated)
                                mask = np.zeros(bs, dtype=np.float32)

                                if not gen_manager.should_generate(modality_raw):
                                    return images, mask

                                height, width = images.shape[1], images.shape[2]
                                generated = gen_manager.generate_images(
                                    modality_raw, phase,
                                    batch_size=bs,
                                    target_height=height,
                                    target_width=width
                                )
                                if generated is None:
                                    return images, mask

                                gen_np = generated.numpy() if hasattr(generated, 'numpy') else np.array(generated)

                                # Save first 3 generated images for visual verification
                                if _saved_gen_samples[0] < 3:
                                    try:
                                        from PIL import Image
                                        save_dir = os.path.join('results', 'visualizations')
                                        os.makedirs(save_dir, exist_ok=True)
                                        for si in range(min(len(gen_np), 3 - _saved_gen_samples[0])):
                                            _saved_gen_samples[0] += 1
                                            img = (gen_np[si] * 255).clip(0, 255).astype(np.uint8)
                                            save_path = os.path.join(save_dir, f'gen{_saved_gen_samples[0]}.png')
                                            Image.fromarray(img).save(save_path)
                                            print(f"  Saved generated sample: {save_path} (phase: {phase}, modality: {modality_raw})", flush=True)
                                    except Exception as save_err:
                                        print(f"  Warning: Could not save generated sample: {save_err}")

                                # Determine how many images to replace (mix_ratio range from config)
                                mix_ratio = np.random.uniform(
                                    config.generative_settings['mix_ratio_range'][0],
                                    config.generative_settings['mix_ratio_range'][1]
                                )
                                num_to_replace = max(1, int(bs * mix_ratio))
                                num_to_replace = min(num_to_replace, len(gen_np))

                                # Replace random indices with generated images
                                indices = np.random.choice(bs, size=num_to_replace, replace=False)
                                for i, idx in enumerate(indices):
                                    if i < len(gen_np):
                                        images[idx] = gen_np[i]
                                        mask[idx] = 1.0

                                return images, mask

                            except Exception as e:
                                print(f"Error in generative augmentation: {str(e)}")
                                import traceback
                                traceback.print_exc()
                                return images_batch.numpy(), np.zeros(images_batch.shape[0], dtype=np.float32)

                        modality_tensor = tf.constant(modality, dtype=tf.string)
                        result_images, gen_mask = tf.py_function(
                            func=maybe_generate_and_mix,
                            inp=[value, current_phase, modality_tensor],
                            Tout=[tf.float32, tf.float32]
                        )
                        result_images.set_shape(value.shape)
                        gen_mask.set_shape([None])

                        value = result_images
                        generated_indices_mask = tf.cast(gen_mask, tf.bool)

                    # Apply regular augmentations ONLY to real images (not generated ones)
                    if modality in config.modality_settings and config.modality_settings[modality]['regular_augmentations']['enabled']:
                        seed = tf.random.uniform([], maxval=1000000, dtype=tf.int32)

                        # Augment the entire batch
                        augmented_value = augment_image(value, modality, seed, config)

                        # Use mask to keep generated images unaugmented, augment only real images
                        # Where mask is True (generated), use original value; where False (real), use augmented
                        real_mask = tf.logical_not(generated_indices_mask)
                        real_mask_expanded = tf.reshape(real_mask, [-1, 1, 1, 1])
                        real_mask_expanded = tf.broadcast_to(real_mask_expanded, tf.shape(value))

                        value = tf.where(real_mask_expanded, augmented_value, value)

                output_features[key] = value

            return output_features, label_tensor

        return augment_batch(features, label)

    return apply_augmentation


class GenerativeAugmentationCallback(tf.keras.callbacks.Callback):
    """Callback to manage generative augmentation resources and print summary"""
    def __init__(self, gen_manager):
        super().__init__()
        self.gen_manager = gen_manager

    def on_train_begin(self, logs=None):
        # Reset counter at start of training
        reset_generation_counter()

    def on_train_end(self, logs=None):
        # Print final generation summary
        if _gen_image_counter.get_count() > 0:
            print_generation_summary()

        # Final cleanup
        if self.gen_manager is not None:
            self.gen_manager.cleanup()


# Alias for backward compatibility with V2 imports
GenerativeAugmentationManager = GenerativeAugmentationManagerSDXL
