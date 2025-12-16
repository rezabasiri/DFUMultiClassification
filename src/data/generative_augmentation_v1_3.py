import os
import gc
from pathlib import Path
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline
import random
import tensorflow as tf
import traceback
import numpy as np
import threading

class AugmentationConfig:
    def __init__(self):
        self.modality_settings = {
            'depth_rgb': {
                'regular_augmentations': {
                    'enabled': True,
                    'prob': 0.6,
                    'brightness': {'enabled': True, 'max_delta': 0.2},
                    'contrast': {'enabled': True, 'range': (0.8, 1.2)},
                    'saturation': {'enabled': True, 'range': (0.8, 1.2)},
                    'gaussian_noise': {'enabled': True, 'stddev': 0.03},
                    'gamma': {'enabled': True, 'range': (0.8, 1.2)},
                    'jpeg_quality': {'enabled': True, 'range': (85, 100)}
                },
                'generative_augmentations': {
                    'enabled': True
                }
            },
            'thermal_rgb': {
                'regular_augmentations': {
                    'enabled': True,
                    'prob': 0.6,
                    'brightness': {'enabled': True, 'max_delta': 0.2},
                    'contrast': {'enabled': True, 'range': (0.8, 1.2)},
                    'saturation': {'enabled': True, 'range': (0.8, 1.2)},
                    'gaussian_noise': {'enabled': True, 'stddev': 0.03},
                    'gamma': {'enabled': True, 'range': (0.8, 1.2)},
                    'jpeg_quality': {'enabled': True, 'range': (85, 100)}
                },
                'generative_augmentations': {
                    'enabled': True
                }
            },
            'thermal_map': {
                'regular_augmentations': {
                    'enabled': True,
                    'prob': 0.6,
                    'brightness': {'enabled': True, 'max_delta': 0.1},
                    'contrast': {'enabled': True, 'range': (0.9, 1.1)},
                    'saturation': {'enabled': False},
                    'gaussian_noise': {'enabled': True, 'stddev': 0.015},
                    'local_variation': {'enabled': True, 'intensity': 0.02}
                },
                'generative_augmentations': {
                    'enabled': False
                }
            },
            'depth_map': {
                'regular_augmentations': {
                    'enabled': True,
                    'prob': 0.6,
                    'brightness': {'enabled': True, 'max_delta': 0.1},
                    'contrast': {'enabled': True, 'range': (0.9, 1.1)},
                    'saturation': {'enabled': False},
                    'gaussian_noise': {'enabled': True, 'stddev': 0.015},
                    'local_variation': {'enabled': True, 'intensity': 0.02}
                },
                'generative_augmentations': {
                    'enabled': False
                }
            }
        }
        
        # Global generative settings
        self.generative_settings = {
            'output_size': {'height': 64, 'width': 64},
            'prob': 0.4,
            'mix_ratio_range': (0.1, 0.4),
            'inference_steps': 50,
            'batch_size_limit': 40,
            'max_loaded_models': 3
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
            # method=tf.image.ResizeMethod.LANCZOS3
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
    if not config.modality_settings[modality]['regular_augmentations']['enabled']:
        return image
        
    if len(tf.shape(image)) != 4 or tf.shape(image)[-1] != 3:
        return image
        
    seed_val = tf.get_static_value(seed)
    if seed_val is None:
        seed_val = 42
        
    try:
        settings = config.modality_settings[modality]['regular_augmentations']
        if modality in ['depth_rgb', 'thermal_rgb']:
            return apply_pixel_augmentation_rgb(image, seed_val, settings)
        else:
            return apply_pixel_augmentation_map(image, seed_val, settings)
    except Exception as e:
        print(f"Error in augment_image: {str(e)}")
        return image

@tf.function(reduce_retracing=True)
def apply_pixel_augmentation_rgb(image_batch, seed, settings):
    """Apply random combinations of RGB-specific augmentations based on settings"""
    # Ensure we're working with a batch of images [batch, height, width, channels]
    if len(tf.shape(image_batch)) != 4:
        return image_batch
        
    def augment_single_image(image):
        # Only proceed if main probability threshold is met
        if tf.random.uniform([], seed=seed) < settings['prob']:
            # For each augmentation, independently decide whether to apply it
            if settings['brightness']['enabled'] and tf.random.uniform([], seed=seed+1) < 0.5:
                image = tf.image.random_brightness(
                    image, 
                    settings['brightness']['max_delta'], 
                    seed=seed
                )
            
            if settings['contrast']['enabled'] and tf.random.uniform([], seed=seed+2) < 0.5:
                image = tf.image.random_contrast(
                    image,
                    settings['contrast']['range'][0],
                    settings['contrast']['range'][1],
                    seed=seed+1
                )
                
            if settings['saturation']['enabled'] and tf.random.uniform([], seed=seed+3) < 0.5:
                image = tf.image.random_saturation(
                    image,
                    settings['saturation']['range'][0],
                    settings['saturation']['range'][1],
                    seed=seed+2
                )

            if settings['gaussian_noise']['enabled'] and tf.random.uniform([], seed=seed+4) < 0.5:
                noise = tf.random.normal(
                    shape=tf.shape(image),
                    mean=0.0,
                    stddev=settings['gaussian_noise']['stddev'],
                    seed=seed+3
                )
                image = tf.clip_by_value(image + noise, 0.0, 1.0)
                
            if settings['gamma']['enabled'] and tf.random.uniform([], seed=seed+5) < 0.5:
                gamma = tf.random.uniform(
                    [],
                    settings['gamma']['range'][0],
                    settings['gamma']['range'][1],
                    seed=seed+4
                )
                image = tf.pow(image, gamma)
                
            if settings['jpeg_quality']['enabled'] and tf.random.uniform([], seed=seed+6) < 0.5:
                image = tf.cast(image * 255.0, tf.uint8)
                image = tf.image.random_jpeg_quality(
                    image,
                    settings['jpeg_quality']['range'][0],
                    settings['jpeg_quality']['range'][1],
                    seed=seed+5
                )
                image = tf.cast(image, tf.float32) / 255.0
        
        return image
    
    # Apply augmentations to each image in the batch
    return tf.map_fn(augment_single_image, image_batch)

@tf.function(reduce_retracing=True)
def apply_pixel_augmentation_map(image_batch, seed, settings):
    """Apply random combinations of map-specific augmentations based on settings"""
    # Ensure we're working with a batch of images [batch, height, width, channels]
    if len(tf.shape(image_batch)) != 4:
        return image_batch
        
    def augment_single_image(image):
        # Only proceed if main probability threshold is met
        if tf.random.uniform([], seed=seed) < settings['prob']:
            if settings['brightness']['enabled'] and tf.random.uniform([], seed=seed+1) < 0.5:
                image = tf.image.random_brightness(
                    image, 
                    settings['brightness']['max_delta'], 
                    seed=seed
                )
            
            if settings['contrast']['enabled'] and tf.random.uniform([], seed=seed+2) < 0.5:
                image = tf.image.random_contrast(
                    image,
                    settings['contrast']['range'][0],
                    settings['contrast']['range'][1],
                    seed=seed+1
                )
                
            if settings['saturation']['enabled'] and tf.random.uniform([], seed=seed+3) < 0.5:
                image = tf.image.random_saturation(
                    image,
                    settings['saturation']['range'][0],
                    settings['saturation']['range'][1],
                    seed=seed+2
                )

            if settings['gaussian_noise']['enabled'] and tf.random.uniform([], seed=seed+4) < 0.5:
                noise = tf.random.normal(
                    shape=tf.shape(image),
                    mean=0.0,
                    stddev=settings['gaussian_noise']['stddev'] * 0.5,  # Reduced intensity
                    seed=seed+3
                )
                image = tf.clip_by_value(image + noise, 0.0, 1.0)
                
            if settings['local_variation']['enabled'] and tf.random.uniform([], seed=seed+5) < 0.5:
                variation_size = tf.cast(tf.shape(image)[0:2] // 8, tf.int32)
                variation = tf.image.resize(
                    tf.random.normal(
                        [1, variation_size[0], variation_size[1], 1],
                        mean=0.0,
                        stddev=settings['local_variation']['intensity'],
                        seed=seed+4
                    ),
                    tf.shape(image)[0:2]
                )
                image = tf.clip_by_value(image + variation, 0.0, 1.0)
        
        return image
    
    # Apply augmentations to each image in the batch
    return tf.map_fn(augment_single_image, image_batch)

class PromptGenerator:
    def __init__(self):
        # Phase descriptions remain the same...
        self.phase_descriptions = {
            'I': {
                'name': 'Inflammatory',
                'characteristics': [
                    'redness and swelling',
                    'early wound healing',
                    'acute inflammation',
                    'wound debridement phase'
                ]
            },
            'P': {
                'name': 'Proliferative',
                'characteristics': [
                    'granulation tissue formation',
                    'pink to red tissue bed',
                    'active wound healing',
                    'new tissue growth'
                ]
            },
            'R': {
                'name': 'Remodeling',
                'characteristics': [
                    'wound contraction',
                    'epithelialization',
                    'scar tissue formation',
                    'mature healing'
                ]
            }
        }
        
        # Update modality descriptions to match model paths
        self.modality_descriptions = {
            'rgb': 'visible light photograph of',  # Used for both thermal_rgb and depth_rgb
            'depth_map': 'depth mapping of',
            'thermal_map': 'thermal imaging of'
        }
        # Add modality mapping to match GenerativeAugmentationManager
        self.modality_mapping = {
            'thermal_rgb': 'rgb',
            'depth_rgb': 'rgb',
            'thermal_map': 'thermal_map',
            'depth_map': 'depth_map'
        }

    def generate_prompt(self, modality, phase):
        # Map modality if needed
        modality = self.modality_mapping.get(modality, modality)
        
        phase_info = self.phase_descriptions[phase]
        base_desc = f"{self.modality_descriptions[modality]} diabetic foot ulcer (DFU)"
        phase_desc = f"in {phase_info['name']} phase"
        characteristics = f"showing {', '.join(phase_info['characteristics'])}"
        prompt = f"Medical imaging: {base_desc} {phase_desc}, {characteristics}"
        negative_prompt = "blurry, distorted, unrealistic, non-medical, artistic, cartoon"
        return prompt, negative_prompt
from collections import OrderedDict

class GenerativeAugmentationManager:
    def __init__(self, base_dir, config, device=None):
        """
        Initializes the GenerativeAugmentationManager with configuration for generative augmentation.
        
        Args:
            base_dir (Path): Base directory where the generative models are stored.
            config (AugmentationConfig): Configuration object for augmentation.
            device (torch.device): Device to run the models on, defaults to GPU if available.
        """
        self.base_dir = Path(base_dir)
        self.prompt_generator = PromptGenerator()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.model_cache = OrderedDict()  # Use OrderedDict for LRU behavior
        self.tokenizer = None
        self.lock = threading.Lock()  # Ensure thread safety for model loading and cleanup
        self.modality_mapping = {
            'thermal_rgb': 'rgb',
            'depth_rgb': 'rgb',
            'thermal_map': 'thermal_map',
            'depth_map': 'depth_map'
        }

    def cleanup_model_cache(self):
        """
        Ensures the model cache does not exceed the configured maximum number of loaded models.
        Removes the least recently used (LRU) model if the limit is exceeded.
        """
        while len(self.model_cache) > self.config.generative_settings['max_loaded_models']:
            # Pop the least recently used model
            oldest_key, oldest_model = self.model_cache.popitem(last=False)
            print(f"Evicting model: {oldest_key}")
            del oldest_model
            torch.cuda.empty_cache()
            gc.collect()

    def load_model(self, modality, phase):
        """
        Loads a specific generative model based on the modality and phase.
        
        Args:
            modality (str): The input modality (e.g., 'thermal_rgb', 'depth_rgb').
            phase (str): The phase of the model to load (e.g., 'I', 'P', 'R').
            
        Returns:
            StableDiffusionPipeline: The loaded generative model pipeline.
        """
        modality = self.modality_mapping.get(modality, modality)
        model_key = f"{modality}_{phase}"

        with self.lock:  # Ensure thread safety
            if model_key in self.model_cache:
                # Move the accessed model to the end to mark it as recently used
                self.model_cache.move_to_end(model_key)
                return self.model_cache[model_key]

            model_path = self.base_dir / model_key
            if not model_path.exists():
                print(f"No generative model found for {modality}_{phase}")
                return None

            self.cleanup_model_cache()  # Ensure cache limit is not exceeded

            try:
                print(f"Loading generative model for {modality}_{phase}")
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(self.device)

                pipeline.enable_attention_slicing()

                if self.tokenizer is None:
                    self.tokenizer = pipeline.tokenizer

                # Add the model to the cache and mark it as recently used
                self.model_cache[model_key] = pipeline
                return pipeline

            except Exception as e:
                print(f"Error loading model for {modality}_{phase}: {str(e)}")
                traceback.print_exc()
                return None
            finally:
                if 'pipeline' in locals():
                    del pipeline

    def generate_images(self, modality, phase, batch_size=1):
        """
        Generates images using the loaded generative model.
        
        Args:
            modality (str): The input modality (e.g., 'thermal_rgb', 'depth_rgb').
            phase (str): The phase of the model to use for generation.
            batch_size (int): Number of images to generate in one batch.
            
        Returns:
            tf.Tensor: Generated images as a TensorFlow tensor of shape [batch_size, height, width, 3].
        """
        batch_size = min(batch_size, self.config.generative_settings['batch_size_limit'])

        try:
            pipeline = self.load_model(modality, phase)
            if pipeline is None:
                return None

            with torch.no_grad():
                output = pipeline(
                    prompt=self.prompt_generator.generate_prompt(modality, phase)[0],
                    negative_prompt=self.prompt_generator.generate_prompt(modality, phase)[1],
                    num_images_per_prompt=batch_size,
                    num_inference_steps=self.config.generative_settings['inference_steps'],
                    height=self.config.generative_settings['output_size']['height'],
                    width=self.config.generative_settings['output_size']['width']
                ).images

            # Convert PIL images to normalized numpy arrays
            tensors = [np.array(img).astype(np.float32) / 255.0 for img in output]
            return tf.convert_to_tensor(np.stack(tensors), dtype=tf.float32)

        except Exception as e:
            print(f"Error generating images for {modality}_{phase}: {str(e)}")
            traceback.print_exc()
            return None

    def should_generate(self, modality):
        """
        Determines whether to apply generative augmentation for a given modality.
        
        Args:
            modality (str): The input modality (e.g., 'thermal_rgb', 'depth_rgb').
            
        Returns:
            bool: True if generation is enabled and meets the probability threshold.
        """
        return (self.config.modality_settings[modality]['generative_augmentations']['enabled'] and
                random.random() < self.config.generative_settings['prob'])

    def cleanup(self):
        """Release all loaded models and GPU memory."""
        with self.lock:  # Ensure thread-safe cleanup
            self.model_cache.clear()
            torch.cuda.empty_cache()
            gc.collect()
            # TensorFlow-specific cleanup
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for i, _ in enumerate(gpus):
                    tf.config.experimental.reset_memory_stats(f'GPU:{i}')

def create_enhanced_augmentation_fn(gen_manager, config):
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

                    if gen_manager.should_generate(modality):
                        try:
                            def generate_images_wrapper(phase_tensor, modality_str, batch_size_val):
                                try:
                                    phase = phase_tensor.numpy().decode('utf-8')
                                    modality_raw = modality_str.numpy().decode('utf-8')
                                    batch_size = int(batch_size_val.numpy())

                                    # Call updated generate_images
                                    generated = gen_manager.generate_images(
                                        modality_raw, phase, batch_size=batch_size
                                    )
                                    # Properly resize and pad generated images
                                    generated = resize_and_pad_generated_images(
                                        generated,
                                        (value.shape[1], value.shape[2])
                                    )
                                    if generated is None:
                                        return np.zeros([batch_size, value.shape[1], value.shape[2], 3], dtype=np.float32)
                                    
                                    return generated.numpy()

                                except Exception as e:
                                    print(f"Error in generate_images_wrapper: {str(e)}")
                                    return np.zeros([batch_size, value.shape[1], value.shape[2], 3], dtype=np.float32)
                            
                            modality_tensor = tf.constant(modality, dtype=tf.string)
                            batch_size = tf.shape(value)[0]
                            generated = tf.py_function(
                                func=generate_images_wrapper,
                                inp=[current_phase, modality_tensor, batch_size],
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

                    if config.modality_settings[modality]['regular_augmentations']['enabled']:
                        seed = tf.random.uniform([], maxval=1000000, dtype=tf.int32)
                        value = augment_image(value, modality, seed, config)

                output_features[key] = value

            return output_features, label_tensor

        return augment_batch(features, label)

    return apply_augmentation

class GenerativeAugmentationCallback(tf.keras.callbacks.Callback):
    """Callback to manage generative augmentation resources"""
    def __init__(self, gen_manager):
        super().__init__()
        self.gen_manager = gen_manager
    
    def on_epoch_end(self, epoch, logs=None):
        # Clean up GPU memory after each epoch
        self.gen_manager.cleanup()
    
    def on_train_end(self, logs=None):
        # Final cleanup
        self.gen_manager.cleanup()