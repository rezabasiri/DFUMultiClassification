#!/usr/bin/env python3
"""
Train LoRA-adapted Stable Diffusion model for wound image generation

Features:
- Multi-GPU training with Accelerate
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Perceptual loss for better visual quality
- EMA (Exponential Moving Average) for smoother models
- Quality metrics (FID, SSIM, LPIPS) validation
- Full checkpoint and resume capability
- Configurable via YAML files

Usage:
    # Train Phase I model
    python train_lora_model.py --config configs/phase_I_config.yaml

    # Resume from checkpoint
    python train_lora_model.py --config configs/phase_I_config.yaml --resume latest

    # Train Phase R model
    python train_lora_model.py --config configs/phase_R_config.yaml
"""

# CRITICAL: Set PyTorch CUDA memory allocator to use expandable segments
# This prevents memory fragmentation when moving models to/from CPU during metrics computation
# Without this, restoring EMA model after FID computation causes OOM due to fragmentation
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Enable unbuffered output for live logging
import sys
os.environ['PYTHONUNBUFFERED'] = '1'


class TeeLogger:
    """Tee output to both console and log file with live flushing"""
    def __init__(self, log_file, stream):
        self.log_file = log_file
        self.stream = stream
        self.encoding = getattr(stream, 'encoding', 'utf-8')

    def write(self, message):
        self.stream.write(message)
        self.stream.flush()
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.stream.flush()
        if self.log_file:
            self.log_file.flush()

    def fileno(self):
        return self.stream.fileno()


def setup_logging(config_path: str, logging_dir: str):
    """Setup automatic logging to file based on config name"""
    from pathlib import Path
    from datetime import datetime

    # Create logging directory
    log_dir = Path(logging_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate log filename from config name
    config_name = Path(config_path).stem  # e.g., "full_sdxl" from "full_sdxl.yaml"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{config_name}_{timestamp}.log"
    log_path = log_dir / log_filename

    # Also create a "latest" symlink for easy access
    latest_link = log_dir / f"{config_name}_latest.log"

    # Open log file
    log_file = open(log_path, 'w', buffering=1)  # Line buffered

    # Tee stdout and stderr to both console and file
    sys.stdout = TeeLogger(log_file, sys.__stdout__)
    sys.stderr = TeeLogger(log_file, sys.__stderr__)

    # Update symlink to latest log
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(log_path.name)
    except OSError:
        pass  # Symlink creation may fail on some systems

    return log_path

import argparse
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from datetime import datetime

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

# Diffusers and transformers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from peft import LoraConfig, get_peft_model

# Local utilities
from typing import Tuple, Dict
from quality_metrics import QualityMetrics
from data_loader import (
    create_dataloaders,
    load_reference_images,
    load_reference_images_by_phase,
    save_reference_images_to_disk,
    load_reference_images_from_disk
)
from training_utils import (
    PerceptualLoss,
    EMAModel,
    EarlyStoppingTracker,
    MetricsTracker,
    get_optimizer,
    get_lr_scheduler
)
from checkpoint_utils import CheckpointManager


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_accelerator(config: dict) -> Accelerator:
    """Setup Accelerate for multi-GPU training"""
    # Create project configuration
    project_config = ProjectConfiguration(
        project_dir=config['checkpointing']['output_dir'],
        logging_dir=config['logging']['logging_dir']
    )

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        mixed_precision=config['hardware']['mixed_precision'],
        log_with="tensorboard" if config['logging']['use_tensorboard'] else None,
        project_config=project_config
    )

    return accelerator


def load_base_models(config: dict, accelerator: Accelerator):
    """
    Load base Stable Diffusion models

    Returns:
        (vae, text_encoder, tokenizer, unet, noise_scheduler)
    """
    model_id = config['model']['base_model']
    is_sdxl = 'xl' in model_id.lower()

    if accelerator.is_main_process:
        print(f"Loading base model: {model_id}")
        if is_sdxl:
            print("Detected SDXL model - loading dual text encoders")

    # Determine weight dtype based on mixed precision setting
    mixed_precision = config['hardware']['mixed_precision']
    if mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=weight_dtype
    )

    # Load text encoder(s)
    if is_sdxl:
        # SDXL uses two text encoders
        text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=weight_dtype
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_id,
            subfolder="text_encoder_2",
            torch_dtype=weight_dtype
        )
    else:
        text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=weight_dtype
        )
        text_encoder_2 = None

    # Load tokenizer(s)
    if is_sdxl:
        tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer"
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer_2"
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer"
        )
        tokenizer_2 = None

    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=weight_dtype
    )

    # Load noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler"
    )

    # Freeze VAE and text encoder(s) (we only train UNet)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if text_encoder_2 is not None:
        text_encoder_2.requires_grad_(False)

    # Enable xFormers if requested (memory-efficient attention)
    if config['hardware']['use_xformers']:
        try:
            vae.enable_xformers_memory_efficient_attention()
            unet.enable_xformers_memory_efficient_attention()
            print("Enabled xFormers memory-efficient attention")
        except Exception as e:
            print(f"Warning: Could not enable xFormers: {e}")

    # Enable gradient checkpointing if requested (saves memory)
    if config['hardware']['gradient_checkpointing']:
        unet.enable_gradient_checkpointing()
        if accelerator.is_main_process:
            print("Enabled gradient checkpointing")

    return vae, text_encoder, tokenizer, unet, noise_scheduler, text_encoder_2, tokenizer_2


def setup_lora(unet, config: dict, accelerator: Accelerator):
    """
    Add LoRA adapters to UNet

    Args:
        unet: Base UNet model
        config: Configuration dict

    Returns:
        UNet with LoRA adapters
    """
    lora_config = LoraConfig(
        r=config['lora']['rank'],
        lora_alpha=config['lora']['alpha'],
        init_lora_weights="gaussian",
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias=config['lora']['bias']
    )

    # Add LoRA to UNet
    unet_lora = get_peft_model(unet, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in unet_lora.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet_lora.parameters())

    if accelerator.is_main_process:
        print(f"LoRA Configuration:")
        print(f"  Rank: {config['lora']['rank']}")
        print(f"  Alpha: {config['lora']['alpha']}")
        print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return unet_lora


def compute_diffusion_loss(
    model_pred: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler
) -> torch.Tensor:
    """
    Compute diffusion (denoising) loss

    Args:
        model_pred: Model prediction
        noise: Target noise
        timesteps: Timesteps
        noise_scheduler: Noise scheduler

    Returns:
        Loss tensor
    """
    # Get target based on scheduler's prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(model_pred, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

    # MSE loss - ensure target is on same device as model_pred
    target = target.to(model_pred.device)
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    return loss


def train_one_epoch(
    epoch: int,
    unet_lora,
    vae,
    text_encoder,
    noise_scheduler,
    train_loader: DataLoader,
    optimizer,
    lr_scheduler,
    accelerator: Accelerator,
    config: dict,
    perceptual_loss_fn=None,
    ema_model=None,
    text_encoder_2=None,
    is_sdxl=False,
    tokenizer=None,
    tokenizer_2=None
) -> float:
    """
    Train for one epoch

    Returns:
        Average training loss
    """
    unet_lora.train()

    total_loss = 0.0
    num_batches = 0

    num_batches_total = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        with accelerator.accumulate(unet_lora):
            # Get latents from VAE
            pixel_values = batch['pixel_values'].to(device=vae.device, dtype=vae.dtype)
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)

            # Sample random timesteps
            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=latents.device
            ).long()

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings with optional prompt dropout (CFG training)
            prompt_dropout = config['training'].get('prompt_dropout', 0.0)
            input_ids = batch['input_ids'].to(text_encoder.device)

            # Apply prompt dropout: randomly replace some prompts with unconditional (empty) embeddings
            if prompt_dropout > 0 and tokenizer is not None:
                dropout_mask = torch.rand(batch_size) < prompt_dropout
                if dropout_mask.any():
                    # Create unconditional (empty string) input_ids
                    uncond_input = tokenizer(
                        "",
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids.to(input_ids.device)
                    # Replace dropped samples with unconditional input
                    for i in range(batch_size):
                        if dropout_mask[i]:
                            input_ids[i] = uncond_input[0]

            encoder_hidden_states = text_encoder(input_ids)[0]

            # SDXL-specific: Concatenate embeddings and compute pooled embeddings/time_ids
            added_cond_kwargs = None
            if is_sdxl and text_encoder_2 is not None:

                # Get hidden states from second text encoder (apply same dropout mask)
                input_ids_2 = batch['input_ids'].to(text_encoder_2.device)

                # Apply prompt dropout to second encoder as well
                if prompt_dropout > 0 and tokenizer_2 is not None and dropout_mask.any():
                    uncond_input_2 = tokenizer_2(
                        "",
                        padding="max_length",
                        max_length=tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids.to(input_ids_2.device)
                    for i in range(batch_size):
                        if dropout_mask[i]:
                            input_ids_2[i] = uncond_input_2[0]

                encoder_output_2 = text_encoder_2(input_ids_2, output_hidden_states=True)
                encoder_hidden_states_2 = encoder_output_2.hidden_states[-2]  # Penultimate layer

                # Concatenate embeddings from both encoders (CLIP-L 768 + OpenCLIP 1280 = 2048)
                encoder_hidden_states = torch.cat([
                    encoder_hidden_states.to(accelerator.device),
                    encoder_hidden_states_2.to(accelerator.device)
                ], dim=-1)

                # Get pooled embeddings from second text encoder
                pooled_embeds = encoder_output_2[0]

                # Create time_ids (original_size, crops_coords_top_left, target_size)
                # Format: [original_h, original_w, crops_top, crops_left, target_h, target_w]
                # MUST be float dtype - using long causes NaN in SDXL UNet
                resolution = config['model']['resolution']
                time_ids = torch.tensor([
                    [resolution, resolution, 0, 0, resolution, resolution]
                ], dtype=torch.float32).repeat(batch_size, 1).to(accelerator.device)

                added_cond_kwargs = {
                    "text_embeds": pooled_embeds.to(accelerator.device),
                    "time_ids": time_ids
                }

            # Predict noise (ensure all inputs on same device)
            model_pred = unet_lora(
                noisy_latents.to(accelerator.device),
                timesteps.to(accelerator.device),
                encoder_hidden_states.to(accelerator.device),
                added_cond_kwargs=added_cond_kwargs
            ).sample

            # Compute diffusion loss
            diff_loss = compute_diffusion_loss(
                model_pred,
                noise,
                timesteps,
                noise_scheduler
            )

            loss = diff_loss

            # Add perceptual loss if enabled
            if perceptual_loss_fn is not None and config['quality']['perceptual_loss']['enabled']:
                # Decode predictions to image space for perceptual loss
                with torch.no_grad():
                    # Manually compute denoised latents (don't use scheduler.step() which expects inference mode)
                    # Based on scheduler's prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        # model_pred is noise, we need to denoise the noisy_latents
                        # x0 = (x_t - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)
                        # For simplicity, use the formula: x0_pred = noisy_latents - model_pred
                        # This is an approximation but works for perceptual loss
                        alpha_t = noise_scheduler.alphas_cumprod[timesteps].to(accelerator.device)
                        alpha_t = alpha_t.view(-1, 1, 1, 1)
                        sigma_t = (1 - alpha_t).sqrt()
                        pred_latents = (noisy_latents.to(accelerator.device) - sigma_t * model_pred.to(accelerator.device)) / alpha_t.sqrt()
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        # For v-prediction, convert to x0
                        alpha_t = noise_scheduler.alphas_cumprod[timesteps].to(accelerator.device)
                        alpha_t = alpha_t.view(-1, 1, 1, 1)
                        sigma_t = (1 - alpha_t).sqrt()
                        pred_latents = alpha_t.sqrt() * noisy_latents.to(accelerator.device) - sigma_t * model_pred.to(accelerator.device)
                    else:
                        # Sample prediction - model directly predicts x0
                        pred_latents = model_pred.to(accelerator.device)

                    # Decode to image space
                    pred_images = vae.decode(
                        (pred_latents / vae.config.scaling_factor).to(device=vae.device, dtype=vae.dtype)
                    ).sample
                    target_images = batch['pixel_values'].to(device=vae.device, dtype=vae.dtype)

                # Compute perceptual loss (move to accelerator device)
                perc_loss = perceptual_loss_fn(
                    pred_images.to(accelerator.device),
                    target_images.to(accelerator.device)
                )
                loss = loss + config['quality']['perceptual_loss']['weight'] * perc_loss

            # Backward pass
            accelerator.backward(loss)

            # Clip gradients
            if accelerator.sync_gradients:
                max_grad_norm = config['training']['max_grad_norm']
                accelerator.clip_grad_norm_(unet_lora.parameters(), max_grad_norm)

            # Optimizer step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Update EMA
            if ema_model is not None and accelerator.sync_gradients:
                ema_model.update(unet_lora)

        # Update progress
        total_loss += loss.detach().item()
        num_batches += 1

        # Log progress every N steps (also log step 1 for immediate feedback)
        log_every = config['logging'].get('log_every_n_steps', 50)
        if (num_batches == 1 or num_batches % log_every == 0) and accelerator.is_main_process:
            avg_loss_so_far = total_loss / num_batches
            print(f"  Step {num_batches}/{num_batches_total} - Loss: {avg_loss_so_far:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Print final epoch results
    avg_loss = total_loss / num_batches
    if accelerator.is_main_process:
        print(f"  Epoch {epoch+1} complete - Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
    return avg_loss


@torch.no_grad()
def validate(
    unet_lora,
    vae,
    text_encoder,
    noise_scheduler,
    val_loader: DataLoader,
    accelerator: Accelerator,
    config: dict,
    use_ema: bool = False,
    ema_model=None,
    text_encoder_2=None,
    is_sdxl=False
) -> float:
    """
    Run validation

    Returns:
        Average validation loss
    """
    # Use EMA model if requested
    if use_ema and ema_model is not None:
        model_to_eval = ema_model.ema_model
    else:
        model_to_eval = unet_lora

    model_to_eval.eval()

    total_loss = 0.0
    num_batches = 0

    num_val_batches = len(val_loader)
    for batch_idx, batch in enumerate(val_loader):
        # Get latents
        pixel_values = batch['pixel_values'].to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Sample noise
        noise = torch.randn_like(latents)

        # Sample timesteps
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=latents.device
        ).long()

        # Add noise
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get text embeddings
        input_ids = batch['input_ids'].to(text_encoder.device)
        encoder_hidden_states = text_encoder(input_ids)[0]

        # SDXL-specific: Concatenate embeddings and compute pooled embeddings/time_ids
        added_cond_kwargs = None
        if is_sdxl and text_encoder_2 is not None:
            # Get hidden states from second text encoder
            input_ids_2 = batch['input_ids'].to(text_encoder_2.device)
            encoder_output_2 = text_encoder_2(input_ids_2, output_hidden_states=True)
            encoder_hidden_states_2 = encoder_output_2.hidden_states[-2]  # Penultimate layer

            # Concatenate embeddings from both encoders (CLIP-L 768 + OpenCLIP 1280 = 2048)
            encoder_hidden_states = torch.cat([
                encoder_hidden_states.to(accelerator.device),
                encoder_hidden_states_2.to(accelerator.device)
            ], dim=-1)

            # Get pooled embeddings from second text encoder
            pooled_embeds = encoder_output_2[0]

            # Create time_ids (MUST be float dtype - using long causes NaN)
            resolution = config['model']['resolution']
            time_ids = torch.tensor([
                [resolution, resolution, 0, 0, resolution, resolution]
            ], dtype=torch.float32).repeat(batch_size, 1).to(accelerator.device)

            added_cond_kwargs = {
                "text_embeds": pooled_embeds.to(accelerator.device),
                "time_ids": time_ids
            }

        # Predict (ensure all inputs on same device)
        model_pred = model_to_eval(
            noisy_latents.to(accelerator.device),
            timesteps.to(accelerator.device),
            encoder_hidden_states.to(accelerator.device),
            added_cond_kwargs=added_cond_kwargs
        ).sample

        # Compute loss
        loss = compute_diffusion_loss(model_pred, noise, timesteps, noise_scheduler)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    if accelerator.is_main_process:
        print(f"  Validation: {num_val_batches} batches - Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def generate_validation_samples(
    epoch: int,
    unet_lora,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    accelerator: Accelerator,
    config: dict,
    num_samples: int = 8,
    use_ema: bool = False,
    ema_model=None,
    text_encoder_2=None,
    tokenizer_2=None,
    is_sdxl=False
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Generate sample images for visual inspection using phase-specific prompts

    Returns:
        Tuple of:
        - Tensor of all generated images [N, C, H, W] in [0, 1] range
        - Dictionary mapping phase name to tensor of images for that phase
    """
    # Use EMA model if requested
    if use_ema and ema_model is not None:
        unet_to_use = ema_model.ema_model
    else:
        unet_to_use = unet_lora

    # Create pipeline for generation (SDXL vs SD 1.5/2.x)
    if is_sdxl:
        from diffusers import StableDiffusionXLPipeline
        pipeline = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=accelerator.unwrap_model(unet_to_use),
            scheduler=noise_scheduler,
        )
    else:
        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet_to_use),
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    negative_prompt = config['prompts']['negative']

    # Use configurable generation batch size (default 4 for 512x512 on 24GB GPU)
    gen_batch_size = config.get('metrics', {}).get('generation_batch_size', 4)

    # Check if we have phase-specific prompts
    phase_prompts = config['prompts'].get('phase_prompts', None)

    all_images = []
    phase_images = {}  # Dictionary to store images by phase

    if phase_prompts is not None and len(phase_prompts) > 0:
        # Generate images using phase-specific prompts (balanced across phases)
        phases = sorted(phase_prompts.keys())  # e.g., ['I', 'P', 'R']
        samples_per_phase = num_samples // len(phases)
        remainder = num_samples % len(phases)

        print(f"Generating {num_samples} images across {len(phases)} phases...")

        for i, phase in enumerate(phases):
            # Distribute remainder across first phases
            n_samples_this_phase = samples_per_phase + (1 if i < remainder else 0)
            if n_samples_this_phase == 0:
                continue

            prompt = phase_prompts[phase]
            phase_image_list = []

            print(f"  Phase {phase}: generating {n_samples_this_phase} images...")

            for j in range(0, n_samples_this_phase, gen_batch_size):
                batch_size = min(gen_batch_size, n_samples_this_phase - j)
                batch_images = pipeline(
                    prompt=[prompt] * batch_size,
                    negative_prompt=[negative_prompt] * batch_size,
                    num_inference_steps=config['quality']['inference_steps_training'],
                    guidance_scale=config['quality']['guidance_scale'],
                    height=config['model']['resolution'],
                    width=config['model']['resolution']
                ).images
                phase_image_list.extend(batch_images)

                # Clear cache between batches
                torch.cuda.empty_cache()

            all_images.extend(phase_image_list)

            # Convert phase images to tensor and store
            import torchvision.transforms as T
            to_tensor = T.ToTensor()
            phase_images[phase] = torch.stack([to_tensor(img) for img in phase_image_list])
    else:
        # Fallback: use single prompt (backward compatibility)
        prompt = config['prompts']['positive']

        for i in range(0, num_samples, gen_batch_size):
            batch_size = min(gen_batch_size, num_samples - i)
            batch_images = pipeline(
                prompt=[prompt] * batch_size,
                negative_prompt=[negative_prompt] * batch_size,
                num_inference_steps=config['quality']['inference_steps_training'],
                guidance_scale=config['quality']['guidance_scale'],
                height=config['model']['resolution'],
                width=config['model']['resolution']
            ).images
            all_images.extend(batch_images)

            # Clear cache between batches
            torch.cuda.empty_cache()

    # Convert all images to tensor
    import torchvision.transforms as T
    to_tensor = T.ToTensor()
    image_tensors = torch.stack([to_tensor(img) for img in all_images])

    # Save samples (organized by phase if available)
    if accelerator.is_main_process:
        save_dir = Path(config['logging']['logging_dir']) / "samples"
        save_dir.mkdir(parents=True, exist_ok=True)

        from torchvision.utils import save_image

        # Save all images in a grid (use epoch + 1 to match displayed epoch number)
        save_image(
            image_tensors,
            save_dir / f"epoch_{epoch + 1:04d}.png",
            nrow=4,
            normalize=True
        )

        # Also save per-phase samples if available
        if len(phase_images) > 0:
            for phase, phase_tensor in phase_images.items():
                save_image(
                    phase_tensor,
                    save_dir / f"epoch_{epoch + 1:04d}_phase_{phase}.png",
                    nrow=4,
                    normalize=True
                )

    # CRITICAL: Delete pipeline and move all model components back to CPU
    # The pipeline holds references to all models on GPU
    # Explicitly move each component to CPU before deleting
    pipeline.unet.to('cpu')
    pipeline.vae.to('cpu')
    pipeline.text_encoder.to('cpu')
    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
        pipeline.text_encoder_2.to('cpu')

    # CRITICAL: Reset the noise scheduler timesteps for training
    # The pipeline modifies scheduler.timesteps during inference, which breaks training
    # Set timesteps to the full training steps to restore proper state
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps, device=accelerator.device)

    del pipeline

    # Aggressive memory cleanup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return image_tensors, phase_images


def main():
    parser = argparse.ArgumentParser(description="Train LoRA Stable Diffusion for wound generation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="Resume from checkpoint ('auto' to auto-detect, 'latest', 'best', path to checkpoint, or 'none' to start fresh)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup accelerator first (needed for is_main_process checks)
    accelerator = setup_accelerator(config)

    # Setup automatic logging (only on main process to avoid duplicate logs)
    log_path = None
    if accelerator.is_main_process:
        log_path = setup_logging(args.config, config['logging']['logging_dir'])
        print(f"Logging to: {log_path}")

    # Only print from main process to avoid duplicates
    if accelerator.is_main_process:
        print("=" * 80)
        print(f"Training Configuration: {args.config}")
        print(f"Phase: {config['data']['phase']}")
        print(f"Modality: {config['data']['modality']}")
        print(f"Resolution: {config['model']['resolution']}")
        print("=" * 80)

    # Set random seed
    if 'seed' in config['reproducibility']:
        torch.manual_seed(config['reproducibility']['seed'])
        np.random.seed(config['reproducibility']['seed'])

    # Load base models
    vae, text_encoder, tokenizer, unet, noise_scheduler, text_encoder_2, tokenizer_2 = load_base_models(config, accelerator)
    is_sdxl = text_encoder_2 is not None

    # Move frozen models to GPU
    vae = vae.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device)
    if text_encoder_2 is not None:
        text_encoder_2 = text_encoder_2.to(accelerator.device)

    # Setup training method: LoRA or Full Fine-tuning
    training_method = config['model'].get('training_method', 'lora').lower()

    if training_method == 'full':
        # Full fine-tuning: train entire UNet (100% parameters)
        unet_lora = unet
        unet_lora.requires_grad_(True)  # Enable gradients for all parameters

        if accelerator.is_main_process:
            trainable_params = sum(p.numel() for p in unet_lora.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in unet_lora.parameters())
            print(f"Full Fine-tuning:")
            print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    else:
        # LoRA training (parameter-efficient fine-tuning)
        unet_lora = setup_lora(unet, config, accelerator)

    # Setup perceptual loss
    perceptual_loss_fn = None
    if config['quality']['perceptual_loss']['enabled']:
        perceptual_loss_fn = PerceptualLoss(
            layer_index=config['quality']['perceptual_loss']['vgg_layer'],
            device=accelerator.device
        )
        if accelerator.is_main_process:
            print("Enabled perceptual loss")

    # Setup EMA - keep on CPU to save GPU memory (~2.5GB for SDXL)
    # EMA will be moved to GPU only during sample generation
    ema_model = None
    if config['quality']['ema']['enabled']:
        ema_model = EMAModel(
            unet_lora,
            decay=config['quality']['ema']['decay'],
            device='cpu'  # Keep EMA on CPU to save GPU memory
        )
        if accelerator.is_main_process:
            print("Enabled EMA (on CPU to save GPU memory)")

    # Create dataloaders
    # Use phase-specific prompts if training on all phases
    phase_prompts = None
    single_prompt = None

    if config['prompts']['enabled']:
        if config['data']['phase'].lower() == 'all' and 'phase_prompts' in config['prompts']:
            # Use phase-specific prompts for multi-phase training
            phase_prompts = config['prompts']['phase_prompts']
            if accelerator.is_main_process:
                print("Using phase-specific prompts for conditioning")
        else:
            # Single prompt for all images
            single_prompt = config['prompts']['positive']

    train_loader, val_loader, train_size, val_size = create_dataloaders(
        data_root=config['data']['data_root'],
        modality=config['data']['modality'],
        phase=config['data']['phase'],
        resolution=config['model']['resolution'],
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size_per_gpu'],
        train_val_split=config['data']['train_val_split'],
        split_seed=config['data']['split_seed'],
        prompt=single_prompt,
        phase_prompts=phase_prompts,
        augmentation=config['data']['augmentation'],
        bbox_file=config['data'].get('bbox_file', None),
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        max_samples=config['data'].get('max_samples', None),
        data_percentage=config['data'].get('data_percentage', 100.0),
        class_weights=config['data'].get('class_weights', None)
    )

    # TESTING: Limit dataset size if max_train_samples is specified
    if 'max_train_samples' in config['training'] and config['training']['max_train_samples'] is not None:
        max_samples = config['training']['max_train_samples']
        if train_size > max_samples:
            from torch.utils.data import Subset
            # Limit train dataset to first max_samples
            train_dataset = train_loader.dataset
            train_dataset = Subset(train_dataset, range(min(max_samples, len(train_dataset))))
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config['training']['batch_size_per_gpu'],
                shuffle=True,
                num_workers=config['hardware']['num_workers'],
                pin_memory=config['hardware']['pin_memory']
            )
            train_size = len(train_dataset)
            print(f"Limited train set to {train_size} samples for fast testing")

    # Setup optimizer
    optimizer = get_optimizer(
        unet_lora.parameters(),
        optimizer_type=config['training']['optimizer'],
        learning_rate=config['training']['learning_rate'],
        beta1=config['training']['adam_beta1'],
        beta2=config['training']['adam_beta2'],
        weight_decay=config['training']['adam_weight_decay'],
        epsilon=config['training']['adam_epsilon']
    )

    # Setup learning rate scheduler
    num_training_steps = len(train_loader) * config['training']['max_epochs']

    # Support both lr_warmup_ratio (percentage) and lr_warmup_steps (fixed value)
    if 'lr_warmup_ratio' in config['training']:
        num_warmup_steps = int(num_training_steps * config['training']['lr_warmup_ratio'])
        if accelerator.is_main_process:
            print(f"Warmup: {config['training']['lr_warmup_ratio']*100:.1f}% of {num_training_steps} steps = {num_warmup_steps} steps")
    else:
        num_warmup_steps = config['training'].get('lr_warmup_steps', 0)

    lr_scheduler = get_lr_scheduler(
        optimizer,
        scheduler_type=config['training']['lr_scheduler'],
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        num_cycles=config['training'].get('lr_num_cycles', 1)
    )

    # Prepare models with accelerator
    unet_lora, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        unet_lora, optimizer, train_loader, val_loader, lr_scheduler
    )

    # Setup checkpoint manager
    # Use val_fid for best checkpoint tracking (proper metric for generative models)
    # CPU offloading during FID computation prevents OOM
    checkpoint_manager = CheckpointManager(
        output_dir=config['checkpointing']['output_dir'],
        keep_last_n=config['checkpointing']['keep_last_n_checkpoints'],
        save_optimizer=config['checkpointing']['save_optimizer_state'],
        monitor_metric='val_fid',
        monitor_mode='min'
    )

    # Reference images are loaded on-demand during metrics computation (not here)
    # This saves GPU memory - reference_images are loaded, used, then freed each time
    reference_images = None

    # Setup quality metrics
    quality_metrics = QualityMetrics(device=accelerator.device)

    # Setup trackers
    early_stopping = EarlyStoppingTracker(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta'],
        monitor=config['training']['early_stopping']['monitor'],
        mode=config['training']['early_stopping']['mode']
    ) if config['training']['early_stopping']['enabled'] else None

    metrics_tracker = MetricsTracker()

    # Resume from checkpoint (auto-detect by default)
    start_epoch = 0
    resume_path = args.resume

    # Handle auto-detection of checkpoint
    if resume_path == "auto":
        # Check if latest checkpoint exists
        latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint is not None:
            resume_path = "latest"
            if accelerator.is_main_process:
                print(f"Auto-detected existing checkpoint, resuming from latest...")
        else:
            resume_path = None  # No checkpoint to resume from
            if accelerator.is_main_process:
                print("No existing checkpoint found, starting fresh training...")
    elif resume_path == "none":
        resume_path = None  # Explicitly start fresh
        # Clean up old checkpoints when explicitly starting fresh
        if accelerator.is_main_process:
            checkpoint_dir = Path(config['checkpointing']['output_dir'])
            if checkpoint_dir.exists():
                old_checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
                if old_checkpoints:
                    for ckpt in old_checkpoints:
                        ckpt.unlink()
                    print(f"Cleaned up {len(old_checkpoints)} checkpoints from {checkpoint_dir}")
                # Also clean history file
                history_file = checkpoint_dir / "checkpoint_history.json"
                if history_file.exists():
                    history_file.unlink()

    if resume_path is not None:
        checkpoint = checkpoint_manager.load_checkpoint(
            checkpoint_path=resume_path,
            unet_lora=accelerator.unwrap_model(unet_lora),
            lr_scheduler=lr_scheduler,
            ema_model=ema_model
        )
        # Checkpoint stores 1-indexed epoch, convert to 0-indexed for loop
        # Resume continues from saved epoch (e.g., checkpoint_epoch_0001.pt -> start at epoch index 1)
        start_epoch = checkpoint.get('epoch', 1)
        if accelerator.is_main_process:
            print(f"Resumed from checkpoint epoch {start_epoch}, continuing training...")

    # Training loop
    if accelerator.is_main_process:
        print("\nStarting training...")
        print(f"Training on {accelerator.num_processes} GPUs")
        print(f"Total epochs: {config['training']['max_epochs']}")
        print(f"Effective batch size: {config['training']['batch_size_per_gpu'] * accelerator.num_processes * config['training']['gradient_accumulation_steps']}")

    for epoch in range(start_epoch, config['training']['max_epochs']):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{config['training']['max_epochs']}")

        # Train
        train_loss = train_one_epoch(
            epoch=epoch,
            unet_lora=unet_lora,
            vae=vae,
            text_encoder=text_encoder,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            config=config,
            perceptual_loss_fn=perceptual_loss_fn,
            ema_model=ema_model,
            text_encoder_2=text_encoder_2,
            is_sdxl=is_sdxl,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2
        )

        if accelerator.is_main_process:
            print(f"Train loss: {train_loss:.4f}")

        # Validate every epoch (needed for early stopping to work properly)
        # NOTE: Validation uses the current model weights (not EMA) because EMA is
        # kept on CPU to save GPU memory. EMA is only moved to GPU for sample generation.
        val_loss = validate(
            unet_lora=unet_lora,
            vae=vae,
            text_encoder=text_encoder,
            noise_scheduler=noise_scheduler,
            val_loader=val_loader,
            accelerator=accelerator,
            config=config,
            use_ema=False,  # Don't use EMA for validation (EMA is on CPU)
            ema_model=None,
            text_encoder_2=text_encoder_2,
            is_sdxl=is_sdxl
        )

        if accelerator.is_main_process:
            print(f"Val loss: {val_loss:.4f}")

        # Generate validation samples and compute metrics (expensive, run less frequently)
        # NOTE: ALL processes must participate in FID computation due to distributed sync in torchmetrics
        # Also run at first epoch to verify the entire pipeline works before long training
        val_metrics = None
        should_compute_metrics = config['validation']['compare_to_real'] and (
            epoch == 0 or (epoch + 1) % config['metrics']['compute_every_n_epochs'] == 0
        )
        if should_compute_metrics:
            # Check if we're using phase-specific generation
            phase_prompts = config['prompts'].get('phase_prompts', None)
            use_per_phase_metrics = phase_prompts is not None and config['data']['phase'].lower() == 'all'

            if use_per_phase_metrics:
                # Load reference images BY PHASE for per-phase comparison
                num_samples = config['validation']['num_real_samples_for_comparison']
                num_phases = len(phase_prompts)
                samples_per_phase = num_samples // num_phases

                # Check if we should save/load reference images from disk
                save_reference = config['validation'].get('save_reference_images', False)
                reference_save_dir = Path(config['logging']['logging_dir']) / "reference_images"

                if epoch == 0:
                    # First epoch: ALWAYS create fresh reference images (resolution/settings may have changed)
                    if accelerator.is_main_process:
                        print("Creating fresh reference images from source data with bbox-aware cropping...")

                    reference_images_by_phase = load_reference_images_by_phase(
                        data_root=config['data']['data_root'],
                        modality=config['data']['modality'],
                        resolution=config['model']['resolution'],
                        num_images_per_phase=config['validation'].get('num_reference_images_per_phase', samples_per_phase),
                        seed=config['validation'].get('reference_images_seed', 42),
                        bbox_file=config['data'].get('bbox_file', None),
                        bbox_crop_prob=config['data']['augmentation'].get('bbox_crop_prob', 0.5),
                        bbox_margin_range=config['data']['augmentation'].get('bbox_margin_range', [0.05, 0.15]),
                        verbose=accelerator.is_main_process
                    )

                    # Save to disk for subsequent epochs (only main process)
                    if save_reference and accelerator.is_main_process:
                        save_reference_images_to_disk(
                            reference_images_by_phase,
                            save_dir=str(reference_save_dir),
                            verbose=True
                        )
                elif save_reference:
                    # Subsequent epochs with save_reference enabled: load from disk cache
                    reference_images_by_phase = load_reference_images_from_disk(
                        save_dir=str(reference_save_dir),
                        phases=list(phase_prompts.keys()),
                        verbose=accelerator.is_main_process
                    )
                    if reference_images_by_phase is None:
                        # Fallback if disk cache missing
                        reference_images_by_phase = load_reference_images_by_phase(
                            data_root=config['data']['data_root'],
                            modality=config['data']['modality'],
                            resolution=config['model']['resolution'],
                            num_images_per_phase=samples_per_phase,
                            bbox_file=config['data'].get('bbox_file', None),
                            bbox_crop_prob=config['data']['augmentation'].get('bbox_crop_prob', 0.5),
                            bbox_margin_range=config['data']['augmentation'].get('bbox_margin_range', [0.05, 0.15]),
                            verbose=accelerator.is_main_process
                        )
                else:
                    # save_reference disabled: load fresh each time
                    reference_images_by_phase = load_reference_images_by_phase(
                        data_root=config['data']['data_root'],
                        modality=config['data']['modality'],
                        resolution=config['model']['resolution'],
                        num_images_per_phase=samples_per_phase,
                        bbox_file=config['data'].get('bbox_file', None),
                        bbox_crop_prob=config['data']['augmentation'].get('bbox_crop_prob', 0.5),
                        bbox_margin_range=config['data']['augmentation'].get('bbox_margin_range', [0.05, 0.15]),
                        verbose=accelerator.is_main_process
                    )

                # Move to device
                for phase in reference_images_by_phase:
                    reference_images_by_phase[phase] = reference_images_by_phase[phase].to(accelerator.device)

                # Also create combined reference for overall metrics
                reference_images = torch.cat(list(reference_images_by_phase.values()), dim=0)
            else:
                # Load reference images (balanced across phases if phase='all')
                reference_images = load_reference_images(
                    data_root=config['data']['data_root'],
                    modality=config['data']['modality'],
                    phase=config['data']['phase'],
                    resolution=config['model']['resolution'],
                    num_images=config['validation']['num_real_samples_for_comparison'],
                    verbose=accelerator.is_main_process
                ).to(accelerator.device)
                reference_images_by_phase = None

            # Move quality metrics models to GPU (all processes need them)
            quality_metrics.fid_metric.to(accelerator.device)
            quality_metrics.is_metric.to(accelerator.device)
            quality_metrics.lpips_metric.to(accelerator.device)
            quality_metrics.ssim_metric.to(accelerator.device)
            torch.cuda.empty_cache()

            generated_images = None
            generated_images_by_phase = None

            # ALL PROCESSES: Offload training models to CPU to free GPU memory
            # This is critical because ALL processes need GPU memory for FID computation
            import gc

            # Get the original device
            original_device = accelerator.device

            # Move large models to CPU (all processes)
            # CRITICAL: Offload ALL models including EMA to free GPU memory for metrics
            unwrapped_unet = accelerator.unwrap_model(unet_lora)
            unwrapped_unet.to('cpu')
            torch.cuda.empty_cache()  # Free memory immediately after each offload

            vae.to('cpu')
            torch.cuda.empty_cache()

            text_encoder.to('cpu')
            torch.cuda.empty_cache()

            if text_encoder_2 is not None:
                text_encoder_2.to('cpu')
                torch.cuda.empty_cache()

            # NOTE: EMA model is NOT offloaded here because it will be kept on CPU
            # during training and only moved to GPU during sample generation.
            # This saves GPU memory for the training models.

            # Clear GPU cache and run garbage collection
            gc.collect()
            torch.cuda.empty_cache()

            # Only main process generates samples
            if accelerator.is_main_process:
                print("Computing quality metrics...")

                # Generate samples (will reload models to GPU inside the function)
                # Returns tuple: (all_images_tensor, dict of phase->images_tensor)
                generated_images, generated_images_by_phase = generate_validation_samples(
                    epoch=epoch,
                    unet_lora=unet_lora,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    noise_scheduler=noise_scheduler,
                    accelerator=accelerator,
                    config=config,
                    num_samples=config['metrics']['num_samples_for_metrics'],
                    use_ema=config['quality']['ema'].get('use_ema_weights_for_inference', False),
                    ema_model=ema_model,
                    text_encoder_2=text_encoder_2,
                    tokenizer_2=tokenizer_2,
                    is_sdxl=is_sdxl
                )

            # CRITICAL: Broadcast generated images to all processes
            # This is required because torchmetrics FID.compute() synchronizes across all processes
            if accelerator.num_processes > 1:
                # Create placeholder on non-main processes
                if not accelerator.is_main_process:
                    generated_images = torch.zeros(
                        (config['metrics']['num_samples_for_metrics'], 3, config['model']['resolution'], config['model']['resolution']),
                        device=accelerator.device
                    )
                else:
                    # Move generated images to GPU for NCCL broadcast (NCCL doesn't support CPU)
                    generated_images = generated_images.to(accelerator.device)

                # Broadcast from rank 0 to all other ranks
                torch.distributed.broadcast(generated_images, src=0)

            # ALL processes compute overall FID (required for distributed sync)
            val_metrics = quality_metrics.compute_all_metrics(
                generated_images=generated_images,
                real_images=reference_images,
                compute_is=config['metrics']['inception_score']['enabled'],
                verbose=accelerator.is_main_process
            )

            # Compute per-phase metrics on main process only (no distributed sync needed for SSIM/LPIPS)
            if use_per_phase_metrics and accelerator.is_main_process and generated_images_by_phase is not None:
                print("\nPer-phase metrics:")
                print("=" * 50)
                for phase in sorted(generated_images_by_phase.keys()):
                    if phase in reference_images_by_phase:
                        gen_phase = generated_images_by_phase[phase].to(accelerator.device)
                        ref_phase = reference_images_by_phase[phase]

                        # Compute SSIM and LPIPS for this phase
                        mean_ssim, _ = quality_metrics.compute_ssim_batch(gen_phase, ref_phase)
                        mean_lpips, _ = quality_metrics.compute_lpips_batch(gen_phase, ref_phase)

                        print(f"Phase {phase}: SSIM={mean_ssim:.4f}, LPIPS={mean_lpips:.4f}")

                        # Store in val_metrics
                        val_metrics[f'ssim_{phase}'] = mean_ssim
                        val_metrics[f'lpips_{phase}'] = mean_lpips

                        del gen_phase
                        torch.cuda.empty_cache()
                print("=" * 50)

            # Clean up memory before restoring models
            del generated_images  # Free generated images (no longer needed)
            if generated_images_by_phase is not None:
                del generated_images_by_phase

            # Free reference_images on ALL processes to reclaim GPU memory
            if reference_images is not None:
                del reference_images
                reference_images = None
            if use_per_phase_metrics and reference_images_by_phase is not None:
                del reference_images_by_phase
                reference_images_by_phase = None

            # Move quality metrics models to CPU to free GPU memory (ALL processes)
            quality_metrics.fid_metric.to('cpu')
            quality_metrics.is_metric.to('cpu')
            quality_metrics.lpips_metric.to('cpu')
            quality_metrics.ssim_metric.to('cpu')

            # Aggressive memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # All processes sync BEFORE restoring models
            accelerator.wait_for_everyone()

            # Only main process prints metrics
            if accelerator.is_main_process:
                print(quality_metrics.format_metrics(val_metrics))
                val_fid = val_metrics.get('fid', 0)
                print(f"Val FID: {val_fid:.2f}")

            # ALL PROCESSES: Restore training models to GPU
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            # Move models back to GPU one at a time
            unwrapped_unet.to(original_device)
            torch.cuda.empty_cache()

            vae.to(original_device)
            torch.cuda.empty_cache()

            text_encoder.to(original_device)
            torch.cuda.empty_cache()

            if text_encoder_2 is not None:
                text_encoder_2.to(original_device)
                torch.cuda.empty_cache()

            # NOTE: EMA stays on CPU - it's only loaded to GPU during sample generation

            # DON'T move quality metrics back to GPU yet - keep them on CPU to save memory
            # They'll be moved to GPU again next time we need them

        # Update metrics tracker (every epoch since we validate every epoch)
        metrics_tracker.update(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
            learning_rate=optimizer.param_groups[0]['lr']
        )

        # Check early stopping - include val_loss in metrics dict (every epoch)
        should_stop = False
        if early_stopping is not None:
            early_stop_metrics = {'val_loss': val_loss}
            if val_metrics is not None:
                early_stop_metrics.update(val_metrics)
            if early_stopping.update(epoch, early_stop_metrics):
                should_stop = True
                if accelerator.is_main_process:
                    print("Early stopping triggered")

        # CRITICAL: Broadcast early stop decision to all processes
        # This prevents rank 0 from breaking while rank 1 continues
        if accelerator.num_processes > 1:
            import torch.distributed as dist
            # Create tensor for broadcasting (1 = stop, 0 = continue)
            stop_tensor = torch.tensor([1 if should_stop else 0], device=accelerator.device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1

        # All processes break together
        if should_stop:
            break

        # Save checkpoint (also save at first epoch to enable early resume)
        if epoch == 0 or (epoch + 1) % config['checkpointing']['save_every_n_epochs'] == 0:
            if accelerator.is_main_process:
                # Prepare metrics for checkpoint
                checkpoint_metrics = {'train_loss': train_loss}
                if 'val_loss' in locals():
                    checkpoint_metrics['val_loss'] = val_loss
                if val_metrics is not None:
                    checkpoint_metrics.update(val_metrics)

                checkpoint_manager.save_checkpoint(
                    epoch=epoch + 1,  # Use 1-indexed epoch numbers for checkpoint filenames
                    unet_lora=accelerator.unwrap_model(unet_lora),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    ema_model=ema_model,
                    metrics=checkpoint_metrics,
                    config=config
                )

    if accelerator.is_main_process:
        print("\nTraining complete!")

    # Save final checkpoint
    if accelerator.is_main_process:
        checkpoint_manager.save_checkpoint(
            epoch=epoch + 1,  # Use actual final epoch (1-indexed)
            unet_lora=accelerator.unwrap_model(unet_lora),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema_model=ema_model,
            config=config,
            is_best=True
        )

        # Save metrics history
        metrics_path = Path(config['logging']['logging_dir']) / "metrics_history.json"
        metrics_tracker.save(str(metrics_path))
        print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
