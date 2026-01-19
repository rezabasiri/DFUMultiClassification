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

import argparse
import os
import sys
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
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from peft import LoraConfig, get_peft_model

# Local utilities
from quality_metrics import QualityMetrics
from data_loader import create_dataloaders, load_reference_images
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

    print(f"Loading base model: {model_id}")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float16 if config['hardware']['mixed_precision'] == 'fp16' else torch.float32
    )

    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float16 if config['hardware']['mixed_precision'] == 'fp16' else torch.float32
    )

    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer"
    )

    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=torch.float16 if config['hardware']['mixed_precision'] == 'fp16' else torch.float32
    )

    # Load noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler"
    )

    # Freeze VAE and text encoder (we only train UNet)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

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
        print("Enabled gradient checkpointing")

    return vae, text_encoder, tokenizer, unet, noise_scheduler


def setup_lora(unet, config: dict):
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

    # MSE loss
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
    ema_model=None
) -> float:
    """
    Train for one epoch

    Returns:
        Average training loss
    """
    unet_lora.train()

    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(
        train_loader,
        disable=not accelerator.is_local_main_process,
        desc=f"Epoch {epoch}"
    )

    for batch_idx, batch in enumerate(progress_bar):
        with accelerator.accumulate(unet_lora):
            # Get latents from VAE
            latents = vae.encode(batch['pixel_values'].to(vae.dtype)).latent_dist.sample()
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

            # Get text embeddings
            encoder_hidden_states = text_encoder(batch['input_ids'])[0]

            # Predict noise
            model_pred = unet_lora(
                noisy_latents,
                timesteps,
                encoder_hidden_states
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
                    # Remove noise from latents using model prediction
                    pred_latents = noise_scheduler.step(
                        model_pred,
                        timesteps[0],
                        noisy_latents
                    ).pred_original_sample

                    # Decode to image space
                    pred_images = vae.decode(pred_latents / vae.config.scaling_factor).sample
                    target_images = batch['pixel_values']

                # Compute perceptual loss
                perc_loss = perceptual_loss_fn(pred_images, target_images)
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

        # Update progress bar
        total_loss += loss.detach().item()
        num_batches += 1
        progress_bar.set_postfix({
            'loss': total_loss / num_batches,
            'lr': optimizer.param_groups[0]['lr']
        })

    avg_loss = total_loss / num_batches
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
    ema_model=None
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

    for batch in tqdm(val_loader, disable=not accelerator.is_local_main_process, desc="Validation"):
        # Get latents
        latents = vae.encode(batch['pixel_values'].to(vae.dtype)).latent_dist.sample()
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
        encoder_hidden_states = text_encoder(batch['input_ids'])[0]

        # Predict
        model_pred = model_to_eval(
            noisy_latents,
            timesteps,
            encoder_hidden_states
        ).sample

        # Compute loss
        loss = compute_diffusion_loss(model_pred, noise, timesteps, noise_scheduler)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
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
    ema_model=None
) -> torch.Tensor:
    """
    Generate sample images for visual inspection

    Returns:
        Tensor of generated images [N, C, H, W] in [0, 1] range
    """
    # Use EMA model if requested
    if use_ema and ema_model is not None:
        unet_to_use = ema_model.ema_model
    else:
        unet_to_use = unet_lora

    # Create pipeline for generation
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

    # Generate samples
    prompt = config['prompts']['positive']
    negative_prompt = config['prompts']['negative']

    images = pipeline(
        prompt=[prompt] * num_samples,
        negative_prompt=[negative_prompt] * num_samples,
        num_inference_steps=config['quality']['inference_steps_training'],
        guidance_scale=config['quality']['guidance_scale'],
        height=config['model']['resolution'],
        width=config['model']['resolution']
    ).images

    # Convert PIL images to tensor
    import torchvision.transforms as T
    to_tensor = T.ToTensor()
    image_tensors = torch.stack([to_tensor(img) for img in images])

    # Save samples
    if accelerator.is_main_process:
        save_dir = Path(config['logging']['logging_dir']) / "samples"
        save_dir.mkdir(parents=True, exist_ok=True)

        from torchvision.utils import save_image
        save_image(
            image_tensors,
            save_dir / f"epoch_{epoch:04d}.png",
            nrow=4,
            normalize=True
        )

    return image_tensors


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
        default=None,
        help="Resume from checkpoint ('latest', 'best', or path to checkpoint)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    print("=" * 80)
    print(f"Training Configuration: {args.config}")
    print(f"Phase: {config['data']['phase']}")
    print(f"Modality: {config['data']['modality']}")
    print(f"Resolution: {config['model']['resolution']}")
    print("=" * 80)

    # Setup accelerator
    accelerator = setup_accelerator(config)

    # Set random seed
    if 'seed' in config['reproducibility']:
        torch.manual_seed(config['reproducibility']['seed'])
        np.random.seed(config['reproducibility']['seed'])

    # Load base models
    vae, text_encoder, tokenizer, unet, noise_scheduler = load_base_models(config, accelerator)

    # Setup LoRA
    unet_lora = setup_lora(unet, config)

    # Setup perceptual loss
    perceptual_loss_fn = None
    if config['quality']['perceptual_loss']['enabled']:
        perceptual_loss_fn = PerceptualLoss(
            layer_index=config['quality']['perceptual_loss']['vgg_layer'],
            device=accelerator.device
        )
        print("Enabled perceptual loss")

    # Setup EMA
    ema_model = None
    if config['quality']['ema']['enabled']:
        ema_model = EMAModel(
            unet_lora,
            decay=config['quality']['ema']['decay'],
            device=accelerator.device
        )
        print("Enabled EMA")

    # Create dataloaders
    train_loader, val_loader, train_size, val_size = create_dataloaders(
        data_root=config['data']['data_root'],
        modality=config['data']['modality'],
        phase=config['data']['phase'],
        resolution=config['model']['resolution'],
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size_per_gpu'],
        train_val_split=config['data']['train_val_split'],
        split_seed=config['data']['split_seed'],
        prompt=config['prompts']['positive'] if config['prompts']['enabled'] else None,
        augmentation=config['data']['augmentation'],
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )

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
    lr_scheduler = get_lr_scheduler(
        optimizer,
        scheduler_type=config['training']['lr_scheduler'],
        num_training_steps=num_training_steps,
        num_warmup_steps=config['training']['lr_warmup_steps'],
        num_cycles=config['training'].get('lr_num_cycles', 1)
    )

    # Prepare models with accelerator
    unet_lora, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        unet_lora, optimizer, train_loader, val_loader, lr_scheduler
    )

    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        output_dir=config['checkpointing']['output_dir'],
        keep_last_n=config['checkpointing']['keep_last_n_checkpoints'],
        save_optimizer=config['checkpointing']['save_optimizer_state'],
        monitor_metric='val_fid',
        monitor_mode='min'
    )

    # Load reference images for quality metrics
    reference_images = None
    if config['validation']['compare_to_real']:
        reference_images = load_reference_images(
            data_root=config['data']['data_root'],
            modality=config['data']['modality'],
            phase=config['data']['phase'],
            resolution=config['model']['resolution'],
            num_images=config['validation']['num_real_samples_for_comparison']
        ).to(accelerator.device)

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

    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume:
        checkpoint = checkpoint_manager.load_checkpoint(
            checkpoint_path=args.resume,
            unet_lora=accelerator.unwrap_model(unet_lora),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema_model=ema_model,
            load_optimizer=True
        )
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print("\nStarting training...")
    print(f"Training on {accelerator.num_processes} GPUs")
    print(f"Total epochs: {config['training']['max_epochs']}")
    print(f"Effective batch size: {config['training']['batch_size_per_gpu'] * accelerator.num_processes * config['training']['gradient_accumulation_steps']}")

    for epoch in range(start_epoch, config['training']['max_epochs']):
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
            ema_model=ema_model
        )

        print(f"Train loss: {train_loss:.4f}")

        # Validate
        if (epoch + 1) % config['validation']['validate_every_n_epochs'] == 0:
            val_loss = validate(
                unet_lora=unet_lora,
                vae=vae,
                text_encoder=text_encoder,
                noise_scheduler=noise_scheduler,
                val_loader=val_loader,
                accelerator=accelerator,
                config=config,
                use_ema=config['quality']['ema'].get('use_ema_weights_for_inference', False),
                ema_model=ema_model
            )

            print(f"Val loss: {val_loss:.4f}")

            # Generate validation samples and compute metrics
            val_metrics = None
            if reference_images is not None and (epoch + 1) % config['metrics']['compute_every_n_epochs'] == 0:
                print("Computing quality metrics...")

                # Generate samples
                generated_images = generate_validation_samples(
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
                    ema_model=ema_model
                )

                # Compute metrics
                val_metrics = quality_metrics.compute_all_metrics(
                    generated_images=generated_images,
                    real_images=reference_images,
                    compute_is=config['metrics']['inception_score']['enabled']
                )

                print(quality_metrics.format_metrics(val_metrics))

            # Update metrics tracker
            metrics_tracker.update(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                learning_rate=optimizer.param_groups[0]['lr']
            )

            # Check early stopping
            if early_stopping is not None and val_metrics is not None:
                if early_stopping.update(epoch, val_metrics):
                    print("Early stopping triggered")
                    break

        # Save checkpoint
        if (epoch + 1) % config['checkpointing']['save_every_n_epochs'] == 0:
            if accelerator.is_main_process:
                checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    unet_lora=accelerator.unwrap_model(unet_lora),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    ema_model=ema_model,
                    metrics={'train_loss': train_loss, 'val_loss': val_loss} if 'val_loss' in locals() else None,
                    config=config
                )

    print("\nTraining complete!")

    # Save final checkpoint
    if accelerator.is_main_process:
        checkpoint_manager.save_checkpoint(
            epoch=config['training']['max_epochs'] - 1,
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
