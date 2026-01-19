#!/usr/bin/env python3
"""
Generate sample images from trained model

Usage:
    # Generate 50 Phase I images
    python generate_samples.py \
        --checkpoint checkpoints/phase_I/checkpoint_best.pt \
        --config configs/phase_I_config.yaml \
        --num_samples 50 \
        --output generated_samples/phase_I

    # Generate with custom prompt
    python generate_samples.py \
        --checkpoint checkpoints/phase_R/checkpoint_best.pt \
        --config configs/phase_R_config.yaml \
        --prompt "Medical photograph of healed wound" \
        --num_samples 20
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "utils"))

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def load_pipeline(checkpoint_path: str, config: dict, device: str = "cuda"):
    """Load trained pipeline from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load base model
    model_id = config['model']['base_model']

    # Load components
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Load trained weights (prefer EMA if available)
    if 'ema_state' in checkpoint:
        unet.load_state_dict(checkpoint['ema_state'], strict=False)
        print("Loaded EMA weights")
    elif 'unet_lora_state' in checkpoint:
        unet.load_state_dict(checkpoint['unet_lora_state'], strict=False)
        print("Loaded LoRA weights")

    # Create pipeline
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    )

    pipeline = pipeline.to(device)

    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate sample images")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (overrides config)"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Custom negative prompt (overrides config)"
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=None,
        help="Number of inference steps (overrides config)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale (overrides config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Save as grid image instead of individual images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup output directory
    if args.output is None:
        output_dir = Path("generated_samples") / config['data']['phase']
    else:
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Image Generation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Phase: {config['data']['phase']}")
    print(f"Modality: {config['data']['modality']}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        print(f"Using seed: {args.seed}")

    # Load pipeline
    pipeline = load_pipeline(args.checkpoint, config, device=args.device)

    # Get generation parameters
    prompt = args.prompt if args.prompt is not None else config['prompts']['positive']
    negative_prompt = args.negative_prompt if args.negative_prompt is not None else config['prompts']['negative']
    inference_steps = args.inference_steps if args.inference_steps is not None else config['quality']['inference_steps_production']
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else config['quality']['guidance_scale']
    resolution = config['model']['resolution']

    print("\nGeneration Parameters:")
    print(f"  Prompt: {prompt[:100]}...")
    print(f"  Negative prompt: {negative_prompt[:100]}...")
    print(f"  Inference steps: {inference_steps}")
    print(f"  Guidance scale: {guidance_scale}")
    print(f"  Resolution: {resolution}x{resolution}")
    print()

    # Generate images
    all_images = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

    for i in tqdm(range(num_batches), desc="Generating"):
        current_batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)

        images = pipeline(
            prompt=[prompt] * current_batch_size,
            negative_prompt=[negative_prompt] * current_batch_size,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            height=resolution,
            width=resolution
        ).images

        all_images.extend(images)

    print(f"\nGenerated {len(all_images)} images")

    # Save images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.grid:
        # Save as grid
        import torchvision.transforms as T
        from torchvision.utils import save_image

        to_tensor = T.ToTensor()
        image_tensors = torch.stack([to_tensor(img) for img in all_images])

        grid_path = output_dir / f"grid_{timestamp}.png"
        save_image(
            image_tensors,
            grid_path,
            nrow=int(len(all_images) ** 0.5),
            normalize=False
        )
        print(f"Saved grid to: {grid_path}")

    else:
        # Save individual images
        for idx, image in enumerate(all_images):
            image_path = output_dir / f"sample_{timestamp}_{idx:04d}.png"
            image.save(image_path)

        print(f"Saved {len(all_images)} images to: {output_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
