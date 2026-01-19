#!/usr/bin/env python3
"""
Evaluate quality of trained generative models

Computes comprehensive quality metrics:
- FID (Fréchet Inception Distance)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- IS (Inception Score)

Usage:
    # Evaluate Phase I model
    python evaluate_quality.py \
        --checkpoint checkpoints/phase_I/checkpoint_best.pt \
        --config configs/phase_I_config.yaml \
        --num_samples 200 \
        --output reports/phase_I_quality.json

    # Evaluate specific checkpoint
    python evaluate_quality.py \
        --checkpoint checkpoints/phase_R/checkpoint_epoch_0100.pt \
        --config configs/phase_R_config.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import json
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "utils"))

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel

from quality_metrics import QualityMetrics
from data_loader import load_reference_images


def load_model_from_checkpoint(checkpoint_path: str, config: dict, device: str = "cuda"):
    """
    Load trained model from checkpoint

    Returns:
        Stable Diffusion pipeline with trained LoRA weights
    """
    print(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load base model
    model_id = config['model']['base_model']

    print(f"Loading base model: {model_id}")

    # Load components
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Load LoRA weights
    if 'unet_lora_state' in checkpoint:
        unet.load_state_dict(checkpoint['unet_lora_state'], strict=False)
        print("Loaded LoRA weights")
    elif 'ema_state' in checkpoint:
        unet.load_state_dict(checkpoint['ema_state'], strict=False)
        print("Loaded EMA weights")
    else:
        raise ValueError("No LoRA or EMA weights found in checkpoint")

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
    pipeline.set_progress_bar_config(disable=False)

    return pipeline


@torch.no_grad()
def generate_images(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str,
    num_images: int,
    resolution: int,
    inference_steps: int,
    guidance_scale: float,
    batch_size: int = 8,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Generate images using the pipeline

    Returns:
        Tensor of generated images [N, C, H, W] in [0, 1] range
    """
    all_images = []

    num_batches = (num_images + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Generating images"):
        current_batch_size = min(batch_size, num_images - i * batch_size)

        images = pipeline(
            prompt=[prompt] * current_batch_size,
            negative_prompt=[negative_prompt] * current_batch_size,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            height=resolution,
            width=resolution
        ).images

        # Convert PIL to tensor
        import torchvision.transforms as T
        to_tensor = T.ToTensor()
        image_tensors = torch.stack([to_tensor(img) for img in images])
        all_images.append(image_tensors)

    # Concatenate all batches
    all_images = torch.cat(all_images, dim=0)

    return all_images


def main():
    parser = argparse.ArgumentParser(description="Evaluate generative model quality")
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
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of images to generate for evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save generated images to disk"
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("Quality Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Phase: {config['data']['phase']}")
    print(f"Modality: {config['data']['modality']}")
    print(f"Number of samples: {args.num_samples}")
    print("=" * 80)

    # Load model
    pipeline = load_model_from_checkpoint(args.checkpoint, config, device=args.device)

    # Load reference images
    print("\nLoading reference images...")
    reference_images = load_reference_images(
        data_root=config['data']['data_root'],
        modality=config['data']['modality'],
        phase=config['data']['phase'],
        resolution=config['model']['resolution'],
        num_images=None  # Load all available
    ).to(args.device)

    print(f"Loaded {len(reference_images)} reference images")

    # Generate images
    print(f"\nGenerating {args.num_samples} images...")
    generated_images = generate_images(
        pipeline=pipeline,
        prompt=config['prompts']['positive'],
        negative_prompt=config['prompts']['negative'],
        num_images=args.num_samples,
        resolution=config['model']['resolution'],
        inference_steps=config['quality']['inference_steps_production'],
        guidance_scale=config['quality']['guidance_scale'],
        batch_size=8,
        device=args.device
    )

    print(f"Generated {len(generated_images)} images")

    # Save generated images if requested
    if args.save_images:
        save_dir = Path(args.checkpoint).parent / "generated_samples"
        save_dir.mkdir(parents=True, exist_ok=True)

        from torchvision.utils import save_image
        save_image(
            generated_images,
            save_dir / "generated_evaluation.png",
            nrow=10,
            normalize=True
        )
        print(f"Saved generated images to: {save_dir}")

    # Compute quality metrics
    print("\nComputing quality metrics...")
    quality_metrics = QualityMetrics(device=args.device)

    metrics = quality_metrics.compute_all_metrics(
        generated_images=generated_images,
        real_images=reference_images,
        compute_is=True
    )

    # Print results
    print("\n" + "=" * 80)
    print(quality_metrics.format_metrics(metrics))
    print("=" * 80)

    # Check against thresholds
    print("\nQuality Assessment:")
    print("-" * 80)

    assessments = []

    if 'fid' in metrics:
        fid = metrics['fid']
        if fid < 30:
            status = "✓ EXCELLENT"
        elif fid < 50:
            status = "✓ GOOD"
        elif fid < 100:
            status = "⚠ FAIR"
        else:
            status = "✗ POOR"
        assessments.append(f"FID: {status} (score: {fid:.2f}, target: < 50)")

    if 'ssim_mean' in metrics:
        ssim = metrics['ssim_mean']
        if ssim > 0.80:
            status = "✓ EXCELLENT"
        elif ssim > 0.70:
            status = "✓ GOOD"
        elif ssim > 0.60:
            status = "⚠ FAIR"
        else:
            status = "✗ POOR"
        assessments.append(f"SSIM: {status} (score: {ssim:.4f}, target: > 0.70)")

    if 'lpips_mean' in metrics:
        lpips = metrics['lpips_mean']
        if lpips < 0.20:
            status = "✓ EXCELLENT"
        elif lpips < 0.30:
            status = "✓ GOOD"
        elif lpips < 0.40:
            status = "⚠ FAIR"
        else:
            status = "✗ POOR"
        assessments.append(f"LPIPS: {status} (score: {lpips:.4f}, target: < 0.30)")

    if 'is_mean' in metrics:
        is_score = metrics['is_mean']
        if is_score > 3.0:
            status = "✓ EXCELLENT"
        elif is_score > 2.0:
            status = "✓ GOOD"
        elif is_score > 1.5:
            status = "⚠ FAIR"
        else:
            status = "✗ POOR"
        assessments.append(f"IS: {status} (score: {is_score:.2f}, target: > 2.0)")

    for assessment in assessments:
        print(assessment)

    print("-" * 80)

    # Determine overall quality
    fid_good = metrics.get('fid', float('inf')) < 50
    ssim_good = metrics.get('ssim_mean', 0.0) > 0.70
    lpips_good = metrics.get('lpips_mean', float('inf')) < 0.30

    passing_metrics = sum([fid_good, ssim_good, lpips_good])

    if passing_metrics >= 2:
        overall = "✓ PASS - Model meets quality requirements"
    else:
        overall = "✗ FAIL - Model does not meet quality requirements"

    print(f"\nOverall: {overall}")
    print("=" * 80)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'checkpoint': str(args.checkpoint),
            'config': str(args.config),
            'phase': config['data']['phase'],
            'modality': config['data']['modality'],
            'num_samples': args.num_samples,
            'num_reference_images': len(reference_images),
            'metrics': metrics,
            'assessments': {
                'fid_good': fid_good,
                'ssim_good': ssim_good,
                'lpips_good': lpips_good,
                'overall_pass': passing_metrics >= 2
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
