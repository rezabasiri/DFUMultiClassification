#!/usr/bin/env python3
"""
Compare quality metrics across multiple model checkpoints

Usage:
    # Compare all checkpoints for Phase I
    python compare_models.py \
        --checkpoint_dir checkpoints/phase_I \
        --config configs/phase_I_config.yaml \
        --output reports/phase_I_comparison.json

    # Compare specific checkpoints
    python compare_models.py \
        --checkpoints checkpoint_epoch_0050.pt checkpoint_epoch_0100.pt \
        --config configs/phase_R_config.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import json
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "utils"))

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from quality_metrics import QualityMetrics
from data_loader import load_reference_images


def load_pipeline_from_checkpoint(checkpoint_path: str, config: dict, device: str = "cuda"):
    """Load pipeline from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model_id = config['model']['base_model']

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Load weights (prefer EMA)
    if 'ema_state' in checkpoint:
        unet.load_state_dict(checkpoint['ema_state'], strict=False)
    elif 'unet_lora_state' in checkpoint:
        unet.load_state_dict(checkpoint['unet_lora_state'], strict=False)

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

    return pipeline.to(device), checkpoint.get('epoch', -1)


@torch.no_grad()
def generate_and_evaluate(
    pipeline,
    config: dict,
    reference_images: torch.Tensor,
    quality_metrics: QualityMetrics,
    num_samples: int = 100,
    device: str = "cuda"
):
    """Generate images and compute quality metrics"""
    # Generate images
    all_images = []
    batch_size = 8
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        current_batch_size = min(batch_size, num_samples - i * batch_size)

        images = pipeline(
            prompt=[config['prompts']['positive']] * current_batch_size,
            negative_prompt=[config['prompts']['negative']] * current_batch_size,
            num_inference_steps=config['quality']['inference_steps_production'],
            guidance_scale=config['quality']['guidance_scale'],
            height=config['model']['resolution'],
            width=config['model']['resolution']
        ).images

        # Convert to tensor
        import torchvision.transforms as T
        to_tensor = T.ToTensor()
        image_tensors = torch.stack([to_tensor(img) for img in images])
        all_images.append(image_tensors)

    all_images = torch.cat(all_images, dim=0).to(device)

    # Compute metrics
    metrics = quality_metrics.compute_all_metrics(
        generated_images=all_images,
        real_images=reference_images,
        compute_is=True
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare model checkpoints")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing checkpoints (will evaluate all .pt files)"
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Specific checkpoint files to compare"
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
        default=100,
        help="Number of samples to generate per checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for comparison results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )

    args = parser.parse_args()

    # Get list of checkpoints
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        # Also include best checkpoint if exists
        best_ckpt = checkpoint_dir / "checkpoint_best.pt"
        if best_ckpt.exists():
            checkpoints.append(best_ckpt)
    elif args.checkpoints:
        checkpoints = [Path(ckpt) for ckpt in args.checkpoints]
    else:
        raise ValueError("Must provide either --checkpoint_dir or --checkpoints")

    print("=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(f"Number of checkpoints: {len(checkpoints)}")
    print(f"Samples per checkpoint: {args.num_samples}")
    print("=" * 80)

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load reference images
    print("\nLoading reference images...")
    reference_images = load_reference_images(
        data_root=config['data']['data_root'],
        modality=config['data']['modality'],
        phase=config['data']['phase'],
        resolution=config['model']['resolution']
    ).to(args.device)

    # Initialize quality metrics
    quality_metrics = QualityMetrics(device=args.device)

    # Evaluate each checkpoint
    results = []

    for checkpoint_path in tqdm(checkpoints, desc="Evaluating checkpoints"):
        print(f"\nEvaluating: {checkpoint_path.name}")

        try:
            # Load pipeline
            pipeline, epoch = load_pipeline_from_checkpoint(
                str(checkpoint_path),
                config,
                device=args.device
            )

            # Generate and evaluate
            metrics = generate_and_evaluate(
                pipeline=pipeline,
                config=config,
                reference_images=reference_images,
                quality_metrics=quality_metrics,
                num_samples=args.num_samples,
                device=args.device
            )

            # Store results
            result = {
                'checkpoint': str(checkpoint_path),
                'epoch': epoch,
                'fid': metrics.get('fid', None),
                'ssim_mean': metrics.get('ssim_mean', None),
                'ssim_std': metrics.get('ssim_std', None),
                'lpips_mean': metrics.get('lpips_mean', None),
                'lpips_std': metrics.get('lpips_std', None),
                'is_mean': metrics.get('is_mean', None),
                'is_std': metrics.get('is_std', None)
            }

            results.append(result)

            # Print metrics
            print(f"  FID: {metrics.get('fid', 'N/A'):.2f}")
            print(f"  SSIM: {metrics.get('ssim_mean', 'N/A'):.4f}")
            print(f"  LPIPS: {metrics.get('lpips_mean', 'N/A'):.4f}")

            # Clean up
            del pipeline
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Create comparison table
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    df = pd.DataFrame(results)

    # Sort by FID (lower is better)
    if 'fid' in df.columns:
        df = df.sort_values('fid')

    # Print table
    print("\n" + df.to_string(index=False))

    # Find best model
    if len(results) > 0 and 'fid' in df.columns:
        best_idx = df['fid'].idxmin()
        best_model = df.loc[best_idx]

        print("\n" + "=" * 80)
        print("BEST MODEL")
        print("=" * 80)
        print(f"Checkpoint: {best_model['checkpoint']}")
        print(f"Epoch: {best_model['epoch']}")
        print(f"FID: {best_model['fid']:.2f}")
        print(f"SSIM: {best_model['ssim_mean']:.4f}")
        print(f"LPIPS: {best_model['lpips_mean']:.4f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

        # Save CSV
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")

    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
