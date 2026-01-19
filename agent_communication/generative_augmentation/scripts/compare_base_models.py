#!/usr/bin/env python3
"""
Compare Base Models: SDXL 1.0 vs SD 3.5 Medium
Quick training and quality comparison at 128x128 resolution
"""

import sys
import os
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
import yaml

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

import torch
from quality_metrics import QualityMetrics, load_reference_images


def print_section(title):
    """Print section header"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")


def train_model(config_path, model_name):
    """Train a model with given config"""
    print_section(f"Training {model_name}")

    start_time = time.time()

    # Run training script
    cmd = [
        "python",
        str(Path(__file__).parent / "train_lora_model.py"),
        "--config", config_path
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"\n✓ {model_name} training completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {model_name} training failed with exit code {e.returncode}")
        return False


def evaluate_model(checkpoint_dir, config_path, model_name):
    """Evaluate model quality"""
    print_section(f"Evaluating {model_name}")

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    checkpoint_dir = Path(checkpoint_dir)

    # Find latest checkpoint
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        print(f"✗ No checkpoints found in {checkpoint_dir}")
        return None

    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
    print(f"Using checkpoint: {latest_checkpoint.name}")

    # Generate samples
    print("\nGenerating samples...")
    from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline
    from transformers import CLIPTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine pipeline type
    if "stable-diffusion-3" in config['model']['base_model']:
        pipeline_class = StableDiffusion3Pipeline
    else:
        pipeline_class = StableDiffusionPipeline

    # Load base pipeline
    print(f"Loading {config['model']['base_model']}...")
    try:
        pipeline = pipeline_class.from_pretrained(
            config['model']['base_model'],
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)

        # Load LoRA weights
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        if 'unet_lora_state_dict' in checkpoint:
            # Load LoRA weights into UNet
            pipeline.unet.load_state_dict(checkpoint['unet_lora_state_dict'], strict=False)

        # Generate samples
        num_samples = 50
        generated_images = []

        prompt = config['prompts']['positive']
        negative_prompt = config['prompts']['negative']

        print(f"Generating {num_samples} samples...")
        for i in range(num_samples):
            with torch.no_grad():
                image = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=50,
                    guidance_scale=12.0,
                    height=128,
                    width=128,
                    generator=torch.Generator(device).manual_seed(42 + i)
                ).images[0]

                # Convert to tensor
                import torchvision.transforms as T
                image_tensor = T.ToTensor()(image).unsqueeze(0)
                generated_images.append(image_tensor)

            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{num_samples}")

        generated_images = torch.cat(generated_images, dim=0).to(device)

        # Load reference images
        print("\nLoading reference images...")
        reference_images = load_reference_images(
            data_root=config['data']['data_root'],
            modality=config['data']['modality'],
            phase=config['data']['phase'],
            resolution=config['model']['resolution'],
            num_images=50,
            seed=42
        ).to(device)

        # Compute metrics
        print("\nComputing quality metrics...")
        metrics = QualityMetrics(device=device)

        results = metrics.compute_all_metrics(
            generated_images=generated_images,
            real_images=reference_images,
            compute_is=True
        )

        del pipeline
        torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(sdxl_results, sd3_results):
    """Compare and display results"""
    print_section("Quality Comparison Results")

    print(f"{'Metric':<20} {'SDXL 1.0':<20} {'SD 3.5 Medium':<20} {'Winner':<15}")
    print(f"{'-'*20} {'-'*20} {'-'*20} {'-'*15}")

    winner_count = {'sdxl': 0, 'sd3': 0}

    # FID (lower is better)
    sdxl_fid = sdxl_results['fid']
    sd3_fid = sd3_results['fid']
    winner = 'SDXL' if sdxl_fid < sd3_fid else 'SD3.5'
    if sdxl_fid < sd3_fid:
        winner_count['sdxl'] += 1
    else:
        winner_count['sd3'] += 1
    print(f"{'FID (lower better)':<20} {sdxl_fid:<20.2f} {sd3_fid:<20.2f} {winner:<15}")

    # SSIM (higher is better)
    sdxl_ssim = sdxl_results['ssim_mean']
    sd3_ssim = sd3_results['ssim_mean']
    winner = 'SDXL' if sdxl_ssim > sd3_ssim else 'SD3.5'
    if sdxl_ssim > sd3_ssim:
        winner_count['sdxl'] += 1
    else:
        winner_count['sd3'] += 1
    print(f"{'SSIM (higher better)':<20} {sdxl_ssim:<20.4f} {sd3_ssim:<20.4f} {winner:<15}")

    # LPIPS (lower is better)
    sdxl_lpips = sdxl_results['lpips_mean']
    sd3_lpips = sd3_results['lpips_mean']
    winner = 'SDXL' if sdxl_lpips < sd3_lpips else 'SD3.5'
    if sdxl_lpips < sd3_lpips:
        winner_count['sdxl'] += 1
    else:
        winner_count['sd3'] += 1
    print(f"{'LPIPS (lower better)':<20} {sdxl_lpips:<20.4f} {sd3_lpips:<20.4f} {winner:<15}")

    # IS (higher is better)
    sdxl_is = sdxl_results['is_mean']
    sd3_is = sd3_results['is_mean']
    winner = 'SDXL' if sdxl_is > sd3_is else 'SD3.5'
    if sdxl_is > sd3_is:
        winner_count['sdxl'] += 1
    else:
        winner_count['sd3'] += 1
    print(f"{'IS (higher better)':<20} {sdxl_is:<20.4f} {sd3_is:<20.4f} {winner:<15}")

    print(f"\n{'-'*80}")
    print(f"Overall Winner: ", end='')
    if winner_count['sdxl'] > winner_count['sd3']:
        print(f"✓ SDXL 1.0 ({winner_count['sdxl']}/4 metrics)")
    elif winner_count['sd3'] > winner_count['sdxl']:
        print(f"✓ SD 3.5 Medium ({winner_count['sd3']}/4 metrics)")
    else:
        print(f"Tie ({winner_count['sdxl']}-{winner_count['sd3']})")
    print(f"{'-'*80}\n")

    return 'sdxl' if winner_count['sdxl'] > winner_count['sd3'] else 'sd3'


def main():
    """Main comparison workflow"""
    print_section("Base Model Comparison: SDXL 1.0 vs SD 3.5 Medium")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Resolution: 128×128")
    print(f"Training: 3 epochs each")
    print(f"Evaluation: 50 samples, 4 quality metrics")

    # Paths
    base_dir = Path(__file__).parent.parent
    configs_dir = base_dir / "configs"
    checkpoints_dir = base_dir / "checkpoints"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    sdxl_config = configs_dir / "test_sdxl.yaml"
    sd3_config = configs_dir / "test_sd3_medium.yaml"

    # Check configs exist
    if not sdxl_config.exists():
        print(f"✗ SDXL config not found: {sdxl_config}")
        return 1

    if not sd3_config.exists():
        print(f"✗ SD3 config not found: {sd3_config}")
        return 1

    results = {}

    # Train SDXL
    if train_model(sdxl_config, "SDXL 1.0"):
        results['sdxl'] = evaluate_model(
            checkpoints_dir / "test_sdxl",
            sdxl_config,
            "SDXL 1.0"
        )
    else:
        print("✗ SDXL training failed, skipping evaluation")
        results['sdxl'] = None

    # Train SD3.5 Medium
    if train_model(sd3_config, "SD 3.5 Medium"):
        results['sd3'] = evaluate_model(
            checkpoints_dir / "test_sd3_medium",
            sd3_config,
            "SD 3.5 Medium"
        )
    else:
        print("✗ SD3.5 training failed, skipping evaluation")
        results['sd3'] = None

    # Compare results
    if results['sdxl'] is not None and results['sd3'] is not None:
        winner = compare_results(results['sdxl'], results['sd3'])

        # Save results
        comparison_report = {
            'timestamp': datetime.now().isoformat(),
            'resolution': '128x128',
            'training_epochs': 3,
            'sdxl_1.0': results['sdxl'],
            'sd3.5_medium': results['sd3'],
            'winner': winner
        }

        report_path = reports_dir / "model_comparison.json"
        with open(report_path, 'w') as f:
            json.dump(comparison_report, f, indent=2)

        print(f"\n✓ Full comparison saved to: {report_path}")

        # Print recommendation
        print_section("Recommendation")
        if winner == 'sdxl':
            print("✓ Use SDXL 1.0 for production training")
            print("  - Better quality metrics overall")
            print("  - Update phase_I_config.yaml and phase_R_config.yaml to use SDXL")
        else:
            print("✓ Use SD 3.5 Medium for production training")
            print("  - Better quality metrics overall")
            print("  - Update phase_I_config.yaml and phase_R_config.yaml to use SD3.5")

        return 0
    else:
        print("\n✗ Comparison incomplete due to training failures")
        return 1


if __name__ == "__main__":
    sys.exit(main())
