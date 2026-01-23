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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))

import torch
from quality_metrics import QualityMetrics
from data_loader import load_reference_images


def print_section(title):
    """Print section header"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")


def save_samples_to_pdf(images_tensor, output_path, model_name, num_cols=10):
    """
    Save generated image samples to PDF with grid layout

    Args:
        images_tensor: Tensor of shape [N, C, H, W] in range [0, 1]
        output_path: Path to save PDF
        model_name: Name of model for title
        num_cols: Number of columns in grid
    """
    print(f"\nSaving {len(images_tensor)} samples to PDF: {output_path}")

    # Convert to numpy and prepare for display
    images_np = images_tensor.cpu().numpy()
    images_np = np.transpose(images_np, (0, 2, 3, 1))  # [N, H, W, C]

    num_images = len(images_np)
    num_rows = (num_images + num_cols - 1) // num_cols

    # Create PDF
    with PdfPages(output_path) as pdf:
        # Create figure with grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 2 * num_rows))
        fig.suptitle(f'{model_name} - Generated Samples (128×128)',
                     fontsize=16, fontweight='bold')

        # Flatten axes for easy iteration
        if num_rows == 1:
            axes = [axes] if num_cols == 1 else axes
        else:
            axes = axes.flatten()

        # Plot each image
        for idx in range(num_rows * num_cols):
            ax = axes[idx]

            if idx < num_images:
                # Show image
                ax.imshow(images_np[idx])
                ax.set_title(f'Sample {idx+1}', fontsize=8)
                ax.axis('off')
            else:
                # Empty subplot
                ax.axis('off')

        plt.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close()

    print(f"✓ PDF saved: {output_path}")


def create_side_by_side_comparison(sdxl_images, sd3_images, output_path, num_pairs=25):
    """
    Create side-by-side comparison PDF of SDXL vs SD3.5 samples

    Args:
        sdxl_images: Tensor [N, C, H, W] for SDXL samples
        sd3_images: Tensor [N, C, H, W] for SD3.5 samples
        output_path: Path to save comparison PDF
        num_pairs: Number of side-by-side pairs to show
    """
    print(f"\nCreating side-by-side comparison PDF: {output_path}")

    # Convert to numpy
    sdxl_np = sdxl_images.numpy()
    sdxl_np = np.transpose(sdxl_np, (0, 2, 3, 1))  # [N, H, W, C]

    sd3_np = sd3_images.numpy()
    sd3_np = np.transpose(sd3_np, (0, 2, 3, 1))  # [N, H, W, C]

    num_pairs = min(num_pairs, len(sdxl_np), len(sd3_np))

    with PdfPages(output_path) as pdf:
        # Create comparison grid (5 pairs per row, 5 rows per page)
        pairs_per_page = 25
        num_pages = (num_pairs + pairs_per_page - 1) // pairs_per_page

        for page in range(num_pages):
            start_idx = page * pairs_per_page
            end_idx = min(start_idx + pairs_per_page, num_pairs)
            pairs_on_page = end_idx - start_idx

            # 5x5 grid, but each pair takes 2 columns
            rows = 5
            fig, axes = plt.subplots(rows, 10, figsize=(20, 10))
            fig.suptitle(f'Side-by-Side Comparison: SDXL 1.0 (left) vs SD 3.5 Medium (right) - Page {page+1}/{num_pages}',
                         fontsize=14, fontweight='bold')

            for i in range(pairs_on_page):
                row = i // 5
                col_offset = (i % 5) * 2

                img_idx = start_idx + i

                # SDXL on left
                ax_left = axes[row, col_offset]
                ax_left.imshow(sdxl_np[img_idx])
                ax_left.set_title(f'SDXL {img_idx+1}', fontsize=8)
                ax_left.axis('off')

                # SD3.5 on right
                ax_right = axes[row, col_offset + 1]
                ax_right.imshow(sd3_np[img_idx])
                ax_right.set_title(f'SD3.5 {img_idx+1}', fontsize=8)
                ax_right.axis('off')

            # Hide unused axes
            for i in range(pairs_on_page, rows * 5):
                row = i // 5
                col_offset = (i % 5) * 2
                axes[row, col_offset].axis('off')
                axes[row, col_offset + 1].axis('off')

            plt.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close()

    print(f"✓ Comparison PDF saved: {output_path}")


def train_model(config_path, model_name):
    """Train a model with given config"""
    print_section(f"Training {model_name}")

    start_time = time.time()

    # Run training script
    cmd = [
        "python",
        str(Path(__file__).parent / "train_lora_model.py"),
        "--config", str(config_path)
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

        # Save samples to PDF for visual inspection
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        pdf_filename = f"{model_name.lower().replace(' ', '_').replace('.', '_')}_samples.pdf"
        pdf_path = reports_dir / pdf_filename
        save_samples_to_pdf(generated_images, pdf_path, model_name, num_cols=10)

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

        # Return both results and generated images for side-by-side comparison
        return results, generated_images.cpu()

    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


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
    images = {}

    # Train SDXL
    if train_model(sdxl_config, "SDXL 1.0"):
        sdxl_result, sdxl_imgs = evaluate_model(
            checkpoints_dir / "test_sdxl",
            sdxl_config,
            "SDXL 1.0"
        )
        results['sdxl'] = sdxl_result
        images['sdxl'] = sdxl_imgs
    else:
        print("✗ SDXL training failed, skipping evaluation")
        results['sdxl'] = None
        images['sdxl'] = None

    # Train SD3.5 Medium
    if train_model(sd3_config, "SD 3.5 Medium"):
        sd3_result, sd3_imgs = evaluate_model(
            checkpoints_dir / "test_sd3_medium",
            sd3_config,
            "SD 3.5 Medium"
        )
        results['sd3'] = sd3_result
        images['sd3'] = sd3_imgs
    else:
        print("✗ SD3.5 training failed, skipping evaluation")
        results['sd3'] = None
        images['sd3'] = None

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

        # Create side-by-side comparison PDF
        if images['sdxl'] is not None and images['sd3'] is not None:
            comparison_pdf = reports_dir / "side_by_side_comparison.pdf"
            create_side_by_side_comparison(
                images['sdxl'],
                images['sd3'],
                comparison_pdf,
                num_pairs=25
            )

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
