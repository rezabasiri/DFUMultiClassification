"""
Quality Metrics for Generative Model Evaluation

Implements:
- FID (Fréchet Inception Distance)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- IS (Inception Score)

All metrics are used to:
1. Evaluate model quality during training
2. Filter low-quality generated images during inference
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
from PIL import Image
import os

# Import metric libraries
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.inception import InceptionScore
    import lpips
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some metrics unavailable: {e}")
    print("Install with: pip install torchmetrics lpips")
    METRICS_AVAILABLE = False


class QualityMetrics:
    """
    Comprehensive quality metrics for generated medical images

    Attributes:
        device: torch.device for computation
        fid_metric: FID metric instance
        ssim_metric: SSIM metric instance
        lpips_metric: LPIPS metric instance
        is_metric: Inception Score metric instance
    """

    def __init__(
        self,
        device: str = "cuda",
        lpips_network: str = "alex",
        fid_feature: int = 2048
    ):
        """
        Initialize quality metrics

        Args:
            device: Device for computation ('cuda' or 'cpu')
            lpips_network: LPIPS network ('alex', 'vgg', or 'squeeze')
            fid_feature: FID feature dimension (2048 for default Inception)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if not METRICS_AVAILABLE:
            raise ImportError(
                "Required metric libraries not available. "
                "Install with: pip install torchmetrics lpips"
            )

        # Initialize FID (Fréchet Inception Distance)
        # Measures distribution distance between real and generated images
        # Lower is better (< 50 is good, < 30 is very good)
        self.fid_metric = FrechetInceptionDistance(
            feature=fid_feature,
            normalize=True,  # Normalize images to [0, 1]
            dist_sync_on_step=False  # Disable distributed sync (only main process computes FID)
        ).to(self.device)

        # Initialize SSIM (Structural Similarity Index)
        # Measures structural similarity between images
        # Higher is better (> 0.7 is good, > 0.8 is very good)
        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=1.0  # Images in [0, 1] range
        ).to(self.device)

        # Initialize LPIPS (Learned Perceptual Image Patch Similarity)
        # Measures perceptual similarity using deep features
        # Lower is better (< 0.3 is good, < 0.2 is very good)
        self.lpips_metric = lpips.LPIPS(net=lpips_network).to(self.device)
        self.lpips_metric.eval()

        # Initialize Inception Score
        # Measures quality and diversity of generated images
        # Higher is better (> 2.0 is good for medical images)
        self.is_metric = InceptionScore(
            normalize=True,
            dist_sync_on_step=False  # Disable distributed sync (only main process computes IS)
        ).to(self.device)

        # Set all metrics to eval mode
        self.fid_metric.eval()
        self.ssim_metric.eval()
        self.is_metric.eval()

    @torch.no_grad()
    def compute_fid(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor
    ) -> float:
        """
        Compute FID between real and generated images

        Args:
            real_images: Real images [N, C, H, W] in [0, 1] range
            generated_images: Generated images [N, C, H, W] in [0, 1] range

        Returns:
            FID score (lower is better)
        """
        # Ensure images are on correct device
        real_images = real_images.to(self.device)
        generated_images = generated_images.to(self.device)

        # Ensure RGB (3 channels)
        if real_images.shape[1] == 1:
            real_images = real_images.repeat(1, 3, 1, 1)
        if generated_images.shape[1] == 1:
            generated_images = generated_images.repeat(1, 3, 1, 1)

        # FID requires uint8 images [0, 255], convert from float [0, 1]
        real_images_uint8 = (real_images * 255).byte()
        generated_images_uint8 = (generated_images * 255).byte()

        # Update metric with real images (as "real" distribution)
        self.fid_metric.update(real_images_uint8, real=True)

        # Update metric with generated images (as "fake" distribution)
        self.fid_metric.update(generated_images_uint8, real=False)

        # Compute FID
        fid_score = self.fid_metric.compute().item()

        # Reset metric for next computation
        self.fid_metric.reset()

        return fid_score

    @torch.no_grad()
    def compute_ssim(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor
    ) -> float:
        """
        Compute SSIM between two images

        Args:
            image1: First image [C, H, W] or [N, C, H, W] in [0, 1] range
            image2: Second image [C, H, W] or [N, C, H, W] in [0, 1] range

        Returns:
            SSIM score (higher is better, range [0, 1])
        """
        # Ensure images are on correct device
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)

        # Add batch dimension if needed
        if image1.dim() == 3:
            image1 = image1.unsqueeze(0)
        if image2.dim() == 3:
            image2 = image2.unsqueeze(0)

        # Ensure RGB (3 channels)
        if image1.shape[1] == 1:
            image1 = image1.repeat(1, 3, 1, 1)
        if image2.shape[1] == 1:
            image2 = image2.repeat(1, 3, 1, 1)

        # Compute SSIM
        ssim_score = self.ssim_metric(image1, image2).item()

        return ssim_score

    @torch.no_grad()
    def compute_ssim_batch(
        self,
        generated_images: torch.Tensor,
        real_images: torch.Tensor
    ) -> Tuple[float, List[float]]:
        """
        Compute SSIM for a batch of generated images against real images
        Each generated image is compared to its most similar real image

        Args:
            generated_images: Generated images [N, C, H, W] in [0, 1] range
            real_images: Real reference images [M, C, H, W] in [0, 1] range

        Returns:
            (mean_ssim, individual_ssim_scores)
        """
        ssim_scores = []

        for gen_img in generated_images:
            # Compare to all real images, keep best SSIM
            best_ssim = -1.0
            for real_img in real_images:
                ssim = self.compute_ssim(gen_img, real_img)
                best_ssim = max(best_ssim, ssim)

            ssim_scores.append(best_ssim)

        mean_ssim = np.mean(ssim_scores)

        return mean_ssim, ssim_scores

    @torch.no_grad()
    def compute_lpips(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor
    ) -> float:
        """
        Compute LPIPS between two images

        Args:
            image1: First image [C, H, W] or [N, C, H, W] in [0, 1] range
            image2: Second image [C, H, W] or [N, C, H, W] in [0, 1] range

        Returns:
            LPIPS score (lower is better, typically [0, 1])
        """
        # Ensure images are on correct device
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)

        # Add batch dimension if needed
        if image1.dim() == 3:
            image1 = image1.unsqueeze(0)
        if image2.dim() == 3:
            image2 = image2.unsqueeze(0)

        # Ensure RGB (3 channels)
        if image1.shape[1] == 1:
            image1 = image1.repeat(1, 3, 1, 1)
        if image2.shape[1] == 1:
            image2 = image2.repeat(1, 3, 1, 1)

        # LPIPS expects images in [-1, 1] range, convert from [0, 1]
        image1 = image1 * 2 - 1
        image2 = image2 * 2 - 1

        # Compute LPIPS
        lpips_score = self.lpips_metric(image1, image2).item()

        return lpips_score

    @torch.no_grad()
    def compute_lpips_batch(
        self,
        generated_images: torch.Tensor,
        real_images: torch.Tensor
    ) -> Tuple[float, List[float]]:
        """
        Compute LPIPS for a batch of generated images against real images
        Each generated image is compared to its most similar real image

        Args:
            generated_images: Generated images [N, C, H, W] in [0, 1] range
            real_images: Real reference images [M, C, H, W] in [0, 1] range

        Returns:
            (mean_lpips, individual_lpips_scores)
        """
        lpips_scores = []

        for gen_img in generated_images:
            # Compare to all real images, keep best (lowest) LPIPS
            best_lpips = float('inf')
            for real_img in real_images:
                lpips_score = self.compute_lpips(gen_img, real_img)
                best_lpips = min(best_lpips, lpips_score)

            lpips_scores.append(best_lpips)

        mean_lpips = np.mean(lpips_scores)

        return mean_lpips, lpips_scores

    @torch.no_grad()
    def compute_inception_score(
        self,
        generated_images: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Compute Inception Score for generated images

        Args:
            generated_images: Generated images [N, C, H, W] in [0, 1] range

        Returns:
            (mean_is, std_is)
        """
        # Ensure images are on correct device
        generated_images = generated_images.to(self.device)

        # Ensure RGB (3 channels)
        if generated_images.shape[1] == 1:
            generated_images = generated_images.repeat(1, 3, 1, 1)

        # IS requires uint8 images [0, 255], convert from float [0, 1]
        generated_images_uint8 = (generated_images * 255).byte()

        # Update and compute IS
        self.is_metric.update(generated_images_uint8)
        is_mean, is_std = self.is_metric.compute()

        # Reset metric
        self.is_metric.reset()

        return is_mean.item(), is_std.item()

    def compute_all_metrics(
        self,
        generated_images: torch.Tensor,
        real_images: torch.Tensor,
        compute_is: bool = True,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute all quality metrics

        Args:
            generated_images: Generated images [N, C, H, W] in [0, 1] range
            real_images: Real reference images [M, C, H, W] in [0, 1] range
            compute_is: Whether to compute Inception Score (expensive)
            verbose: Whether to print progress messages

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        # Compute FID
        if verbose:
            print("Computing FID...")
        metrics['fid'] = self.compute_fid(real_images, generated_images)

        # Compute SSIM (average best match)
        if verbose:
            print("Computing SSIM...")
        mean_ssim, ssim_scores = self.compute_ssim_batch(generated_images, real_images)
        metrics['ssim_mean'] = mean_ssim
        metrics['ssim_std'] = np.std(ssim_scores)

        # Compute LPIPS (average best match)
        if verbose:
            print("Computing LPIPS...")
        mean_lpips, lpips_scores = self.compute_lpips_batch(generated_images, real_images)
        metrics['lpips_mean'] = mean_lpips
        metrics['lpips_std'] = np.std(lpips_scores)

        # Compute Inception Score (optional, expensive)
        if compute_is:
            if verbose:
                print("Computing Inception Score...")
            is_mean, is_std = self.compute_inception_score(generated_images)
            metrics['is_mean'] = is_mean
            metrics['is_std'] = is_std

        return metrics

    def format_metrics(self, metrics: Dict[str, float]) -> str:
        """
        Format metrics as a readable string

        Args:
            metrics: Dictionary of metric scores

        Returns:
            Formatted string
        """
        lines = ["Quality Metrics:"]
        lines.append("=" * 50)

        if 'fid' in metrics:
            lines.append(f"FID: {metrics['fid']:.2f} (lower is better, < 50 is good)")

        if 'ssim_mean' in metrics:
            lines.append(f"SSIM: {metrics['ssim_mean']:.4f} ± {metrics.get('ssim_std', 0):.4f} (higher is better, > 0.7 is good)")

        if 'lpips_mean' in metrics:
            lines.append(f"LPIPS: {metrics['lpips_mean']:.4f} ± {metrics.get('lpips_std', 0):.4f} (lower is better, < 0.3 is good)")

        if 'is_mean' in metrics:
            lines.append(f"IS: {metrics['is_mean']:.2f} ± {metrics.get('is_std', 0):.2f} (higher is better, > 2.0 is good)")

        lines.append("=" * 50)

        return "\n".join(lines)


class QualityFilter:
    """
    Filter generated images based on quality thresholds
    Used during training to reject low-quality synthetic data
    """

    def __init__(
        self,
        quality_metrics: QualityMetrics,
        thresholds: Dict[str, float],
        min_passing_metrics: int = 2,
        reference_images: Optional[torch.Tensor] = None
    ):
        """
        Initialize quality filter

        Args:
            quality_metrics: QualityMetrics instance
            thresholds: Dictionary of metric thresholds
                Example: {'ssim_min': 0.7, 'lpips_max': 0.3}
            min_passing_metrics: Minimum metrics that must pass
            reference_images: Reference real images for comparison
        """
        self.quality_metrics = quality_metrics
        self.thresholds = thresholds
        self.min_passing_metrics = min_passing_metrics
        self.reference_images = reference_images

        # Statistics
        self.total_checked = 0
        self.total_rejected = 0

    def check_image_quality(
        self,
        image: torch.Tensor,
        reference_images: Optional[torch.Tensor] = None
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if an image passes quality thresholds

        Args:
            image: Generated image [C, H, W] in [0, 1] range
            reference_images: Optional override for reference images

        Returns:
            (passes, metric_results)
            passes: True if image passes quality check
            metric_results: Dict of individual metric pass/fail
        """
        if reference_images is None:
            reference_images = self.reference_images

        if reference_images is None:
            raise ValueError("No reference images provided for quality check")

        self.total_checked += 1

        metric_results = {}
        passing_count = 0

        # Check SSIM if threshold provided
        if 'ssim_min' in self.thresholds:
            # Find best SSIM among reference images
            best_ssim = max([
                self.quality_metrics.compute_ssim(image, ref_img)
                for ref_img in reference_images
            ])

            passes = best_ssim >= self.thresholds['ssim_min']
            metric_results['ssim'] = passes
            if passes:
                passing_count += 1

        # Check LPIPS if threshold provided
        if 'lpips_max' in self.thresholds:
            # Find best (lowest) LPIPS among reference images
            best_lpips = min([
                self.quality_metrics.compute_lpips(image, ref_img)
                for ref_img in reference_images
            ])

            passes = best_lpips <= self.thresholds['lpips_max']
            metric_results['lpips'] = passes
            if passes:
                passing_count += 1

        # Overall pass/fail
        passes = passing_count >= self.min_passing_metrics

        if not passes:
            self.total_rejected += 1

        return passes, metric_results

    def get_rejection_rate(self) -> float:
        """Get current rejection rate"""
        if self.total_checked == 0:
            return 0.0
        return self.total_rejected / self.total_checked

    def get_statistics(self) -> Dict[str, int]:
        """Get filtering statistics"""
        return {
            'total_checked': self.total_checked,
            'total_rejected': self.total_rejected,
            'total_accepted': self.total_checked - self.total_rejected,
            'rejection_rate': self.get_rejection_rate()
        }
