"""
Bilateral Filter denoising implementation

Based on:
    Tomasi, C., & Manduchi, R. (1998).
    "Bilateral filtering for gray and color images"
    IEEE International Conference on Computer Vision (ICCV).

The bilateral filter is an edge-preserving and noise-reducing smoothing filter.
It replaces each pixel's intensity with a weighted average of intensity values
from nearby pixels. The weights depend on both spatial distance and intensity
difference, preserving sharp edges.
"""

from src.base_model import TraditionalMethod
import torch
import numpy as np
import cv2


class BilateralFilter(TraditionalMethod):
    """
    Bilateral Filter for edge-preserving denoising

    The bilateral filter smooths images while preserving edges by combining
    spatial proximity and intensity similarity. Pixels are weighted by both
    their spatial distance and color distance from the center pixel.

    Args:
        d: Diameter of the pixel neighborhood (if <= 0, computed from sigma_space).
           Typical values: 5-9
        sigma_color: Filter sigma in the color space. Larger value means farther
                    colors will be mixed together. Typical values: 20-150
        sigma_space: Filter sigma in the coordinate space. Larger value means
                    farther pixels influence each other. Typical values: 20-150
    """

    def __init__(
        self, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0, **kwargs
    ):
        """Initialize Bilateral Filter"""
        super().__init__(
            name="Bilateral Filter",
            description="Bilateral filtering for edge-preserving denoising (Tomasi & Manduchi, 1998)",
        )

        # Validate parameters
        assert sigma_color > 0, f"sigma_color must be positive, got {sigma_color}"
        assert sigma_space > 0, f"sigma_space must be positive, got {sigma_space}"

        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Bilateral filtering to a batch of images

        Args:
            x: Tensor of shape [B, C, H, W] with noisy images in range [0, 1]

        Returns:
            Tensor of shape [B, C, H, W] with denoised images in range [0, 1]
        """
        assert x.dim() == 4, f"Expected 4D tensor [B,C,H,W], got {x.dim()}D"
        assert x.shape[1] == 3, f"Expected 3 channels, got {x.shape[1]}"
        assert 0 <= x.min() and x.max() <= 1.0, (
            f"Expected values in [0,1], got [{x.min()}, {x.max()}]"
        )

        denoised_batch = []

        for img in x:
            # Convert from tensor [C, H, W] to numpy [H, W, C] in uint8 [0, 255]
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # Apply OpenCV's bilateral filter
            denoised = cv2.bilateralFilter(
                img_np,
                d=self.d,
                sigmaColor=self.sigma_color,
                sigmaSpace=self.sigma_space,
            )

            # Convert back to tensor [C, H, W] in float [0, 1]
            denoised = torch.from_numpy(denoised / 255.0).permute(2, 0, 1).float()
            denoised_batch.append(denoised)

        # Stack batch and move to same device as input
        result = torch.stack(denoised_batch).to(x.device)

        # Ensure output is in valid range
        result = torch.clamp(result, 0.0, 1.0)

        return result


if __name__ == "__main__":
    # Test Bilateral Filter
    print("Testing Bilateral Filter denoising...")

    # Create test noisy image
    torch.manual_seed(42)
    clean_img = torch.rand(2, 3, 32, 32)  # Batch of 2 images
    noise = torch.randn_like(clean_img) * 0.1
    noisy_img = torch.clamp(clean_img + noise, 0, 1)

    # Test with default parameters
    bilateral = BilateralFilter(d=9, sigma_color=75, sigma_space=75)
    print(f"\nModel: {bilateral.name}")
    print(f"Type: {bilateral.model_type}")
    print(
        f"Parameters: d={bilateral.d}, sigma_color={bilateral.sigma_color}, sigma_space={bilateral.sigma_space}"
    )

    denoised = bilateral(noisy_img)
    print(f"Input shape: {noisy_img.shape}")
    print(f"Output shape: {denoised.shape}")
    print(f"Input range: [{noisy_img.min():.3f}, {noisy_img.max():.3f}]")
    print(f"Output range: [{denoised.min():.3f}, {denoised.max():.3f}]")

    # Calculate simple MSE
    mse = torch.mean((denoised - noisy_img) ** 2).item()
    print(f"MSE difference: {mse:.6f}")

    print("\n✓ Bilateral Filter implementation complete!")
