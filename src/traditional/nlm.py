"""
Non-Local Means (NLM) denoising implementation

Based on:
    Buades, A., Coll, B., & Morel, J. M. (2005).
    "A non-local algorithm for image denoising"
    IEEE Computer Vision and Pattern Recognition (CVPR).

NLM is an edge-preserving denoising algorithm that replaces each pixel
with a weighted average of pixels with similar neighborhoods, where
similarity is measured across the entire image (non-local).
"""

from src.base_model import TraditionalMethod
import torch
import numpy as np
import cv2


class NonLocalMeans(TraditionalMethod):
    """
    Non-Local Means denoising filter

    This method searches for similar patches across the entire image and
    averages them to denoise, preserving fine details and textures better
    than local filters.

    Args:
        h: Filter strength. Higher h value removes more noise but also
           removes details. (10 is recommended for noise std ~25/255)
        template_window_size: Size of the template patch (should be odd).
                            Recommended: 7
        search_window_size: Size of the area to search for similar patches
                          (should be odd). Recommended: 21
    """

    def __init__(
        self,
        h: float = 10.0,
        template_window_size: int = 7,
        search_window_size: int = 21,
        **kwargs,
    ):
        """Initialize Non-Local Means filter"""
        super().__init__(
            name="Non-Local Means",
            description="Non-Local Means denoising filter (Buades et al., 2005)",
        )

        # Validate parameters
        assert h > 0, f"h must be positive, got {h}"
        assert template_window_size > 0 and template_window_size % 2 == 1, (
            f"template_window_size must be positive odd number, got {template_window_size}"
        )
        assert search_window_size > 0 and search_window_size % 2 == 1, (
            f"search_window_size must be positive odd number, got {search_window_size}"
        )

        self.h = h
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Non-Local Means denoising to a batch of images

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

            # Apply OpenCV's fast NLM for color images
            # Note: OpenCV uses BGR, but since we denoise all channels equally, it doesn't matter
            denoised = cv2.fastNlMeansDenoisingColored(
                img_np,
                None,
                h=self.h,
                hColor=self.h,  # Same h for color components
                templateWindowSize=self.template_window_size,
                searchWindowSize=self.search_window_size,
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
    # Test Non-Local Means
    print("Testing Non-Local Means denoising...")

    # Create test noisy image
    torch.manual_seed(42)
    clean_img = torch.rand(2, 3, 32, 32)  # Batch of 2 images
    noise = torch.randn_like(clean_img) * 0.1
    noisy_img = torch.clamp(clean_img + noise, 0, 1)

    # Test with default parameters
    nlm = NonLocalMeans(h=10, template_window_size=7, search_window_size=21)
    print(f"\nModel: {nlm.name}")
    print(f"Type: {nlm.model_type}")
    print(
        f"Parameters: h={nlm.h}, template={nlm.template_window_size}, search={nlm.search_window_size}"
    )

    denoised = nlm(noisy_img)
    print(f"Input shape: {noisy_img.shape}")
    print(f"Output shape: {denoised.shape}")
    print(f"Input range: [{noisy_img.min():.3f}, {noisy_img.max():.3f}]")
    print(f"Output range: [{denoised.min():.3f}, {denoised.max():.3f}]")

    # Calculate simple MSE
    mse = torch.mean((denoised - noisy_img) ** 2).item()
    print(f"MSE difference: {mse:.6f}")

    print("\n✓ Non-Local Means implementation complete!")
