"""
Wavelet-based denoising implementation

Based on:
    Donoho, D. L. (1995).
    "De-noising by soft-thresholding"
    IEEE Transactions on Information Theory, 41(3), 613-627.

Wavelet denoising works by:
1. Decomposing the image using wavelet transform
2. Thresholding the wavelet coefficients (soft or hard)
3. Reconstructing the image from thresholded coefficients

This method is effective for removing Gaussian noise while preserving edges.
"""

from src.base_model import TraditionalMethod
import torch
import numpy as np


class WaveletDenoising(TraditionalMethod):
    """
    Wavelet-based denoising using soft or hard thresholding

    Decomposes images into wavelet coefficients, applies thresholding to
    remove noise, and reconstructs the denoised image. Preserves edges
    and fine details better than simple smoothing.

    Args:
        wavelet: Wavelet type. Common choices:
                'db1' (Daubechies 1 / Haar)
                'db2', 'db4', 'db8' (Daubechies wavelets)
                'sym2', 'sym4', 'sym8' (Symlets)
                'coif1', 'coif2' (Coiflets)
        mode: Thresholding mode:
              'soft' - Soft thresholding (recommended, smoother results)
              'hard' - Hard thresholding (sharper but noisier)
        level: Decomposition level (None = automatic based on image size)
               Typical values: 1-5. Higher = more aggressive denoising
        sigma: Noise standard deviation estimate (None = automatic)
               Used to calculate threshold. Typical: 0.01-0.1 for normalized images
    """

    def __init__(
        self,
        wavelet: str = "db1",
        mode: str = "soft",
        level: int = None,
        sigma: float = None,
        **kwargs,
    ):
        """Initialize Wavelet Denoising"""
        super().__init__(
            name="Wavelet Denoising",
            description="Wavelet-based denoising using soft thresholding (Donoho, 1995)",
        )

        # Import pywt here to fail fast if not installed
        import pywt

        # Validate parameters
        assert mode in ["soft", "hard"], f"mode must be 'soft' or 'hard', got {mode}"

        # Check if wavelet exists
        available_wavelets = pywt.wavelist()
        assert wavelet in available_wavelets, (
            f"Wavelet '{wavelet}' not found. Available: {available_wavelets[:10]}..."
        )

        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        self.sigma = sigma

    def _estimate_sigma(self, coeffs):
        """
        Estimate noise standard deviation using Median Absolute Deviation (MAD)

        Args:
            coeffs: Wavelet coefficients from first level detail

        Returns:
            Estimated noise standard deviation
        """
        # Use MAD estimator: sigma = MAD / 0.6745
        # MAD is robust to outliers
        mad = np.median(np.abs(coeffs))
        sigma = mad / 0.6745
        return sigma

    def _denoise_single_channel(self, channel: np.ndarray) -> np.ndarray:
        """
        Denoise a single channel using wavelet transform

        Args:
            channel: 2D numpy array [H, W] in range [0, 1]

        Returns:
            Denoised 2D numpy array [H, W]
        """
        import pywt

        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(channel, wavelet=self.wavelet, level=self.level)

        # Estimate noise sigma if not provided
        sigma = self.sigma
        if sigma is None:
            # Use detail coefficients from first level for estimation
            detail_coeffs = coeffs[1]  # (cH, cV, cD)
            sigma = self._estimate_sigma(detail_coeffs[0])
            # Ensure sigma is positive to avoid division issues
            sigma = max(sigma, 1e-10)

        # Calculate threshold using universal threshold
        # threshold = sigma * sqrt(2 * log(N))
        n_pixels = channel.shape[0] * channel.shape[1]
        threshold = sigma * np.sqrt(2 * np.log(n_pixels))

        # Ensure threshold is reasonable
        threshold = min(threshold, 1.0)  # Don't threshold too aggressively

        # Apply thresholding to all detail coefficients
        coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients unchanged

        for detail_level in coeffs[1:]:
            if self.mode == "soft":
                # Soft thresholding: shrink coefficients
                thresh_detail = tuple(
                    pywt.threshold(c, threshold, mode="soft", substitute=0)
                    for c in detail_level
                )
            else:  # hard
                # Hard thresholding: zero out small coefficients
                thresh_detail = tuple(
                    pywt.threshold(c, threshold, mode="hard") for c in detail_level
                )
            coeffs_thresh.append(thresh_detail)

        # Reconstruct denoised image
        denoised = pywt.waverec2(coeffs_thresh, wavelet=self.wavelet)

        # Crop to original size (wavelet transform may change size slightly)
        denoised = denoised[: channel.shape[0], : channel.shape[1]]

        # Ensure no NaN or Inf values
        denoised = np.nan_to_num(denoised, nan=0.0, posinf=1.0, neginf=0.0)

        return denoised

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply wavelet denoising to a batch of images

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
            # Convert from tensor [C, H, W] to numpy [H, W, C]
            img_np = img.permute(1, 2, 0).cpu().numpy()

            # Process each channel independently
            denoised_channels = []
            for c in range(3):
                denoised_c = self._denoise_single_channel(img_np[:, :, c])
                denoised_channels.append(denoised_c)

            # Stack channels back
            denoised_np = np.stack(denoised_channels, axis=2)

            # Clip to valid range
            denoised_np = np.clip(denoised_np, 0.0, 1.0)

            # Convert back to tensor [C, H, W]
            denoised = torch.from_numpy(denoised_np).permute(2, 0, 1).float()
            denoised_batch.append(denoised)

        # Stack batch and move to same device as input
        result = torch.stack(denoised_batch).to(x.device)

        # Ensure output is in valid range
        result = torch.clamp(result, 0.0, 1.0)

        return result


if __name__ == "__main__":
    # Test Wavelet Denoising
    print("Testing Wavelet Denoising...")

    # Create test noisy image
    torch.manual_seed(42)
    clean_img = torch.rand(2, 3, 32, 32)  # Batch of 2 images
    noise = torch.randn_like(clean_img) * 0.05
    noisy_img = torch.clamp(clean_img + noise, 0, 1)

    # Test with different wavelets and modes
    for wavelet in ["db1", "db4", "sym4"]:
        for mode in ["soft", "hard"]:
            print(f"\nTesting {wavelet} with {mode} thresholding...")

            wvlt = WaveletDenoising(wavelet=wavelet, mode=mode, level=2)
            print(f"Model: {wvlt.name}")
            print(
                f"Parameters: wavelet={wvlt.wavelet}, mode={wvlt.mode}, level={wvlt.level}"
            )

            denoised = wvlt(noisy_img)
            print(f"Input shape: {noisy_img.shape}")
            print(f"Output shape: {denoised.shape}")
            print(f"Output range: [{denoised.min():.3f}, {denoised.max():.3f}]")

            # Calculate improvement
            mse_noisy = torch.mean((noisy_img - clean_img) ** 2).item()
            mse_denoised = torch.mean((denoised - clean_img) ** 2).item()
            print(f"MSE before: {mse_noisy:.6f}, after: {mse_denoised:.6f}")

    print("\n✓ Wavelet Denoising implementation complete!")
