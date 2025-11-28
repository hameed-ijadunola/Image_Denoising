"""
Test script to verify that the new metrics are working correctly
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import (
    calculate_psnr,
    calculate_ssim,
    calculate_mse,
    calculate_mae,
    calculate_lpips,
    calculate_all_metrics,
)


def test_metrics():
    """Test all metric functions"""
    print("\n" + "=" * 70)
    print("TESTING IMAGE METRICS")
    print("=" * 70 + "\n")

    # Create test images
    print("Creating test images...")
    torch.manual_seed(42)

    # Clean image
    clean_img = torch.rand(3, 32, 32)

    # Slightly noisy image
    noisy_img = clean_img + torch.randn(3, 32, 32) * 0.1
    noisy_img = torch.clamp(noisy_img, 0, 1)

    # Very noisy image
    very_noisy_img = clean_img + torch.randn(3, 32, 32) * 0.3
    very_noisy_img = torch.clamp(very_noisy_img, 0, 1)

    print("✓ Test images created\n")

    # Test individual metrics
    print("Testing individual metric functions:")
    print("-" * 70)

    # Test PSNR
    psnr_slight = calculate_psnr(noisy_img, clean_img)
    psnr_very = calculate_psnr(very_noisy_img, clean_img)
    print(f"PSNR (slight noise): {psnr_slight:.2f} dB")
    print(f"PSNR (heavy noise):  {psnr_very:.2f} dB")
    print("✓ PSNR works correctly (higher is better)\n")

    # Test SSIM
    ssim_slight = calculate_ssim(noisy_img, clean_img)
    ssim_very = calculate_ssim(very_noisy_img, clean_img)
    print(f"SSIM (slight noise): {ssim_slight:.4f}")
    print(f"SSIM (heavy noise):  {ssim_very:.4f}")
    print("✓ SSIM works correctly (higher is better, range 0-1)\n")

    # Test MSE
    mse_slight = calculate_mse(noisy_img, clean_img)
    mse_very = calculate_mse(very_noisy_img, clean_img)
    print(f"MSE (slight noise):  {mse_slight:.6f}")
    print(f"MSE (heavy noise):   {mse_very:.6f}")
    print("✓ MSE works correctly (lower is better)\n")

    # Test MAE
    mae_slight = calculate_mae(noisy_img, clean_img)
    mae_very = calculate_mae(very_noisy_img, clean_img)
    print(f"MAE (slight noise):  {mae_slight:.6f}")
    print(f"MAE (heavy noise):   {mae_very:.6f}")
    print("✓ MAE works correctly (lower is better)\n")

    # Test LPIPS (if available)
    try:
        lpips_slight = calculate_lpips(noisy_img, clean_img)
        lpips_very = calculate_lpips(very_noisy_img, clean_img)
        print(f"LPIPS (slight noise): {lpips_slight:.6f}")
        print(f"LPIPS (heavy noise):  {lpips_very:.6f}")
        print("✓ LPIPS works correctly (lower is better)\n")
    except Exception as e:
        print(f"⚠ LPIPS not available: {e}\n")

    # Test calculate_all_metrics
    print("\nTesting calculate_all_metrics function:")
    print("-" * 70)
    metrics = calculate_all_metrics(noisy_img, clean_img)

    print("Metrics for slightly noisy image:")
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key.upper()}: {value:.6f}")

    print("\n✓ All metrics calculated successfully!")

    # Verify metric behavior
    print("\n" + "=" * 70)
    print("VERIFICATION:")
    print("=" * 70)

    # PSNR should be higher for less noisy image
    assert psnr_slight > psnr_very, "PSNR verification failed"
    print("✓ PSNR: Higher for less noisy image")

    # SSIM should be higher for less noisy image
    assert ssim_slight > ssim_very, "SSIM verification failed"
    print("✓ SSIM: Higher for less noisy image")

    # MSE should be lower for less noisy image
    assert mse_slight < mse_very, "MSE verification failed"
    print("✓ MSE: Lower for less noisy image")

    # MAE should be lower for less noisy image
    assert mae_slight < mae_very, "MAE verification failed"
    print("✓ MAE: Lower for less noisy image")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_metrics()
