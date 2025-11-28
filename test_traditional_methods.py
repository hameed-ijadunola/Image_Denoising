"""
Test script for all traditional denoising methods

This script tests:
1. Non-Local Means (NLM)
2. Bilateral Filter
3. Wavelet Denoising

All methods should work without training and produce valid outputs.
"""

import torch
import sys
from config import MODEL_REGISTRY
from src.registry import get_model_from_registry


def test_traditional_method(method_key: str):
    """Test a traditional denoising method from the registry"""
    print("\n" + "=" * 70)
    print(f"Testing: {method_key.upper()}")
    print("=" * 70)

    # Check if method exists in registry
    assert method_key in MODEL_REGISTRY, f"Method '{method_key}' not found in registry"

    config = MODEL_REGISTRY[method_key]
    print(f"Name: {config['name']}")
    print(f"Type: {config['type']}")
    print(f"Paper: {config.get('paper', 'N/A')}")
    print(f"Requires training: {config['requires_training']}")
    print(f"Default params: {config['default_params']}")

    # Load model from registry
    print("\nLoading model from registry...")
    model = get_model_from_registry(method_key, MODEL_REGISTRY)
    print(f"✓ Model loaded: {model.name}")

    # Create synthetic test data
    print("\nCreating test data...")
    torch.manual_seed(42)
    batch_size = 4
    clean_img = torch.rand(batch_size, 3, 32, 32)

    # Add Gaussian noise
    noise_std = 0.1
    noise = torch.randn_like(clean_img) * noise_std
    noisy_img = torch.clamp(clean_img + noise, 0, 1)

    print(f"  Clean image shape: {clean_img.shape}")
    print(f"  Noisy image range: [{noisy_img.min():.3f}, {noisy_img.max():.3f}]")
    print(f"  Noise std: {noise_std}")

    # Test denoising
    print("\nApplying denoising...")
    with torch.no_grad():
        denoised = model(noisy_img)

    print(f"  Denoised image shape: {denoised.shape}")
    print(f"  Denoised image range: [{denoised.min():.3f}, {denoised.max():.3f}]")

    # Validate output
    assert denoised.shape == noisy_img.shape, "Output shape mismatch"
    assert 0 <= denoised.min() and denoised.max() <= 1.0, (
        "Output values out of range [0, 1]"
    )
    assert torch.isfinite(denoised).all(), "Output contains NaN or Inf"

    # Calculate metrics
    print("\nMetrics:")
    mse_noisy = torch.mean((noisy_img - clean_img) ** 2).item()
    mse_denoised = torch.mean((denoised - clean_img) ** 2).item()
    psnr_noisy = -10 * torch.log10(torch.tensor(mse_noisy)).item()
    psnr_denoised = -10 * torch.log10(torch.tensor(mse_denoised)).item()

    print(f"  MSE (noisy):    {mse_noisy:.6f}")
    print(f"  MSE (denoised): {mse_denoised:.6f}")
    print(f"  PSNR (noisy):    {psnr_noisy:.2f} dB")
    print(f"  PSNR (denoised): {psnr_denoised:.2f} dB")
    print(f"  Improvement:     {psnr_denoised - psnr_noisy:+.2f} dB")

    if mse_denoised < mse_noisy:
        print("  ✓ Denoising improved image quality")
    else:
        print("  ⚠ Denoising did not improve MSE (may still preserve edges better)")

    print(f"\n✓ {config['name']} test passed!")
    return True


def main():
    """Test all traditional methods"""
    print("\n" + "=" * 70)
    print("TESTING ALL TRADITIONAL DENOISING METHODS")
    print("=" * 70)

    # List all traditional methods in registry
    traditional_methods = [
        key
        for key, config in MODEL_REGISTRY.items()
        if config.get("type") == "traditional"
    ]

    print(f"\nFound {len(traditional_methods)} traditional methods:")
    for method in traditional_methods:
        print(f"  - {method}: {MODEL_REGISTRY[method]['name']}")

    # Test each method
    results = {}
    for method_key in traditional_methods:
        try:
            test_traditional_method(method_key)
            results[method_key] = "PASSED"
        except Exception as e:
            print(f"\n✗ {method_key} test failed: {e}")
            results[method_key] = f"FAILED: {e}"
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for method, status in results.items():
        status_symbol = "✓" if status == "PASSED" else "✗"
        print(f"{status_symbol} {method}: {status}")

    # Check if all passed
    all_passed = all(status == "PASSED" for status in results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
