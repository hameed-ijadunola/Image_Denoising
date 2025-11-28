"""
Quick demo script to test the implementation
This script trains a small model for 1 epoch to verify everything works
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from unet import UNet
from dataset import get_dataloaders
from train import Trainer
from utils import evaluate_model, visualize_denoising


def main():
    print("\n" + "=" * 70)
    print("U-NET IMAGE DENOISING - QUICK DEMO")
    print("=" * 70)
    print("\nThis demo will:")
    print("  1. Load a small batch of CIFAR-10 images")
    print("  2. Add Gaussian noise")
    print("  3. Train U-Net for 1 epoch")
    print("  4. Evaluate PSNR improvement")
    print("  5. Visualize results")
    print("\n" + "=" * 70 + "\n")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load small dataset for demo
    print("\nLoading CIFAR-10 dataset (this may take a moment)...")
    train_loader, test_loader = get_dataloaders(
        batch_size=16,
        noise_type="gaussian",
        noise_param=0.05,
        num_workers=0,  # Use 0 for demo to avoid multiprocessing issues
    )
    print(f"✓ Loaded {len(train_loader.dataset)} training images")
    print(f"✓ Loaded {len(test_loader.dataset)} test images")

    # Create model
    print("\nCreating U-Net model...")
    model = UNet(n_channels=3, n_classes=3, dropout_rate=0.2)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {num_params:,} parameters")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_name="adam",
        lr=0.001,
        save_dir="./models/demo",
    )
    print("✓ Trainer ready")

    # Train for 1 epoch
    print("\nTraining for 1 epoch (this will take a few minutes)...")
    train_losses, test_losses = trainer.train(num_epochs=1)

    # Evaluate
    print("\nEvaluating model...")
    avg_psnr_noisy, avg_psnr_denoised = evaluate_model(model, test_loader, device)

    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Average Noisy PSNR:     {avg_psnr_noisy:.2f} dB")
    print(f"Average Denoised PSNR:  {avg_psnr_denoised:.2f} dB")
    print(f"PSNR Improvement:       +{avg_psnr_denoised - avg_psnr_noisy:.2f} dB")
    print(f"Final Training Loss:    {train_losses[-1]:.5f}")
    print(f"Final Test Loss:        {test_losses[-1]:.5f}")
    print("=" * 70)

    # Visualize
    print("\nGenerating visualization...")
    visualize_denoising(
        model,
        test_loader,
        device,
        num_samples=3,
        save_path="./results/demo_samples.png",
    )

    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  • Train for more epochs: python main.py --mode train --epochs 10")
    print("  • Try different optimizers: --optimizer rmsprop or --optimizer sgd")
    print("  • Try different noise types: --noise-type poisson")
    print("  • See README.md for more options")
    print("")


if __name__ == "__main__":
    main()
