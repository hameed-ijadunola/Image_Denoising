"""
Demo script for R3L (Residual Recovery using Reinforcement Learning)

This script demonstrates:
1. Model initialization
2. Training on a small batch
3. Inference/denoising
4. Comparison with ground truth
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from src.r3l import R3L
from src.r3l_trainer import R3LTrainer


def create_synthetic_data(num_samples=100, image_size=32, noise_level=0.1):
    """Create synthetic noisy images for demonstration"""
    print(f"Creating {num_samples} synthetic images ({image_size}x{image_size})...")

    # Generate clean images (simple patterns)
    clean_images = torch.rand(num_samples, 3, image_size, image_size)

    # Add Gaussian noise
    noise = torch.randn_like(clean_images) * noise_level
    noisy_images = torch.clamp(clean_images + noise, 0, 1)

    return noisy_images, clean_images


def demo_r3l_inference():
    """Demonstrate R3L inference on a single image"""
    print("\n" + "=" * 70)
    print("R3L INFERENCE DEMO")
    print("=" * 70)

    # Create model
    print("\nInitializing R3L model...")
    model = R3L(in_channels=3, num_stages=5, action_range=(-13, 13), gamma=0.95)
    model.eval()

    # Create a noisy image
    clean = torch.rand(1, 3, 64, 64)
    noise = torch.randn_like(clean) * 0.1
    noisy = torch.clamp(clean + noise, 0, 1)

    print(f"Input image shape: {noisy.shape}")
    print(f"Noise level: 0.1 (sigma)")

    # Denoise
    print("\nRunning R3L denoising (5 stages)...")
    with torch.no_grad():
        denoised = model(noisy, training=False)

    # Calculate metrics
    psnr_noisy = -10 * torch.log10(F.mse_loss(noisy, clean))
    psnr_denoised = -10 * torch.log10(F.mse_loss(denoised, clean))

    print(f"\nResults:")
    print(f"  Noisy PSNR: {psnr_noisy:.2f} dB")
    print(f"  Denoised PSNR: {psnr_denoised:.2f} dB")
    print(f"  Improvement: {psnr_denoised - psnr_noisy:.2f} dB")

    # Note: Random initialization won't denoise well - this is just a demo!
    print("\nNote: Model is randomly initialized, so results may vary.")
    print("Train the model to see actual denoising performance!")


def demo_r3l_training():
    """Demonstrate R3L training on synthetic data"""
    print("\n" + "=" * 70)
    print("R3L TRAINING DEMO")
    print("=" * 70)

    # Create synthetic dataset
    train_noisy, train_clean = create_synthetic_data(
        num_samples=50, image_size=32, noise_level=0.1
    )
    val_noisy, val_clean = create_synthetic_data(
        num_samples=10, image_size=32, noise_level=0.1
    )

    # Create dataloaders
    train_dataset = TensorDataset(train_noisy, train_clean)
    val_dataset = TensorDataset(val_noisy, val_clean)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Create model
    print("\nInitializing R3L model...")
    model = R3L(in_channels=3, num_stages=5, action_range=(-13, 13), gamma=0.95)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create trainer
    print("\nCreating R3L trainer...")
    trainer = R3LTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate_policy=1e-4,
        learning_rate_value=1e-3,
        num_epochs=3,  # Just a few epochs for demo
        save_dir="models/r3l_demo",
        use_wandb=False,
    )

    # Train
    print("\nTraining for 3 epochs (demo only)...")
    print("Note: Real training requires many more epochs!\n")

    history = trainer.train()

    print("\nTraining complete!")
    print(f"Final validation PSNR: {history['val_psnr'][-1]:.2f} dB")

    return model, history


def visualize_denoising(model, device="cpu"):
    """Visualize R3L denoising process"""
    print("\n" + "=" * 70)
    print("R3L DENOISING VISUALIZATION")
    print("=" * 70)

    model.eval()
    model.to(device)

    # Create a noisy image
    clean = torch.rand(1, 3, 64, 64).to(device)
    noise = torch.randn_like(clean) * 0.15
    noisy = torch.clamp(clean + noise, 0, 1)

    # Denoise stage by stage
    print("\nDenoising image through 5 stages...")
    stages = [noisy.clone()]

    current = noisy.clone()
    with torch.no_grad():
        for stage in range(5):
            state = model.encode_state(current)
            policy = model.get_policy(state)
            _, residual = model.sample_action(policy, training=False)
            current = torch.clamp(current + residual, 0, 1)
            stages.append(current.clone())

    # Calculate PSNR for each stage
    print("\nPSNR progression:")
    for i, stage_img in enumerate(stages):
        psnr = -10 * torch.log10(F.mse_loss(stage_img, clean))
        stage_name = "Noisy" if i == 0 else f"Stage {i}"
        print(f"  {stage_name}: {psnr:.2f} dB")

    # Plot
    try:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        # Convert to numpy for plotting
        def to_numpy(img):
            return img[0].cpu().permute(1, 2, 0).numpy()

        # Plot stages
        titles = ["Noisy Input", "Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]
        for i in range(6):
            axes[i].imshow(np.clip(to_numpy(stages[i]), 0, 1))
            axes[i].set_title(titles[i])
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig("results/r3l_demo_stages.png", dpi=150, bbox_inches="tight")
        print("\nVisualization saved to: results/r3l_demo_stages.png")
        plt.close()
    except Exception as e:
        print(f"\nVisualization failed: {e}")


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("R3L (Residual Recovery using Reinforcement Learning) DEMO")
    print("=" * 70)
    print("\nThis demo showcases the R3L model implementation.")
    print("Paper: Zhang et al., 2021 - Connecting Deep Reinforcement")
    print("       Learning to Recurrent Neural Networks for Image Denoising")

    # Demo 1: Inference
    demo_r3l_inference()

    # Demo 2: Training (optional - commented out by default to save time)
    print("\n\nSkipping training demo (uncomment in script to enable)")
    # model, history = demo_r3l_training()
    # visualize_denoising(model)

    # Demo 3: Show architecture
    print("\n" + "=" * 70)
    print("R3L ARCHITECTURE SUMMARY")
    print("=" * 70)

    model = R3L()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    policy_params = sum(p.numel() for p in model.policy_net.parameters())
    value_params = sum(p.numel() for p in model.value_net.parameters())

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nBreakdown:")
    print(f"  FCN Encoder: {encoder_params:,}")
    print(f"  Policy Network: {policy_params:,}")
    print(f"  Value Network: {value_params:,}")

    print(f"\nModel configuration:")
    print(f"  Input channels: {model.in_channels}")
    print(f"  Number of stages: {model.num_stages}")
    print(f"  Action space size: {model.action_space_size}")
    print(f"  Action range: [{model.action_min}, {model.action_max}]")
    print(f"  Discount factor (gamma): {model.gamma}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("\nTo train R3L on real data, run:")
    print("  python main.py --mode train --model r3l --epochs 50 --batch-size 32")
    print("\nFor more information, see:")
    print("  docs/R3L_implementation.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
