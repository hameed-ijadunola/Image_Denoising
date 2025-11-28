"""
Evaluation and visualization utilities for image denoising
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from math import log10


def calculate_psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images

    Args:
        img1: First image tensor (B, C, H, W) or (C, H, W)
        img2: Second image tensor (same shape as img1)

    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")

    max_pixel = 1.0  # Assuming images are normalized to [0, 1]
    psnr = 20 * log10(max_pixel / torch.sqrt(mse).item())
    return psnr


def evaluate_model(model, data_loader, device):
    """
    Evaluate model on a dataset and calculate average PSNR

    Args:
        model: Trained U-Net model
        data_loader: DataLoader with noisy and clean images
        device: torch device

    Returns:
        avg_psnr_noisy: Average PSNR of noisy images
        avg_psnr_denoised: Average PSNR of denoised images
    """
    model.eval()
    psnr_noisy_list = []
    psnr_denoised_list = []

    with torch.no_grad():
        for noisy_images, clean_images in data_loader:
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            # Denoise images
            denoised_images = model(noisy_images)

            # Calculate PSNR for each image in batch
            for i in range(noisy_images.size(0)):
                psnr_noisy = calculate_psnr(noisy_images[i], clean_images[i])
                psnr_denoised = calculate_psnr(denoised_images[i], clean_images[i])

                psnr_noisy_list.append(psnr_noisy)
                psnr_denoised_list.append(psnr_denoised)

    avg_psnr_noisy = np.mean(psnr_noisy_list)
    avg_psnr_denoised = np.mean(psnr_denoised_list)

    return avg_psnr_noisy, avg_psnr_denoised


def visualize_denoising(model, data_loader, device, num_samples=5, save_path=None):
    """
    Visualize denoising results

    Args:
        model: Trained U-Net model
        data_loader: DataLoader with noisy and clean images
        device: torch device
        num_samples: Number of samples to visualize
        save_path: Path to save the figure
    """
    model.eval()

    # Get a batch of images
    noisy_images, clean_images = next(iter(data_loader))
    noisy_images = noisy_images.to(device)
    clean_images = clean_images.to(device)

    # Denoise images
    with torch.no_grad():
        denoised_images = model(noisy_images)

    # Move to CPU and convert to numpy
    noisy_np = noisy_images.cpu().numpy()
    clean_np = clean_images.cpu().numpy()
    denoised_np = denoised_images.cpu().numpy()

    # Plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(min(num_samples, noisy_np.shape[0])):
        # Transpose from (C, H, W) to (H, W, C) for plotting
        noisy_img = np.transpose(noisy_np[i], (1, 2, 0))
        clean_img = np.transpose(clean_np[i], (1, 2, 0))
        denoised_img = np.transpose(denoised_np[i], (1, 2, 0))

        # Clip values to [0, 1]
        noisy_img = np.clip(noisy_img, 0, 1)
        clean_img = np.clip(clean_img, 0, 1)
        denoised_img = np.clip(denoised_img, 0, 1)

        # Calculate PSNR
        psnr_noisy = calculate_psnr(
            torch.from_numpy(noisy_np[i]), torch.from_numpy(clean_np[i])
        )
        psnr_denoised = calculate_psnr(
            torch.from_numpy(denoised_np[i]), torch.from_numpy(clean_np[i])
        )

        # Plot noisy image
        axes[i, 0].imshow(noisy_img)
        axes[i, 0].set_title(f"Noisy (PSNR: {psnr_noisy:.2f} dB)")
        axes[i, 0].axis("off")

        # Plot denoised image
        axes[i, 1].imshow(denoised_img)
        axes[i, 1].set_title(f"Denoised (PSNR: {psnr_denoised:.2f} dB)")
        axes[i, 1].axis("off")

        # Plot clean image
        axes[i, 2].imshow(clean_img)
        axes[i, 2].set_title("Clean (Ground Truth)")
        axes[i, 2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")

    plt.show()


def plot_training_history(train_losses, test_losses, save_path=None):
    """
    Plot training and test loss curves

    Args:
        train_losses: List of training losses
        test_losses: List of test losses
        save_path: Path to save the figure
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs, test_losses, "r-", label="Test Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("Training and Test Loss over Epochs", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Loss plot saved to: {save_path}")

    plt.show()


def plot_psnr_comparison(psnr_noisy, psnr_denoised, save_path=None):
    """
    Plot PSNR comparison between noisy and denoised images

    Args:
        psnr_noisy: Average PSNR of noisy images
        psnr_denoised: Average PSNR of denoised images
        save_path: Path to save the figure
    """
    categories = ["Noisy", "Denoised"]
    psnr_values = [psnr_noisy, psnr_denoised]
    colors = ["#ff6b6b", "#4ecdc4"]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        categories,
        psnr_values,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, psnr_values)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f} dB",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.title(
        "PSNR Comparison: Noisy vs Denoised Images", fontsize=14, fontweight="bold"
    )
    plt.ylim(0, max(psnr_values) * 1.2)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"PSNR comparison plot saved to: {save_path}")

    plt.show()

    # Print improvement
    improvement = psnr_denoised - psnr_noisy
    print(f"\nPSNR Improvement: +{improvement:.2f} dB")
    print(f"Percentage Improvement: {(improvement / psnr_noisy) * 100:.2f}%")


if __name__ == "__main__":
    # Test PSNR calculation
    img1 = torch.rand(3, 32, 32)
    img2 = img1 + torch.randn(3, 32, 32) * 0.1
    psnr = calculate_psnr(img1, img2)
    print(f"Test PSNR: {psnr:.2f} dB")
