"""
Evaluation and visualization utilities for image denoising
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from math import log10
from skimage.metrics import structural_similarity as ssim
import lpips


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


def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index (SSIM) between two images

    Args:
        img1: First image tensor (B, C, H, W) or (C, H, W)
        img2: Second image tensor (same shape as img1)

    Returns:
        SSIM value (0 to 1, higher is better)
    """
    # Convert to numpy and ensure proper shape
    if len(img1.shape) == 4:  # Batch dimension
        img1 = img1[0]
        img2 = img2[0]

    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()

    # SSIM expects (H, W, C) format
    img1_np = np.transpose(img1_np, (1, 2, 0))
    img2_np = np.transpose(img2_np, (1, 2, 0))

    # Calculate SSIM
    ssim_value = ssim(img1_np, img2_np, data_range=1.0, channel_axis=2)
    return ssim_value


def calculate_mse(img1, img2):
    """
    Calculate Mean Squared Error (MSE) between two images

    Args:
        img1: First image tensor (B, C, H, W) or (C, H, W)
        img2: Second image tensor (same shape as img1)

    Returns:
        MSE value (lower is better)
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    return mse


def calculate_mae(img1, img2):
    """
    Calculate Mean Absolute Error (MAE) between two images

    Args:
        img1: First image tensor (B, C, H, W) or (C, H, W)
        img2: Second image tensor (same shape as img1)

    Returns:
        MAE value (lower is better)
    """
    mae = torch.mean(torch.abs(img1 - img2)).item()
    return mae


def calculate_lpips(img1, img2, lpips_model=None):
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS)

    Args:
        img1: First image tensor (B, C, H, W) or (C, H, W)
        img2: Second image tensor (same shape as img1)
        lpips_model: Pre-initialized LPIPS model (optional)

    Returns:
        LPIPS value (lower is better, typically 0-1)
    """
    # Ensure batch dimension
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # Initialize LPIPS model if not provided
    if lpips_model is None:
        lpips_model = lpips.LPIPS(net="alex").to(img1.device)

    # Assert device consistency - fail fast if tensors on wrong device
    assert img1.device == img2.device, f"img1 on {img1.device}, img2 on {img2.device}"

    # Get the device of the lpips model's parameters
    lpips_device = next(lpips_model.parameters()).device
    assert img1.device == lpips_device, (
        f"Input tensors on {img1.device} but LPIPS model on {lpips_device}. "
        f"Tensors must be on same device as model."
    )

    # LPIPS expects images in range [-1, 1]
    img1_scaled = img1 * 2 - 1
    img2_scaled = img2 * 2 - 1

    with torch.no_grad():
        lpips_value = lpips_model(img1_scaled, img2_scaled).item()

    return lpips_value


def calculate_all_metrics(img1, img2, lpips_model=None):
    """
    Calculate all metrics between two images

    Args:
        img1: First image tensor (B, C, H, W) or (C, H, W)
        img2: Second image tensor (same shape as img1)
        lpips_model: Pre-initialized LPIPS model (optional)

    Returns:
        Dictionary with all metric values (as Python float)
    """
    # Calculate metrics and ensure they're Python native floats (not numpy types)
    # This prevents JSON serialization errors
    metrics = {
        "psnr": float(calculate_psnr(img1, img2)),
        "ssim": float(calculate_ssim(img1, img2)),
        "mse": float(calculate_mse(img1, img2)),
        "mae": float(calculate_mae(img1, img2)),
        "lpips": float(calculate_lpips(img1, img2, lpips_model)),
    }

    return metrics


def evaluate_model(model, data_loader, device):
    """
    Evaluate model on a dataset and calculate average metrics

    Args:
        model: Trained U-Net model
        data_loader: DataLoader with noisy and clean images
        device: torch device

    Returns:
        metrics_noisy: Dictionary with average metrics for noisy images
        metrics_denoised: Dictionary with average metrics for denoised images
    """
    model.eval()

    # Initialize metric lists
    metrics_lists_noisy = {
        "psnr": [],
        "ssim": [],
        "mse": [],
        "mae": [],
        "lpips": [],
    }
    metrics_lists_denoised = {
        "psnr": [],
        "ssim": [],
        "mse": [],
        "mae": [],
        "lpips": [],
    }

    lpips_model = lpips.LPIPS(net="alex").to(device)

    with torch.no_grad():
        for noisy_images, clean_images in data_loader:
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            # Denoise images
            denoised_images = model(noisy_images)

            # Calculate metrics for each image in batch
            for i in range(noisy_images.size(0)):
                # Metrics for noisy images
                metrics_noisy = calculate_all_metrics(
                    noisy_images[i], clean_images[i], lpips_model
                )
                # Metrics for denoised images
                metrics_denoised = calculate_all_metrics(
                    denoised_images[i], clean_images[i], lpips_model
                )

                # Append to lists
                for key in metrics_noisy:
                    if metrics_noisy[key] is not None:
                        metrics_lists_noisy[key].append(metrics_noisy[key])
                for key in metrics_denoised:
                    if metrics_denoised[key] is not None:
                        metrics_lists_denoised[key].append(metrics_denoised[key])

    # Calculate averages and convert to Python float for JSON serialization
    avg_metrics_noisy = {
        key: float(np.mean(values))
        for key, values in metrics_lists_noisy.items()
        if values
    }
    avg_metrics_denoised = {
        key: float(np.mean(values))
        for key, values in metrics_lists_denoised.items()
        if values
    }

    return avg_metrics_noisy, avg_metrics_denoised


def visualize_denoising(model, data_loader, device, num_samples=5, save_path=None):
    """
    Visualize denoising results with multiple metrics

    Args:
        model: Trained U-Net model
        data_loader: DataLoader with noisy and clean images
        device: torch device
        num_samples: Number of samples to visualize
        save_path: Path to save the figure
    """
    model.eval()

    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net="alex").to(device)

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

        # Calculate all metrics using original tensors (on correct device)
        # Don't convert to numpy and back - this puts tensors on CPU
        metrics_noisy = calculate_all_metrics(
            noisy_images[i], clean_images[i], lpips_model
        )
        metrics_denoised = calculate_all_metrics(
            denoised_images[i], clean_images[i], lpips_model
        )

        # Plot noisy image
        axes[i, 0].imshow(noisy_img)
        title_noisy = f"Noisy\nPSNR: {metrics_noisy['psnr']:.2f} dB | SSIM: {metrics_noisy['ssim']:.3f}"
        axes[i, 0].set_title(title_noisy, fontsize=9)
        axes[i, 0].axis("off")

        # Plot denoised image
        axes[i, 1].imshow(denoised_img)
        title_denoised = f"Denoised\nPSNR: {metrics_denoised['psnr']:.2f} dB | SSIM: {metrics_denoised['ssim']:.3f}"
        axes[i, 1].set_title(title_denoised, fontsize=9)
        axes[i, 1].axis("off")

        # Plot clean image
        axes[i, 2].imshow(clean_img)
        axes[i, 2].set_title("Clean (Ground Truth)", fontsize=9)
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
    (Legacy function - use plot_metrics_comparison for all metrics)

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


def plot_metrics_comparison(metrics_noisy, metrics_denoised, save_path=None):
    """
    Plot comprehensive metrics comparison between noisy and denoised images

    Args:
        metrics_noisy: Dictionary with average metrics for noisy images
        metrics_denoised: Dictionary with average metrics for denoised images
        save_path: Path to save the figure
    """
    # Define metric properties (name, higher_is_better, unit)
    metric_info = {
        "psnr": ("PSNR", True, "dB"),
        "ssim": ("SSIM", True, ""),
        "mse": ("MSE", False, ""),
        "mae": ("MAE", False, ""),
        "lpips": ("LPIPS", False, ""),
    }

    # Filter available metrics
    available_metrics = [k for k in metric_info.keys() if k in metrics_noisy]
    n_metrics = len(available_metrics)

    # Create subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for idx, metric_key in enumerate(available_metrics):
        metric_name, higher_is_better, unit = metric_info[metric_key]

        noisy_val = metrics_noisy[metric_key]
        denoised_val = metrics_denoised[metric_key]

        categories = ["Noisy", "Denoised"]
        values = [noisy_val, denoised_val]

        # Choose colors based on improvement
        if higher_is_better:
            colors = (
                ["#ff6b6b", "#4ecdc4"]
                if denoised_val > noisy_val
                else ["#4ecdc4", "#ff6b6b"]
            )
        else:
            colors = (
                ["#ff6b6b", "#4ecdc4"]
                if denoised_val < noisy_val
                else ["#4ecdc4", "#ff6b6b"]
            )

        # Plot bars
        bars = axes[idx].bar(
            categories,
            values,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels on bars
        for bar, val in zip(bars, values):
            label_text = (
                f"{val:.4f}" if metric_key in ["mse", "mae", "lpips"] else f"{val:.2f}"
            )
            if unit:
                label_text += f" {unit}"
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                label_text,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Set labels and title
        ylabel = f"{metric_name}"
        if unit:
            ylabel += f" ({unit})"
        axes[idx].set_ylabel(ylabel, fontsize=11)
        axes[idx].set_title(metric_name, fontsize=12, fontweight="bold")
        axes[idx].grid(True, axis="y", alpha=0.3)

        # Set y-axis limits
        max_val = max(values)
        axes[idx].set_ylim(0, max_val * 1.15)

    plt.suptitle(
        "Metrics Comparison: Noisy vs Denoised Images",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Metrics comparison plot saved to: {save_path}")

    plt.show()

    # Print improvements
    print("\n" + "=" * 60)
    print("Metric Improvements:")
    print("=" * 60)
    for metric_key in available_metrics:
        metric_name, higher_is_better, unit = metric_info[metric_key]
        noisy_val = metrics_noisy[metric_key]
        denoised_val = metrics_denoised[metric_key]

        if higher_is_better:
            improvement = denoised_val - noisy_val
            pct_improvement = (improvement / noisy_val) * 100
            sign = "+"
        else:
            improvement = noisy_val - denoised_val
            pct_improvement = (improvement / noisy_val) * 100
            sign = "-" if improvement < 0 else "+"

        unit_str = f" {unit}" if unit else ""
        print(
            f"{metric_name:8} | Noisy: {noisy_val:.4f}{unit_str} | "
            f"Denoised: {denoised_val:.4f}{unit_str} | "
            f"Improvement: {sign}{abs(improvement):.4f} ({pct_improvement:+.2f}%)"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test PSNR calculation
    img1 = torch.rand(3, 32, 32)
    img2 = img1 + torch.randn(3, 32, 32) * 0.1
    psnr = calculate_psnr(img1, img2)
    print(f"Test PSNR: {psnr:.2f} dB")
