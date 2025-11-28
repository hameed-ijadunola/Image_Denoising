"""
Main script for training and evaluating U-Net image denoising
"""

import torch
import argparse
import os
import json
from src.unet import UNet
from src.dataset import get_dataloaders
from src.train import Trainer, load_checkpoint
from src.utils import (
    evaluate_model,
    visualize_denoising,
    plot_training_history,
    plot_psnr_comparison,
    plot_metrics_comparison,
)
from config import WANDB_CONFIG
import wandb


def train_model(args):
    """Train the U-Net model"""
    print("\n" + "=" * 70)
    print("IMAGE DENOISING USING U-NET")
    print("=" * 70)

    # Setup wandb
    use_wandb = args.use_wandb
    if use_wandb:
        run_name = f"{args.noise_type}_{args.optimizer}_lr{args.lr}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "optimizer": args.optimizer,
                "noise_type": args.noise_type,
                "noise_param": args.noise_param,
                "dropout": args.dropout,
                "bilinear": args.bilinear,
                "num_workers": args.num_workers,
            },
            save_code=args.wandb_save_code,
        )
        print(f"\n✓ Weights & Biases initialized")
        print(f"  Project: {args.wandb_project}")
        print(f"  Run: {run_name}")

    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"\nDevice: {device}")

    # Load data
    print(f"\nLoading CIFAR-10 dataset...")
    print(f"  Noise type: {args.noise_type}")
    print(f"  Noise parameter: {args.noise_param}")
    print(f"  Batch size: {args.batch_size}")

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        noise_type=args.noise_type,
        noise_param=args.noise_param,
        num_workers=args.num_workers,
    )

    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")

    # Create model
    print(f"\nCreating U-Net model...")
    model = UNet(
        n_channels=3, n_classes=3, bilinear=args.bilinear, dropout_rate=args.dropout
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")

    # Create save directory
    save_dir = os.path.join(
        args.save_dir, f"{args.noise_type}_{args.optimizer}_lr{args.lr}"
    )

    # Create trainer
    wandb_config = {
        "log_interval": args.wandb_log_interval,
        "log_images": args.wandb_log_images,
        "log_model": args.wandb_log_model,
    }

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_name=args.optimizer,
        lr=args.lr,
        save_dir=save_dir,
        use_wandb=use_wandb,
        wandb_config=wandb_config,
    )

    # Train
    train_losses, test_losses = trainer.train(num_epochs=args.epochs)

    # Plot training history
    plot_path = os.path.join(
        args.results_dir, f"{args.noise_type}_{args.optimizer}_loss.png"
    )
    plot_training_history(train_losses, test_losses, save_path=plot_path)

    # Evaluate
    print("\nEvaluating model on test set...")
    metrics_noisy, metrics_denoised = evaluate_model(model, test_loader, device)

    print("\nResults:")
    print("Noisy Images:")
    for key, value in metrics_noisy.items():
        print(f"  {key.upper()}: {value:.4f}")

    print("\nDenoised Images:")
    for key, value in metrics_denoised.items():
        print(f"  {key.upper()}: {value:.4f}")

    # Log to wandb
    if use_wandb:
        log_dict = {}
        for key, value in metrics_noisy.items():
            log_dict[f"final_{key}_noisy"] = value
        for key, value in metrics_denoised.items():
            log_dict[f"final_{key}_denoised"] = value
        wandb.log(log_dict)

    # Plot PSNR comparison (legacy)
    psnr_plot_path = os.path.join(
        args.results_dir, f"{args.noise_type}_{args.optimizer}_psnr.png"
    )
    if "psnr" in metrics_noisy and "psnr" in metrics_denoised:
        plot_psnr_comparison(
            metrics_noisy["psnr"], metrics_denoised["psnr"], save_path=psnr_plot_path
        )

    # Plot all metrics comparison
    metrics_plot_path = os.path.join(
        args.results_dir, f"{args.noise_type}_{args.optimizer}_metrics.png"
    )
    plot_metrics_comparison(
        metrics_noisy, metrics_denoised, save_path=metrics_plot_path
    )

    # Visualize denoising
    vis_path = os.path.join(
        args.results_dir, f"{args.noise_type}_{args.optimizer}_samples.png"
    )
    visualize_denoising(model, test_loader, device, num_samples=5, save_path=vis_path)

    # Log images to wandb
    if use_wandb and args.wandb_log_images:
        # Log training history plot
        if os.path.exists(plot_path):
            wandb.log({"training_loss_curve": wandb.Image(plot_path)})

        # Log PSNR comparison
        if os.path.exists(psnr_plot_path):
            wandb.log({"psnr_comparison": wandb.Image(psnr_plot_path)})

        # Log metrics comparison
        if os.path.exists(metrics_plot_path):
            wandb.log({"metrics_comparison": wandb.Image(metrics_plot_path)})

        # Log denoising samples
        if os.path.exists(vis_path):
            wandb.log({"denoising_samples": wandb.Image(vis_path)})

    # Save results summary
    results = {
        # Model metadata
        "model": {
            "name": "U-Net",
            "type": "deep_learning",
            "architecture": "U-Net",
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        # Dataset metadata
        "dataset": {
            "name": "CIFAR-10",
            "type": "cifar10",
            "num_train": len(train_loader.dataset),
            "num_test": len(test_loader.dataset),
        },
        # Training configuration
        "training": {
            "noise_type": args.noise_type,
            "noise_param": args.noise_param,
            "optimizer": args.optimizer,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "bilinear": args.bilinear,
        },
        # Results
        "results": {
            "metrics_noisy": {k: float(v) for k, v in metrics_noisy.items()},
            "metrics_denoised": {k: float(v) for k, v in metrics_denoised.items()},
            "final_train_loss": float(train_losses[-1]),
            "final_test_loss": float(test_losses[-1]),
            "best_test_loss": float(trainer.best_loss),
        },
    }

    results_path = os.path.join(
        args.results_dir, f"{args.noise_type}_{args.optimizer}_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {results_path}")
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70 + "\n")

    # Finish wandb run
    if use_wandb:
        wandb.finish()


def evaluate_saved_model(args):
    """Evaluate a saved model"""
    print("\n" + "=" * 70)
    print("EVALUATING SAVED MODEL")
    print("=" * 70)

    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"\nDevice: {device}")

    # Load data
    print(f"\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        noise_type=args.noise_type,
        noise_param=args.noise_param,
        num_workers=args.num_workers,
    )

    # Create model
    model = UNet(n_channels=3, n_classes=3, dropout_rate=args.dropout)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, _, epoch, train_losses, test_losses = load_checkpoint(
        model, args.checkpoint, device
    )
    print(f"Loaded model from epoch {epoch}")

    # Evaluate
    print("\nEvaluating...")
    avg_psnr_noisy, avg_psnr_denoised = evaluate_model(model, test_loader, device)

    print(f"\nResults:")
    print(f"  Average Noisy PSNR: {avg_psnr_noisy:.2f} dB")
    print(f"  Average Denoised PSNR: {avg_psnr_denoised:.2f} dB")
    print(f"  Improvement: +{avg_psnr_denoised - avg_psnr_noisy:.2f} dB")

    # Visualize
    vis_path = os.path.join(args.results_dir, "evaluation_samples.png")
    visualize_denoising(model, test_loader, device, num_samples=5, save_path=vis_path)

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="U-Net Image Denoising")

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Mode: train or eval",
    )

    # Model parameters
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2)"
    )
    parser.add_argument(
        "--bilinear", action="store_true", help="Use bilinear upsampling"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "rmsprop", "sgd"],
        help="Optimizer (default: adam)",
    )

    # Noise parameters
    parser.add_argument(
        "--noise-type",
        type=str,
        default="gaussian",
        choices=["gaussian", "poisson", "salt_pepper"],
        help="Type of noise (default: gaussian)",
    )
    parser.add_argument(
        "--noise-param",
        type=float,
        default=0.05,
        help="Noise parameter (default: 0.05)",
    )

    # Data parameters
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loading workers (default: 2)",
    )

    # Paths
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./models",
        help="Directory to save models (default: ./models)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory to save results (default: ./results)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint for evaluation"
    )

    # Device
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU usage even if GPU is available"
    )

    # Weights & Biases parameters
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=WANDB_CONFIG.get("enabled", True),
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=WANDB_CONFIG.get("project", "image-denoising-unet"),
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=WANDB_CONFIG.get("entity", None),
        help="Wandb entity (username or team)",
    )
    parser.add_argument(
        "--wandb-log-interval",
        type=int,
        default=WANDB_CONFIG.get("log_interval", 1),
        help="Log metrics every N batches (default: 1)",
    )
    parser.add_argument(
        "--wandb-log-images",
        action="store_true",
        default=WANDB_CONFIG.get("log_images", True),
        help="Log sample images to wandb",
    )
    parser.add_argument(
        "--wandb-log-model",
        action="store_true",
        default=WANDB_CONFIG.get("log_model", True),
        help="Save model checkpoints to wandb",
    )
    parser.add_argument(
        "--wandb-save-code",
        action="store_true",
        default=WANDB_CONFIG.get("save_code", True),
        help="Save code to wandb",
    )

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Run
    if args.mode == "train":
        train_model(args)
    elif args.mode == "eval":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for evaluation mode")
        evaluate_saved_model(args)


if __name__ == "__main__":
    main()
