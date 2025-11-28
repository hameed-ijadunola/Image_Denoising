"""
Main script for training and evaluating image denoising models
"""

import torch
import argparse
import os
import json
from torch.utils.data import DataLoader
from src.registry import get_model_from_registry, get_dataset_from_registry
from src.train import Trainer, load_checkpoint
from src.utils import (
    evaluate_model,
    visualize_denoising,
    plot_training_history,
    plot_psnr_comparison,
    plot_metrics_comparison,
)
from config import WANDB_CONFIG, MODEL_REGISTRY, DATASET_REGISTRY
import wandb


def train_model(args):
    """Train the denoising model"""
    print("\n" + "=" * 70)
    print("IMAGE DENOISING")
    print("=" * 70)

    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"\nDevice: {device}")

    # Get model and dataset configuration from registry
    assert args.model in MODEL_REGISTRY, (
        f"Model '{args.model}' not found in registry. "
        f"Available: {', '.join(MODEL_REGISTRY.keys())}"
    )
    assert args.dataset in DATASET_REGISTRY, (
        f"Dataset '{args.dataset}' not found in registry. "
        f"Available: {', '.join(DATASET_REGISTRY.keys())}"
    )

    model_config = MODEL_REGISTRY[args.model]
    dataset_config = DATASET_REGISTRY[args.dataset]

    # Setup wandb
    use_wandb = args.use_wandb
    if use_wandb:
        # Create descriptive experiment name: model_dataset_noise_optimizer_lr
        run_name = f"{args.model}_{args.dataset}_{args.noise_type}_{args.optimizer}_lr{args.lr}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                # Training hyperparameters
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "optimizer": args.optimizer,
                "noise_type": args.noise_type,
                "noise_param": args.noise_param,
                "dropout": args.dropout,
                "bilinear": args.bilinear,
                "num_workers": args.num_workers,
                # Model metadata (from registry)
                "model_key": args.model,
                "model_name": model_config["name"],
                "model_type": model_config["type"],
                "model_architecture": model_config.get("class", "Unknown"),
                # Dataset metadata (from registry)
                "dataset_key": args.dataset,
                "dataset_name": dataset_config["name"],
                "dataset_type": args.dataset,
            },
            save_code=args.wandb_save_code,
        )
        print("\n✓ Weights & Biases initialized")
        print(f"  Project: {args.wandb_project}")
        print(f"  Run: {run_name}")

    # Load data
    print(f"\nLoading {dataset_config['name']} dataset...")
    print(f"  Noise type: {args.noise_type}")
    print(f"  Noise parameter: {args.noise_param}")
    print(f"  Batch size: {args.batch_size}")

    train_dataset = get_dataset_from_registry(
        args.dataset,
        DATASET_REGISTRY,
        train=True,
        noise_type=args.noise_type,
        noise_param=args.noise_param,
    )
    test_dataset = get_dataset_from_registry(
        args.dataset,
        DATASET_REGISTRY,
        train=False,
        noise_type=args.noise_type,
        noise_param=args.noise_param,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")

    # Create model
    print(f"\nCreating {model_config['name']} model...")

    # Prepare model parameters - merge defaults with CLI args
    model_params = model_config.get("default_params", {}).copy()

    # Override with command-line arguments if applicable
    if hasattr(args, "dropout") and args.dropout is not None:
        model_params["dropout_rate"] = args.dropout
    if hasattr(args, "bilinear"):
        model_params["bilinear"] = args.bilinear

    model = get_model_from_registry(args.model, MODEL_REGISTRY, **model_params)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")

    # Create save directory with descriptive name: model_dataset_noise_optimizer_lr
    experiment_name = (
        f"{args.model}_{args.dataset}_{args.noise_type}_{args.optimizer}_lr{args.lr}"
    )
    save_dir = os.path.join(args.save_dir, experiment_name)

    # Prepare metadata for tracking (from registry)
    model_metadata = {
        "name": model_config["name"],
        "type": model_config["type"],
        "architecture": model_config.get("class", "Unknown"),
        "parameters": num_params,
        "paper": model_config.get("paper", "N/A"),
    }

    dataset_metadata = {
        "name": dataset_config["name"],
        "type": args.dataset,
        "num_train": len(train_loader.dataset),
        "num_test": len(test_loader.dataset),
        "image_size": dataset_config.get("image_size", "Unknown"),
        "channels": dataset_config.get("channels", 3),
    }

    # Check if model requires training
    requires_training = model_config.get("requires_training", True)

    if requires_training:
        # Deep learning model - needs training
        # Create trainer
        wandb_config = {
            "log_interval": args.wandb_log_interval,
            "log_images": args.wandb_log_images,
            "log_model": args.wandb_log_model,
        }

        # Check if model requires custom trainer (e.g., R3L)
        uses_custom_trainer = model_config.get("custom_trainer", False)

        if uses_custom_trainer and args.model == "r3l":
            # Use R3L-specific trainer
            from src.r3l_trainer import R3LTrainer

            trainer = R3LTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                device=device,
                learning_rate_policy=args.lr,
                learning_rate_value=args.lr * 10,  # Value network learns faster
                num_epochs=args.epochs,
                save_dir=save_dir,
                use_wandb=use_wandb,
            )

            # Train with R3L trainer
            history = trainer.train()

            # Extract losses for compatibility with existing plotting
            train_losses = history.get("train_policy_loss", [])
            test_losses = history.get("val_psnr", [])
        else:
            # Use standard trainer
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
                model_metadata=model_metadata,
                dataset_metadata=dataset_metadata,
            )

            # Train
            train_losses, test_losses = trainer.train(num_epochs=args.epochs)
    else:
        # Traditional method - no training needed
        print("\n" + "-" * 70)
        print("TRADITIONAL METHOD - NO TRAINING REQUIRED")
        print("-" * 70)
        print(f"\n{model_config['name']} is a traditional denoising method.")
        print("Skipping training phase and proceeding directly to evaluation.")
        print("\nNote: Traditional methods have fixed parameters and do not")
        print("require gradient-based optimization.\n")

        # No training losses for traditional methods
        train_losses = []
        test_losses = []

    # Plot training history (only for models that were trained)
    if requires_training and train_losses:
        plot_path = os.path.join(args.results_dir, f"{experiment_name}_loss.png")
        plot_training_history(train_losses, test_losses, save_path=plot_path)
    else:
        plot_path = None

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
    psnr_plot_path = os.path.join(args.results_dir, f"{experiment_name}_psnr.png")
    if "psnr" in metrics_noisy and "psnr" in metrics_denoised:
        plot_psnr_comparison(
            metrics_noisy["psnr"], metrics_denoised["psnr"], save_path=psnr_plot_path
        )

    # Plot all metrics comparison
    metrics_plot_path = os.path.join(args.results_dir, f"{experiment_name}_metrics.png")
    plot_metrics_comparison(
        metrics_noisy, metrics_denoised, save_path=metrics_plot_path
    )

    # Visualize denoising
    vis_path = os.path.join(args.results_dir, f"{experiment_name}_samples.png")
    visualize_denoising(model, test_loader, device, num_samples=5, save_path=vis_path)

    # Log images to wandb
    if use_wandb and args.wandb_log_images:
        # Log training history plot (only if trained)
        if plot_path and os.path.exists(plot_path):
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
        # Model metadata (from registry)
        "model": {
            "key": args.model,
            "name": model_config["name"],
            "type": model_config["type"],
            "architecture": model_config.get("class", "Unknown"),
            "parameters": sum(p.numel() for p in model.parameters()),
            "paper": model_config.get("paper", "N/A"),
        },
        # Dataset metadata (from registry)
        "dataset": {
            "key": args.dataset,
            "name": dataset_config["name"],
            "type": args.dataset,
            "num_train": len(train_loader.dataset),
            "num_test": len(test_loader.dataset),
            "image_size": str(dataset_config.get("image_size", "Unknown")),
            "channels": dataset_config.get("channels", 3),
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
            "final_train_loss": float(train_losses[-1]) if train_losses else None,
            "final_test_loss": float(test_losses[-1]) if test_losses else None,
            "best_test_loss": float(
                getattr(trainer, "best_loss", getattr(trainer, "best_val_psnr", None))
            )
            if requires_training
            else None,
        },
    }

    results_path = os.path.join(args.results_dir, f"{experiment_name}_results.json")
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

    # Get model and dataset configuration from registry
    assert args.model in MODEL_REGISTRY, (
        f"Model '{args.model}' not found in registry. "
        f"Available: {', '.join(MODEL_REGISTRY.keys())}"
    )
    assert args.dataset in DATASET_REGISTRY, (
        f"Dataset '{args.dataset}' not found in registry. "
        f"Available: {', '.join(DATASET_REGISTRY.keys())}"
    )

    model_config = MODEL_REGISTRY[args.model]
    dataset_config = DATASET_REGISTRY[args.dataset]

    # Load data
    print(f"\nLoading {dataset_config['name']} dataset...")

    train_dataset = get_dataset_from_registry(
        args.dataset,
        DATASET_REGISTRY,
        train=True,
        noise_type=args.noise_type,
        noise_param=args.noise_param,
    )
    test_dataset = get_dataset_from_registry(
        args.dataset,
        DATASET_REGISTRY,
        train=False,
        noise_type=args.noise_type,
        noise_param=args.noise_param,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Create model
    print(f"\nCreating {model_config['name']} model...")

    # Prepare model parameters
    model_params = model_config.get("default_params", {}).copy()
    if hasattr(args, "dropout") and args.dropout is not None:
        model_params["dropout_rate"] = args.dropout

    model = get_model_from_registry(args.model, MODEL_REGISTRY, **model_params)

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
    parser = argparse.ArgumentParser(
        description="Image Denoising with Registry-based Models"
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Mode: train or eval",
    )

    # Model selection (from registry)
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        help=f"Model to use from registry (available: {', '.join(MODEL_REGISTRY.keys())})",
    )

    # Dataset selection (from registry)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help=f"Dataset to use from registry (available: {', '.join(DATASET_REGISTRY.keys())})",
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
