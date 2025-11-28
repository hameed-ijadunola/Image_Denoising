"""
Run experiments defined in config.py
Usage: python run_experiment.py <experiment_name>
Example: python run_experiment.py optimizer_comparison
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
from unet import UNet
from dataset import get_dataloaders
from train import Trainer
from utils import (
    evaluate_model,
    plot_training_history,
    plot_psnr_comparison,
    visualize_denoising,
)
from config import EXPERIMENTS, DEFAULT_CONFIG, WANDB_CONFIG

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def run_single_experiment(config, experiment_name, use_wandb=False):
    """Run a single experiment with given configuration"""
    print("\n" + "=" * 80)
    print(f"RUNNING EXPERIMENT: {experiment_name}")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 80 + "\n")

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=WANDB_CONFIG.get("project", "image-denoising-unet"),
            entity=WANDB_CONFIG.get("entity", None),
            name=experiment_name,
            config=config,
            save_code=WANDB_CONFIG.get("save_code", True),
            reinit=True,  # Allow multiple runs in same script
        )
        print("✓ Weights & Biases initialized\n")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_dataloaders(
        batch_size=config.get("batch_size", DEFAULT_CONFIG["batch_size"]),
        noise_type=config.get("noise_type", DEFAULT_CONFIG["noise_type"]),
        noise_param=config.get("noise_param", DEFAULT_CONFIG["noise_param"]),
        num_workers=config.get("num_workers", DEFAULT_CONFIG["num_workers"]),
    )

    # Create model
    print("Creating U-Net model...")
    model = UNet(
        n_channels=3,
        n_classes=3,
        bilinear=config.get("bilinear", DEFAULT_CONFIG["bilinear"]),
        dropout_rate=config.get("dropout", DEFAULT_CONFIG["dropout"]),
    )

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("./models", f"{experiment_name}_{timestamp}")
    results_dir = os.path.join("./results", f"{experiment_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Create trainer
    wandb_config = {
        "log_interval": WANDB_CONFIG.get("log_interval", 1),
        "log_images": WANDB_CONFIG.get("log_images", True),
        "log_model": WANDB_CONFIG.get("log_model", True),
    }

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_name=config.get("optimizer", DEFAULT_CONFIG["optimizer"]),
        lr=config.get("lr", DEFAULT_CONFIG["lr"]),
        save_dir=save_dir,
        use_wandb=use_wandb and WANDB_AVAILABLE,
        wandb_config=wandb_config,
    )

    # Train
    train_losses, test_losses = trainer.train(
        num_epochs=config.get("epochs", DEFAULT_CONFIG["epochs"])
    )

    # Evaluate
    print("\nEvaluating model...")
    avg_psnr_noisy, avg_psnr_denoised = evaluate_model(model, test_loader, device)

    # Log to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log(
            {
                "final_psnr_noisy": avg_psnr_noisy,
                "final_psnr_denoised": avg_psnr_denoised,
                "psnr_improvement": avg_psnr_denoised - avg_psnr_noisy,
            }
        )

    # Save results
    results = {
        "experiment_name": experiment_name,
        "config": config,
        "avg_psnr_noisy": float(avg_psnr_noisy),
        "avg_psnr_denoised": float(avg_psnr_denoised),
        "psnr_improvement": float(avg_psnr_denoised - avg_psnr_noisy),
        "train_losses": [float(x) for x in train_losses],
        "test_losses": [float(x) for x in test_losses],
        "final_train_loss": float(train_losses[-1]),
        "final_test_loss": float(test_losses[-1]),
        "best_test_loss": float(trainer.best_loss),
        "timestamp": timestamp,
    }

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # Generate plots
    plot_training_history(
        train_losses, test_losses, save_path=os.path.join(results_dir, "loss_curve.png")
    )

    plot_psnr_comparison(
        avg_psnr_noisy,
        avg_psnr_denoised,
        save_path=os.path.join(results_dir, "psnr_comparison.png"),
    )

    visualize_denoising(
        model,
        test_loader,
        device,
        num_samples=5,
        save_path=os.path.join(results_dir, "denoising_samples.png"),
    )

    # Log visualizations to wandb
    if use_wandb and WANDB_AVAILABLE and WANDB_CONFIG.get("log_images", True):
        wandb.log(
            {
                "training_loss_curve": wandb.Image(
                    os.path.join(results_dir, "loss_curve.png")
                ),
                "psnr_comparison": wandb.Image(
                    os.path.join(results_dir, "psnr_comparison.png")
                ),
                "denoising_samples": wandb.Image(
                    os.path.join(results_dir, "denoising_samples.png")
                ),
            }
        )

    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS:")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Noisy PSNR: {avg_psnr_noisy:.2f} dB")
    print(f"Denoised PSNR: {avg_psnr_denoised:.2f} dB")
    print(f"Improvement: +{avg_psnr_denoised - avg_psnr_noisy:.2f} dB")
    print(f"Final Train Loss: {train_losses[-1]:.5f}")
    print(f"Final Test Loss: {test_losses[-1]:.5f}")
    print(f"\nResults saved to: {results_dir}")
    print(f"Model saved to: {save_dir}")
    print("=" * 80 + "\n")

    # Finish wandb run
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    return results


def run_experiment_suite(suite_name, use_wandb=False):
    """Run a suite of experiments"""
    if suite_name not in EXPERIMENTS:
        print(f"Error: Unknown experiment suite '{suite_name}'")
        print(f"Available suites: {list(EXPERIMENTS.keys())}")
        return

    experiments = EXPERIMENTS[suite_name]
    print("\n" + "=" * 80)
    print(f"RUNNING EXPERIMENT SUITE: {suite_name}")
    print(f"Total experiments: {len(experiments)}")
    print(
        f"Wandb logging: {'Enabled' if use_wandb and WANDB_AVAILABLE else 'Disabled'}"
    )
    print("=" * 80)

    all_results = []
    for i, config in enumerate(experiments, 1):
        exp_name = config.get("name", f"{suite_name}_{i}")
        print(f"\n[{i}/{len(experiments)}] Starting experiment: {exp_name}")

        try:
            results = run_single_experiment(config, exp_name, use_wandb=use_wandb)
            all_results.append(results)
        except Exception as e:
            print(f"Error in experiment {exp_name}: {str(e)}")
            continue

    # Save summary
    summary_path = os.path.join("./results", f"{suite_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n" + "=" * 80)
    print("SUITE COMPLETE!")
    print("=" * 80)
    print(f"Ran {len(all_results)}/{len(experiments)} experiments successfully")
    print(f"Summary saved to: {summary_path}")
    print("\nResults Summary:")
    print("-" * 80)
    for result in all_results:
        print(
            f"{result['experiment_name']:25s} | "
            f"PSNR: {result['avg_psnr_noisy']:.2f} → {result['avg_psnr_denoised']:.2f} dB "
            f"(+{result['psnr_improvement']:.2f})"
        )
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments from config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py optimizer_comparison
  python run_experiment.py noise_comparison
  python run_experiment.py quick_test

Available experiment suites:
  """
        + "\n  ".join(EXPERIMENTS.keys()),
    )

    parser.add_argument(
        "suite",
        type=str,
        nargs="?",
        default="quick_test",
        help="Name of experiment suite to run (default: quick_test)",
    )

    parser.add_argument(
        "--list", action="store_true", help="List available experiment suites"
    )

    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=WANDB_CONFIG.get("enabled", True),
        help="Enable Weights & Biases logging for experiments",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiment suites:")
        for suite_name, experiments in EXPERIMENTS.items():
            print(f"\n{suite_name}:")
            for exp in experiments:
                print(
                    f"  - {exp.get('name', 'unnamed')}: "
                    f"{exp.get('epochs', DEFAULT_CONFIG['epochs'])} epochs, "
                    f"{exp.get('optimizer', DEFAULT_CONFIG['optimizer'])} optimizer, "
                    f"{exp.get('noise_type', DEFAULT_CONFIG['noise_type'])} noise"
                )
        print()
        return

    run_experiment_suite(args.suite, use_wandb=args.use_wandb)


if __name__ == "__main__":
    main()
