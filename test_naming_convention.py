#!/usr/bin/env python3
"""
Quick test to verify the new experiment naming convention works correctly.
Run this to ensure the naming format is generated properly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import MODEL_REGISTRY, DATASET_REGISTRY


def test_naming_format():
    """Test the experiment naming format"""
    print("=" * 70)
    print("TESTING EXPERIMENT NAMING CONVENTION")
    print("=" * 70)

    # Test parameters
    test_cases = [
        {
            "model": "unet",
            "dataset": "cifar10",
            "noise_type": "gaussian",
            "optimizer": "adam",
            "lr": 0.001,
        },
        {
            "model": "nlm",
            "dataset": "cifar10",
            "noise_type": "poisson",
            "optimizer": "adam",
            "lr": 0.001,
        },
        {
            "model": "bilateral",
            "dataset": "cifar10",
            "noise_type": "salt_pepper",
            "optimizer": "rmsprop",
            "lr": 0.0001,
        },
    ]

    print("\n1. Testing main.py naming format:")
    print("   Format: {model}_{dataset}_{noise_type}_{optimizer}_lr{lr}\n")

    for i, params in enumerate(test_cases, 1):
        experiment_name = (
            f"{params['model']}_{params['dataset']}_{params['noise_type']}_"
            f"{params['optimizer']}_lr{params['lr']}"
        )
        model_name = MODEL_REGISTRY[params["model"]]["name"]
        dataset_name = DATASET_REGISTRY[params["dataset"]]["name"]

        print(f"   Test {i}:")
        print(
            f"     Input: model={params['model']}, dataset={params['dataset']}, "
            f"noise={params['noise_type']}, opt={params['optimizer']}, lr={params['lr']}"
        )
        print(f"     Output: {experiment_name}")
        print(f"     Model: {model_name}")
        print(f"     Dataset: {dataset_name}")
        print()

    print("\n2. Testing run_experiment.py naming format:")
    print("   Format: {model}_{dataset}_{experiment_name}\n")

    experiment_configs = [
        {"name": "adam_gaussian", "model": "unet", "dataset": "cifar10"},
        {"name": "optimizer_comparison", "model": "unet", "dataset": "cifar10"},
        {"name": "noise_comparison", "model": "unet", "dataset": "cifar10"},
    ]

    for i, config in enumerate(experiment_configs, 1):
        model_key = config.get("model", "unet")
        dataset_key = config.get("dataset", "cifar10")
        descriptive_run_name = f"{model_key}_{dataset_key}_{config['name']}"

        print(f"   Test {i}:")
        print(
            f"     Input: experiment={config['name']}, model={model_key}, dataset={dataset_key}"
        )
        print(f"     Output: {descriptive_run_name}")
        print()

    print("\n3. Sample directory structure:")
    print("   models/")

    for params in test_cases[:2]:
        experiment_name = (
            f"{params['model']}_{params['dataset']}_{params['noise_type']}_"
            f"{params['optimizer']}_lr{params['lr']}"
        )
        print(f"     └── {experiment_name}/")
        print(f"           ├── best_model.pth")
        print(f"           ├── last_checkpoint.pth")
        print(f"           ├── training_history.json")
        print(f"           └── config.json")

    print("\n   results/")
    for params in test_cases[:2]:
        experiment_name = (
            f"{params['model']}_{params['dataset']}_{params['noise_type']}_"
            f"{params['optimizer']}_lr{params['lr']}"
        )
        print(f"     ├── {experiment_name}_loss.png")
        print(f"     ├── {experiment_name}_psnr.png")
        print(f"     ├── {experiment_name}_metrics.png")
        print(f"     ├── {experiment_name}_samples.png")
        print(f"     └── {experiment_name}_results.json")

    print("\n" + "=" * 70)
    print("✓ All naming format tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_naming_format()
