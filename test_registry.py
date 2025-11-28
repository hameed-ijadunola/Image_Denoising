"""
Test script to verify the registry and tracking system
Run this to ensure everything is working correctly
"""

import sys
import torch
from config import MODEL_REGISTRY, DATASET_REGISTRY, NOISE_REGISTRY
from src.registry import (
    get_model_from_registry,
    get_dataset_from_registry,
    list_models,
    print_registry_info,
    get_experiment_metadata,
    validate_experiment_config,
)


def test_model_registry():
    """Test model registry functionality"""
    print("\n" + "=" * 70)
    print("TEST 1: Model Registry")
    print("=" * 70)

    try:
        # Get U-Net model
        model = get_model_from_registry("unet", MODEL_REGISTRY)
        print(f"✓ Created model: {model.name}")
        print(f"  Type: {model.model_type}")
        print(f"  Parameters: {model.count_parameters():,}")

        # Test model info
        info = model.get_info()
        print(f"  Model info keys: {list(info.keys())}")

        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        print(f"  Forward pass: {x.shape} -> {y.shape}")
        assert y.shape == x.shape, "Output shape mismatch!"

        print("✓ Model registry test PASSED\n")
        return True
    except Exception as e:
        print(f"✗ Model registry test FAILED: {e}\n")
        return False


def test_dataset_registry():
    """Test dataset registry functionality"""
    print("=" * 70)
    print("TEST 2: Dataset Registry")
    print("=" * 70)

    try:
        # Get CIFAR-10 dataset
        dataset = get_dataset_from_registry(
            "cifar10",
            DATASET_REGISTRY,
            train=True,
            noise_type="gaussian",
            noise_param=0.05,
        )
        print(f"✓ Created dataset: {dataset.name}")
        print(f"  Size: {len(dataset)}")
        print(f"  Shape: {dataset.get_sample_shape()}")

        # Test dataset info
        info = dataset.get_info()
        print(f"  Dataset info keys: {list(info.keys())}")

        # Test getting a sample
        noisy, clean = dataset[0]
        print(f"  Sample shapes: noisy={noisy.shape}, clean={clean.shape}")
        assert noisy.shape == clean.shape, "Image shape mismatch!"

        print("✓ Dataset registry test PASSED\n")
        return True
    except Exception as e:
        print(f"✗ Dataset registry test FAILED: {e}\n")
        return False


def test_experiment_metadata():
    """Test experiment metadata generation"""
    print("=" * 70)
    print("TEST 3: Experiment Metadata")
    print("=" * 70)

    try:
        metadata = get_experiment_metadata(
            model_key="unet",
            dataset_key="cifar10",
            model_registry=MODEL_REGISTRY,
            dataset_registry=DATASET_REGISTRY,
            epochs=50,
            batch_size=32,
            lr=0.001,
            optimizer="adam",
        )

        print("✓ Generated experiment metadata")
        print(f"  Model: {metadata['model']['name']}")
        print(f"  Dataset: {metadata['dataset']['name']}")
        print(f"  Training params: {list(metadata['training'].keys())}")

        print("✓ Metadata generation test PASSED\n")
        return True
    except Exception as e:
        print(f"✗ Metadata generation test FAILED: {e}\n")
        return False


def test_validation():
    """Test configuration validation"""
    print("=" * 70)
    print("TEST 4: Configuration Validation")
    print("=" * 70)

    try:
        # Test valid config
        valid_config = {"model": "unet", "dataset": "cifar10", "epochs": 50}
        validate_experiment_config(valid_config, MODEL_REGISTRY, DATASET_REGISTRY)
        print("✓ Valid config passed validation")

        # Test invalid model
        try:
            invalid_config = {"model": "nonexistent_model", "dataset": "cifar10"}
            validate_experiment_config(invalid_config, MODEL_REGISTRY, DATASET_REGISTRY)
            print("✗ Invalid model should have failed!")
            return False
        except ValueError:
            print("✓ Invalid model correctly rejected")

        # Test invalid dataset
        try:
            invalid_config = {"model": "unet", "dataset": "nonexistent_dataset"}
            validate_experiment_config(invalid_config, MODEL_REGISTRY, DATASET_REGISTRY)
            print("✗ Invalid dataset should have failed!")
            return False
        except ValueError:
            print("✓ Invalid dataset correctly rejected")

        print("✓ Validation test PASSED\n")
        return True
    except Exception as e:
        print(f"✗ Validation test FAILED: {e}\n")
        return False


def test_registry_listing():
    """Test registry listing functions"""
    print("=" * 70)
    print("TEST 5: Registry Listing")
    print("=" * 70)

    try:
        # List all models
        all_models = list_models(MODEL_REGISTRY)
        print(f"✓ Found {len(all_models)} models: {list(all_models.keys())}")

        # List deep learning models
        dl_models = list_models(MODEL_REGISTRY, model_type="deep_learning")
        print(f"✓ Found {len(dl_models)} deep learning models")

        # List traditional models
        trad_models = list_models(MODEL_REGISTRY, model_type="traditional")
        print(f"✓ Found {len(trad_models)} traditional models")

        print("✓ Registry listing test PASSED\n")
        return True
    except Exception as e:
        print(f"✗ Registry listing test FAILED: {e}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("REGISTRY AND TRACKING SYSTEM TESTS")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Model Registry", test_model_registry()))
    results.append(("Dataset Registry", test_dataset_registry()))
    results.append(("Experiment Metadata", test_experiment_metadata()))
    results.append(("Configuration Validation", test_validation()))
    results.append(("Registry Listing", test_registry_listing()))

    # Print summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests PASSED! The tracking system is ready to use.")
        print("\nNext steps:")
        print("  1. See TRACKING_SYSTEM.md for an overview")
        print("  2. See REGISTRY_GUIDE.md for detailed examples")
        print("  3. Start adding new models and datasets!")
    else:
        print("\n⚠️  Some tests FAILED. Please check the errors above.")
        sys.exit(1)

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
