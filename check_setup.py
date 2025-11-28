#!/usr/bin/env python3
"""
Check if the development environment is properly set up
Run this before training to verify everything works
"""

import sys
import os


def check_python_version():
    """Check Python version"""
    print("\n" + "=" * 60)
    print("Checking Python Version")
    print("=" * 60)
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher required!")
        return False
    else:
        print("✅ Python version OK")
        return True


def check_imports():
    """Check if all required packages can be imported"""
    print("\n" + "=" * 60)
    print("Checking Required Packages")
    print("=" * 60)

    packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "numpy": "NumPy",
        "matplotlib": "Matplotlib",
        "PIL": "Pillow",
        "tqdm": "tqdm",
        "skimage": "scikit-image",
    }

    all_ok = True
    for package, name in packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, "__version__", "unknown")
            print(f"✅ {name:20s} - version {version}")
        except ImportError:
            print(f"❌ {name:20s} - NOT INSTALLED")
            all_ok = False

    return all_ok


def check_cuda():
    """Check CUDA availability"""
    print("\n" + "=" * 60)
    print("Checking CUDA/GPU Support")
    print("=" * 60)

    try:
        import torch

        cuda_available = torch.cuda.is_available()

        if cuda_available:
            print(f"✅ CUDA is available")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU device: {torch.cuda.get_device_name(0)}")
            print(
                f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        else:
            print("ℹ️  CUDA not available (will use CPU)")
            print("   This is OK but training will be slower")

        return True
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False


def check_project_structure():
    """Check if project directories exist"""
    print("\n" + "=" * 60)
    print("Checking Project Structure")
    print("=" * 60)

    required_dirs = ["src", "models", "results", "docs"]
    required_files = [
        "main.py",
        "demo.py",
        "requirements.txt",
        "README.md",
        "src/unet.py",
        "src/dataset.py",
        "src/train.py",
        "src/utils.py",
    ]

    all_ok = True

    # Check directories
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"✅ Directory '{dir_name}/' exists")
        else:
            print(f"❌ Directory '{dir_name}/' missing")
            all_ok = False

    # Check files
    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"✅ File '{file_name}' exists")
        else:
            print(f"❌ File '{file_name}' missing")
            all_ok = False

    return all_ok


def test_model_creation():
    """Test if model can be created"""
    print("\n" + "=" * 60)
    print("Testing Model Creation")
    print("=" * 60)

    try:
        import torch

        sys.path.insert(0, "src")
        from unet import UNet

        model = UNet(n_channels=3, n_classes=3, dropout_rate=0.2)
        num_params = sum(p.numel() for p in model.parameters())

        print(f"✅ U-Net model created successfully")
        print(f"   Total parameters: {num_params:,}")

        # Test forward pass
        x = torch.randn(1, 3, 32, 32)
        output = model(x)

        if output.shape == (1, 3, 32, 32):
            print(f"✅ Forward pass successful")
            print(f"   Input shape:  {tuple(x.shape)}")
            print(f"   Output shape: {tuple(output.shape)}")
            return True
        else:
            print(f"❌ Output shape incorrect: {output.shape}")
            return False

    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dataset():
    """Test if dataset can be loaded"""
    print("\n" + "=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)

    try:
        sys.path.insert(0, "src")
        from dataset import NoisyCIFAR10

        print("Creating test dataset (may download CIFAR-10)...")
        dataset = NoisyCIFAR10(
            root="./data",
            train=True,
            noise_type="gaussian",
            noise_param=0.05,
            download=True,
        )

        print(f"✅ Dataset created successfully")
        print(f"   Number of samples: {len(dataset)}")

        # Test getting one sample
        noisy, clean = dataset[0]
        print(f"✅ Sample loaded successfully")
        print(f"   Noisy image shape: {tuple(noisy.shape)}")
        print(f"   Clean image shape: {tuple(clean.shape)}")

        return True

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_disk_space():
    """Check available disk space"""
    print("\n" + "=" * 60)
    print("Checking Disk Space")
    print("=" * 60)

    try:
        import shutil

        total, used, free = shutil.disk_usage(".")

        free_gb = free / (1024**3)
        print(f"Available disk space: {free_gb:.2f} GB")

        if free_gb < 5:
            print(f"⚠️  Low disk space (need at least 5GB)")
            return False
        else:
            print(f"✅ Sufficient disk space")
            return True

    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True  # Don't fail on this


def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("IMAGE DENOISING PROJECT - SETUP CHECK")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_imports),
        ("CUDA/GPU Support", check_cuda),
        ("Project Structure", check_project_structure),
        ("Disk Space", check_disk_space),
        ("Model Creation", test_model_creation),
        ("Dataset Loading", test_dataset),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error during {name} check: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} - {name}")

    print("=" * 60)
    print(f"Checks passed: {passed}/{total}")
    print("=" * 60)

    if passed == total:
        print("\n🎉 All checks passed! You're ready to train.")
        print("\nNext steps:")
        print("  1. Quick demo:  python demo.py")
        print("  2. Start training:  python main.py --mode train")
        print("  3. See README.md for more options")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  • Missing packages: pip install -r requirements.txt")
        print("  • Wrong directory: cd /path/to/Image_Denoising")
        print("  • See TROUBLESHOOTING.md for more help")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
