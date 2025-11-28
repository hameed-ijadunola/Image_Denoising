"""
Configuration file for experiments
Modify these settings and run: python run_experiment.py
"""

# ============================================================================
# MODEL REGISTRY
# ============================================================================
# Register all available denoising models and techniques here
# Add new models/techniques as you implement them

MODEL_REGISTRY = {
    "unet": {
        "name": "U-Net",
        "type": "deep_learning",
        "class": "UNet",  # Class name in src/unet.py
        "module": "src.unet",  # Import path
        "description": "U-Net Convolutional Network for Image Denoising",
        "paper": "Ronneberger et al., 2015",
        "requires_training": True,
        "default_params": {
            "n_channels": 3,
            "n_classes": 3,
            "bilinear": False,
            "dropout_rate": 0.2,
        },
    },
    "r3l": {
        "name": "R3L",
        "type": "deep_learning",
        "class": "R3L",
        "module": "src.r3l",
        "description": "Residual Recovery using Reinforcement Learning for Image Denoising",
        "paper": "Zhang et al., 2021 - Connecting Deep Reinforcement Learning to Recurrent Neural Networks",
        "requires_training": True,
        "custom_trainer": True,  # Uses R3LTrainer instead of standard Trainer
        "default_params": {
            "in_channels": 3,
            "num_stages": 5,
            "action_range": (-13, 13),
            "gamma": 0.95,
        },
    },
    # Example: Add more deep learning models
    # "dncnn": {
    #     "name": "DnCNN",
    #     "type": "deep_learning",
    #     "class": "DnCNN",
    #     "module": "src.dncnn",
    #     "description": "Denoising Convolutional Neural Network",
    #     "paper": "Zhang et al., 2017",
    #     "requires_training": True,
    #     "default_params": {...},
    # },
    # Traditional methods (no training needed)
    "nlm": {
        "name": "Non-Local Means",
        "type": "traditional",
        "class": "NonLocalMeans",
        "module": "src.traditional.nlm",
        "description": "Non-Local Means denoising filter",
        "paper": "Buades et al., 2005",
        "requires_training": False,
        "default_params": {
            "h": 10,
            "template_window_size": 7,
            "search_window_size": 21,
        },
    },
    "bilateral": {
        "name": "Bilateral Filter",
        "type": "traditional",
        "class": "BilateralFilter",
        "module": "src.traditional.bilateral",
        "description": "Bilateral filtering for edge-preserving denoising",
        "paper": "Tomasi and Manduchi, 1998",
        "requires_training": False,
        "default_params": {
            "d": 9,
            "sigma_color": 75,
            "sigma_space": 75,
        },
    },
    "wavelet": {
        "name": "Wavelet Denoising",
        "type": "traditional",
        "class": "WaveletDenoising",
        "module": "src.traditional.wavelet",
        "description": "Wavelet-based denoising using soft thresholding",
        "paper": "Donoho, 1995",
        "requires_training": False,
        "default_params": {
            "wavelet": "db1",
            "mode": "soft",
            "level": None,
        },
    },
}

# ============================================================================
# DATASET REGISTRY
# ============================================================================
# Register all available datasets here

DATASET_REGISTRY = {
    "bsd300": {
        "name": "BSD300",
        "class": "BSD300Dataset",
        "module": "src.bsd300_dataset",
        "description": "BSD300 grayscale images for denoising",
        "image_size": None,  # Variable sizes
        "channels": 1,
        "num_train": 200,  # BSD300 train images
        "default_params": {
            "root": "./data/BSD300/images/train",
            "noise_type": "gaussian",
            "noise_param": 0.05,
            "download": False,
        },
    },
    "cifar10": {
        "name": "CIFAR-10",
        "class": "NoisyCIFAR10",
        "module": "src.dataset",
        "description": "CIFAR-10 dataset (32x32 color images, 10 classes)",
        "image_size": (32, 32),
        "channels": 3,
        "num_classes": 10,
        "num_train": 50000,
        "num_test": 10000,
        "default_params": {
            "root": "./data",
            "noise_type": "gaussian",
            "noise_param": 0.05,
            "download": True,
        },
    },
    "bsd68": {
        "name": "BSD68",
        "class": "BSD68Dataset",
        "module": "src.bsd68_dataset",
        "description": "68 grayscale test images from Berkeley Segmentation Dataset",
        "image_size": None,  # Variable sizes
        "channels": 1,
        "num_test": 68,
        "default_params": {
            "root": "./data/BSD68",
            "noise_type": "gaussian",
            "noise_param": 0.05,
            "download": False,
        },
    },
    # "set12": {
    #     "name": "Set12",
    #     "class": "Set12Dataset",
    #     "module": "src.datasets.set12",
    #     "description": "12 widely used test images",
    #     "image_size": None,
    #     "channels": 1,
    #     "num_test": 12,
    #     "default_params": {...},
    # },
}

# ============================================================================
# NOISE TYPE REGISTRY
# ============================================================================
# Available noise types and their parameters

NOISE_REGISTRY = {
    "gaussian": {
        "name": "Gaussian Noise",
        "param_name": "sigma",
        "param_description": "Standard deviation of noise",
        "default_param": 0.05,
        "param_range": (0.01, 0.15),
    },
    "poisson": {
        "name": "Poisson Noise",
        "param_name": "lambda",
        "param_description": "Poisson rate parameter",
        "default_param": 1.0,
        "param_range": (0.5, 2.0),
    },
    "salt_pepper": {
        "name": "Salt & Pepper Noise",
        "param_name": "probability",
        "param_description": "Probability of corrupted pixels",
        "default_param": 0.05,
        "param_range": (0.01, 0.1),
    },
}

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================
EXPERIMENTS = {
    # BSD68 test experiments for R3L replication (models trained on BSD300)
    "bsd68_r3l_test_sigma25": [
        {
            "name": "bsd68_r3l_test_sigma15",
            "model": "r3l",
            "dataset": "bsd68",
            "epochs": 0,
            "noise_type": "gaussian",
            "noise_param": 0.15,
            "batch_size": 1,
        },
        {
            "name": "bsd68_r3l_test_sigma20",
            "model": "r3l",
            "dataset": "bsd68",
            "epochs": 0,
            "noise_type": "gaussian",
            "noise_param": 0.20,
            "batch_size": 1,
        },
        {
            "name": "bsd68_r3l_test_sigma25",
            "model": "r3l",
            "dataset": "bsd68",
            "epochs": 0,
            "noise_type": "gaussian",
            "noise_param": 0.25,
            "batch_size": 1,
        },
        {
            "name": "bsd68_r3l_test_sigma30",
            "model": "r3l",
            "dataset": "bsd68",
            "epochs": 0,
            "noise_type": "gaussian",
            "noise_param": 0.30,
            "batch_size": 1,
        },
        {
            "name": "bsd68_r3l_test_sigma35",
            "model": "r3l",
            "dataset": "bsd68",
            "epochs": 0,
            "noise_type": "gaussian",
            "noise_param": 0.35,
            "batch_size": 1,
        },
    ],
    "bsd68_r3l_test_sigma35": [
        {
            "name": "bsd68_r3l_test_sigma25",
            "model": "r3l",
            "dataset": "bsd68",
            "epochs": 0,
            "noise_type": "gaussian",
            "noise_param": 0.25,
            "batch_size": 1,
        },
        {
            "name": "bsd68_r3l_test_sigma30",
            "model": "r3l",
            "dataset": "bsd68",
            "epochs": 0,
            "noise_type": "gaussian",
            "noise_param": 0.30,
            "batch_size": 1,
        },
        {
            "name": "bsd68_r3l_test_sigma35",
            "model": "r3l",
            "dataset": "bsd68",
            "epochs": 0,
            "noise_type": "gaussian",
            "noise_param": 0.35,
            "batch_size": 1,
        },
        {
            "name": "bsd68_r3l_test_sigma40",
            "model": "r3l",
            "dataset": "bsd68",
            "epochs": 0,
            "noise_type": "gaussian",
            "noise_param": 0.40,
            "batch_size": 1,
        },
        {
            "name": "bsd68_r3l_test_sigma45",
            "model": "r3l",
            "dataset": "bsd68",
            "epochs": 0,
            "noise_type": "gaussian",
            "noise_param": 0.45,
            "batch_size": 1,
        },
    ],
    # BSD300 training experiments for R3L replication
    "bsd300_r3l_train": [
        {
            "name": "bsd300_r3l_sigma25_train",
            "model": "r3l",
            "dataset": "bsd300",
            "epochs": 50,
            "optimizer": "adam",
            "lr": 0.001,
            "noise_type": "gaussian",
            "noise_param": 0.25,
            "batch_size": 16,
        },
        {
            "name": "bsd300_r3l_sigma35_train",
            "model": "r3l",
            "dataset": "bsd300",
            "epochs": 50,
            "optimizer": "adam",
            "lr": 0.001,
            "noise_type": "gaussian",
            "noise_param": 0.35,
            "batch_size": 16,
        },
    ],
    # Optimizer comparison (as in the paper)
    "optimizer_comparison": [
        {
            "name": "adam_gaussian",
            "epochs": 5,
            "optimizer": "adam",
            "lr": 0.001,
            "noise_type": "gaussian",
            "noise_param": 0.05,
            "batch_size": 16,
        },
        {
            "name": "rmsprop_gaussian",
            "epochs": 5,
            "optimizer": "rmsprop",
            "lr": 0.001,
            "noise_type": "gaussian",
            "noise_param": 0.05,
            "batch_size": 16,
        },
        {
            "name": "sgd_gaussian",
            "epochs": 5,
            "optimizer": "sgd",
            "lr": 0.001,
            "noise_type": "gaussian",
            "noise_param": 0.05,
            "batch_size": 16,
        },
    ],
    # Noise type comparison
    "noise_comparison": [
        {
            "name": "gaussian_noise",
            "epochs": 5,
            "optimizer": "adam",
            "lr": 0.001,
            "noise_type": "gaussian",
            "noise_param": 0.05,
            "batch_size": 16,
        },
        {
            "name": "poisson_noise",
            "epochs": 5,
            "optimizer": "adam",
            "lr": 0.001,
            "noise_type": "poisson",
            "noise_param": 1.0,
            "batch_size": 16,
        },
        {
            "name": "salt_pepper_noise",
            "epochs": 5,
            "optimizer": "adam",
            "lr": 0.001,
            "noise_type": "salt_pepper",
            "noise_param": 0.05,
            "batch_size": 16,
        },
    ],
    # BSD68 experiments for R3L replication
    "bsd68_r3l": [
        {
            "name": "bsd68_r3l_sigma15",
            "model": "r3l",
            "dataset": "bsd68",
            "epochs": 5,
            "optimizer": "adam",
            "lr": 0.001,
            "noise_type": "gaussian",
            "noise_param": 0.15,
            "batch_size": 16,
        },
        {
            "name": "bsd68_r3l_sigma25",
            "model": "r3l",
            "dataset": "bsd68",
            "epochs": 5,
            "optimizer": "adam",
            "lr": 0.001,
            "noise_type": "gaussian",
            "noise_param": 0.25,
            "batch_size": 16,
        },
        {
            "name": "bsd68_r3l_sigma50",
            "model": "r3l",
            "dataset": "bsd68",
            "epochs": 5,
            "optimizer": "adam",
            "lr": 0.001,
            "noise_type": "gaussian",
            "noise_param": 0.50,
            "batch_size": 16,
        },
    ],
    # Quick test (1 epoch)
    "quick_test": [
        {
            "name": "quick_test",
            "epochs": 1,
            "optimizer": "adam",
            "lr": 0.001,
            "noise_type": "gaussian",
            "noise_param": 0.05,
            "batch_size": 16,
        },
    ],
}

# Default training configuration
DEFAULT_CONFIG = {
    # Model configuration
    "model": "unet",  # Key from MODEL_REGISTRY
    # Dataset configuration
    "dataset": "cifar10",  # Key from DATASET_REGISTRY
    # Training parameters
    "epochs": 5,
    "batch_size": 16,
    "lr": 0.001,
    "optimizer": "adam",
    # Noise configuration
    "noise_type": "gaussian",
    "noise_param": 0.05,
    # Model-specific parameters
    "dropout": 0.2,
    "bilinear": False,
    # System parameters
    "num_workers": 2,
}

# Weights & Biases configuration
WANDB_CONFIG = {
    "enabled": True,  # Set to False to disable wandb logging
    "project": "rl_image_denoising",  # wandb project name
    "entity": "image_denoising",  # wandb entity (username or team), None for default
    "log_interval": 1,  # Log metrics every N batches (1 = every batch)
    "log_images": True,  # Log sample denoised images
    "log_gradients": False,  # Log gradient histograms (can be slow)
    "log_model": True,  # Save model checkpoints to wandb
    "save_code": True,  # Save code to wandb
}
