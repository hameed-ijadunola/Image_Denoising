"""
Configuration file for experiments
Modify these settings and run: python run_experiment.py
"""

# Experiment configurations
EXPERIMENTS = {
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
    "epochs": 5,
    "batch_size": 16,
    "lr": 0.001,
    "optimizer": "adam",
    "noise_type": "gaussian",
    "noise_param": 0.05,
    "dropout": 0.2,
    "num_workers": 2,
    "bilinear": False,
}
