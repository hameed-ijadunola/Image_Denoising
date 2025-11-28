"""
Registry utilities for managing models, datasets, and experiments
This module provides helper functions to work with the model and dataset registries
"""

import importlib
from typing import Dict, Any, Optional


def get_model_from_registry(model_key: str, registry: Dict, **kwargs):
    """
    Get a model instance from the MODEL_REGISTRY

    Args:
        model_key: Key in MODEL_REGISTRY (e.g., 'unet', 'dncnn')
        registry: The MODEL_REGISTRY dictionary
        **kwargs: Override default parameters for the model

    Returns:
        Instance of the model

    Example:
        >>> model = get_model_from_registry('unet', MODEL_REGISTRY, dropout_rate=0.3)
    """
    if model_key not in registry:
        available_models = ", ".join(registry.keys())
        raise ValueError(
            f"Model '{model_key}' not found in registry. "
            f"Available models: {available_models}"
        )

    model_config = registry[model_key]

    # Import the module and get the class
    module = importlib.import_module(model_config["module"])
    model_class = getattr(module, model_config["class"])

    # Merge default params with provided kwargs
    params = model_config.get("default_params", {}).copy()
    params.update(kwargs)

    # Create and return model instance
    return model_class(**params)


def get_dataset_from_registry(
    dataset_key: str, registry: Dict, train: bool = True, **kwargs
):
    """
    Get a dataset instance from the DATASET_REGISTRY

    Args:
        dataset_key: Key in DATASET_REGISTRY (e.g., 'cifar10', 'bsd68')
        registry: The DATASET_REGISTRY dictionary
        train: Whether to get training or test dataset
        **kwargs: Override default parameters for the dataset

    Returns:
        Instance of the dataset

    Example:
        >>> dataset = get_dataset_from_registry('cifar10', DATASET_REGISTRY, train=True)
    """
    if dataset_key not in registry:
        available_datasets = ", ".join(registry.keys())
        raise ValueError(
            f"Dataset '{dataset_key}' not found in registry. "
            f"Available datasets: {available_datasets}"
        )

    dataset_config = registry[dataset_key]

    # Import the module and get the class
    module = importlib.import_module(dataset_config["module"])
    dataset_class = getattr(module, dataset_config["class"])

    # Merge default params with provided kwargs
    params = dataset_config.get("default_params", {}).copy()
    params.update(kwargs)
    params["train"] = train

    # Create and return dataset instance
    return dataset_class(**params)


def list_models(registry: Dict, model_type: Optional[str] = None) -> Dict[str, Dict]:
    """
    List all available models in the registry

    Args:
        registry: The MODEL_REGISTRY dictionary
        model_type: Filter by type ('deep_learning' or 'traditional'), None for all

    Returns:
        Dictionary of filtered models
    """
    if model_type is None:
        return registry

    return {
        key: value for key, value in registry.items() if value.get("type") == model_type
    }


def list_datasets(registry: Dict) -> Dict[str, Dict]:
    """
    List all available datasets in the registry

    Args:
        registry: The DATASET_REGISTRY dictionary

    Returns:
        Dictionary of all datasets
    """
    return registry


def print_registry_info(registry: Dict, registry_type: str = "model"):
    """
    Pretty print information about a registry

    Args:
        registry: The registry dictionary to print
        registry_type: Type of registry ('model', 'dataset', 'noise')
    """
    print(f"\n{'=' * 70}")
    print(f"{registry_type.upper()} REGISTRY")
    print(f"{'=' * 70}\n")

    for key, config in registry.items():
        print(f"Key: {key}")
        print(f"  Name: {config.get('name', 'N/A')}")

        if registry_type == "model":
            print(f"  Type: {config.get('type', 'N/A')}")
            print(f"  Requires Training: {config.get('requires_training', 'N/A')}")
            if "paper" in config:
                print(f"  Paper: {config['paper']}")

        if registry_type == "dataset":
            print(f"  Size: {config.get('image_size', 'N/A')}")
            print(f"  Channels: {config.get('channels', 'N/A')}")
            print(f"  Train Samples: {config.get('num_train', 'N/A')}")
            print(f"  Test Samples: {config.get('num_test', 'N/A')}")

        print(f"  Description: {config.get('description', 'N/A')}")
        print()


def get_experiment_metadata(
    model_key: str,
    dataset_key: str,
    model_registry: Dict,
    dataset_registry: Dict,
    **training_params,
) -> Dict[str, Any]:
    """
    Generate comprehensive metadata for an experiment

    Args:
        model_key: Key from MODEL_REGISTRY
        dataset_key: Key from DATASET_REGISTRY
        model_registry: The MODEL_REGISTRY dictionary
        dataset_registry: The DATASET_REGISTRY dictionary
        **training_params: Additional training parameters (epochs, lr, etc.)

    Returns:
        Dictionary containing full experiment metadata
    """
    metadata = {
        "model": {"key": model_key, **model_registry.get(model_key, {})},
        "dataset": {"key": dataset_key, **dataset_registry.get(dataset_key, {})},
        "training": training_params,
    }

    return metadata


def validate_experiment_config(
    config: Dict, model_registry: Dict, dataset_registry: Dict
) -> bool:
    """
    Validate an experiment configuration

    Args:
        config: Configuration dictionary
        model_registry: The MODEL_REGISTRY dictionary
        dataset_registry: The DATASET_REGISTRY dictionary

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check model
    if "model" in config:
        if config["model"] not in model_registry:
            available = ", ".join(model_registry.keys())
            raise ValueError(
                f"Model '{config['model']}' not in registry. Available: {available}"
            )

    # Check dataset
    if "dataset" in config:
        if config["dataset"] not in dataset_registry:
            available = ", ".join(dataset_registry.keys())
            raise ValueError(
                f"Dataset '{config['dataset']}' not in registry. Available: {available}"
            )

    return True


if __name__ == "__main__":
    # Example usage
    from config import MODEL_REGISTRY, DATASET_REGISTRY

    print_registry_info(MODEL_REGISTRY, "model")
    print_registry_info(DATASET_REGISTRY, "dataset")
