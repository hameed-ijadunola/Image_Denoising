"""
Base classes for datasets
This enables easy addition of new datasets
"""

from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDenoisingDataset(ABC, Dataset):
    """
    Abstract base class for all denoising datasets.

    Provides a common interface for different datasets used in image denoising.
    """

    def __init__(self, name: str, description: str = "", **kwargs):
        """
        Args:
            name: Name of the dataset (e.g., 'CIFAR-10', 'BSD68', 'Set12')
            description: Brief description of the dataset
            **kwargs: Dataset-specific parameters
        """
        super().__init__()
        self.name = name
        self.description = description
        self.config = kwargs

    @abstractmethod
    def __len__(self):
        """Return the size of the dataset"""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """
        Get a sample from the dataset

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (noisy_image, clean_image)
        """
        pass

    def get_info(self):
        """Get dataset information"""
        return {
            "name": self.name,
            "description": self.description,
            "size": len(self),
            "config": self.config,
        }

    @abstractmethod
    def get_sample_shape(self):
        """
        Get the shape of samples in the dataset

        Returns:
            Tuple representing (channels, height, width)
        """
        pass
