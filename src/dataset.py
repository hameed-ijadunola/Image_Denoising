"""
Dataset utilities for CIFAR-10 with noise injection
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .base_dataset import BaseDenoisingDataset


class NoisyCIFAR10(BaseDenoisingDataset):
    """
    CIFAR-10 dataset with added noise for denoising tasks
    """

    def __init__(
        self,
        root="./data",
        train=True,
        noise_type="gaussian",
        noise_param=0.05,
        download=True,
    ):
        """
        Args:
            root (str): Root directory of dataset
            train (bool): If True, creates dataset from training set
            noise_type (str): Type of noise to add ('gaussian', 'poisson', 'salt_pepper')
            noise_param (float): Noise parameter (sigma for gaussian, lambda for poisson, probability for salt_pepper)
            download (bool): If True, downloads the dataset
        """
        super().__init__(
            name="CIFAR-10",
            description="CIFAR-10 dataset (32x32 color images, 10 classes)",
            root=root,
            train=train,
            noise_type=noise_type,
            noise_param=noise_param,
            download=download,
        )
        self.noise_type = noise_type
        self.noise_param = noise_param

        # Load CIFAR-10
        self.cifar10 = datasets.CIFAR10(
            root=root, train=train, download=download, transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.cifar10)

    def add_gaussian_noise(self, image, sigma=0.05):
        """Add Gaussian noise to image"""
        noise = torch.randn_like(image) * sigma
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0.0, 1.0)

    def add_poisson_noise(self, image, lam=1.0):
        """Add Poisson noise to image"""
        # Scale image to [0, 255] for Poisson
        scaled_image = image * 255.0
        # Add Poisson noise
        noisy = np.random.poisson(scaled_image.numpy() * lam) / lam
        noisy_image = torch.from_numpy(noisy).float() / 255.0
        return torch.clamp(noisy_image, 0.0, 1.0)

    def add_salt_pepper_noise(self, image, prob=0.05):
        """Add salt and pepper noise to image"""
        noisy_image = image.clone()
        # Salt noise (white)
        salt_mask = torch.rand_like(image) < (prob / 2)
        noisy_image[salt_mask] = 1.0
        # Pepper noise (black)
        pepper_mask = torch.rand_like(image) < (prob / 2)
        noisy_image[pepper_mask] = 0.0
        return noisy_image

    def __getitem__(self, idx):
        """
        Returns:
            clean_image: Original clean image
            noisy_image: Image with added noise
        """
        clean_image, _ = self.cifar10[idx]

        # Add noise based on type
        if self.noise_type == "gaussian":
            noisy_image = self.add_gaussian_noise(clean_image, self.noise_param)
        elif self.noise_type == "poisson":
            noisy_image = self.add_poisson_noise(clean_image, self.noise_param)
        elif self.noise_type == "salt_pepper":
            noisy_image = self.add_salt_pepper_noise(clean_image, self.noise_param)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        return noisy_image, clean_image

    def get_sample_shape(self):
        """Return the shape of CIFAR-10 images: (3, 32, 32)"""
        return (3, 32, 32)


def get_dataloaders(
    batch_size=16, noise_type="gaussian", noise_param=0.05, num_workers=2
):
    """
    Create train and test dataloaders for CIFAR-10 with noise

    Args:
        batch_size (int): Batch size for training
        noise_type (str): Type of noise to add
        noise_param (float): Noise parameter
        num_workers (int): Number of worker processes for data loading

    Returns:
        train_loader, test_loader
    """
    train_dataset = NoisyCIFAR10(
        root="./data",
        train=True,
        noise_type=noise_type,
        noise_param=noise_param,
        download=True,
    )

    test_dataset = NoisyCIFAR10(
        root="./data",
        train=False,
        noise_type=noise_type,
        noise_param=noise_param,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing NoisyCIFAR10 dataset...")
    train_loader, test_loader = get_dataloaders(
        batch_size=4, noise_type="gaussian", noise_param=0.05
    )

    noisy, clean = next(iter(train_loader))
    print(f"Noisy batch shape: {noisy.shape}")
    print(f"Clean batch shape: {clean.shape}")
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
