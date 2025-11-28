"""
BSD300 dataset for image denoising (grayscale, variable size)
Expects images in data/BSD300/images/train/ (PNG, JPG, JPEG, BMP)
"""

import os
import torch
import numpy as np
from PIL import Image
from src.base_dataset import BaseDenoisingDataset


class BSD300Dataset(BaseDenoisingDataset):
    def __init__(
        self,
        root="./data/BSD300/images/train",
        noise_type="gaussian",
        noise_param=0.05,
        download=False,
    ):
        super().__init__(
            name="BSD300",
            description="BSD300 grayscale images for denoising",
            root=root,
            noise_type=noise_type,
            noise_param=noise_param,
            download=download,
        )
        self.noise_type = noise_type
        self.noise_param = noise_param
        self.image_paths = [
            os.path.join(root, fname)
            for fname in os.listdir(root)
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        assert len(self.image_paths) > 0, f"No images found in {root}"

    def __len__(self):
        return len(self.image_paths)

    def add_gaussian_noise(self, image, sigma=0.05):
        noise = torch.randn_like(image) * sigma
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0.0, 1.0)

    def add_poisson_noise(self, image, lam=1.0):
        scaled_image = image * 255.0
        noisy = np.random.poisson(scaled_image.numpy() * lam) / lam
        noisy_image = torch.from_numpy(noisy).float() / 255.0
        return torch.clamp(noisy_image, 0.0, 1.0)

    def add_salt_pepper_noise(self, image, prob=0.05):
        noisy_image = image.clone()
        salt_mask = torch.rand_like(image) < (prob / 2)
        noisy_image[salt_mask] = 1.0
        pepper_mask = torch.rand_like(image) < (prob / 2)
        noisy_image[pepper_mask] = 0.0
        return noisy_image

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("L")  # Grayscale
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # Shape: [1, H, W]
        # Add noise
        if self.noise_type == "gaussian":
            noisy_img = self.add_gaussian_noise(img, self.noise_param)
        elif self.noise_type == "poisson":
            noisy_img = self.add_poisson_noise(img, self.noise_param)
        elif self.noise_type == "salt_pepper":
            noisy_img = self.add_salt_pepper_noise(img, self.noise_param)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        return noisy_img, img

    def get_sample_shape(self):
        # Variable size, but always [1, H, W]
        sample = Image.open(self.image_paths[0]).convert("L")
        arr = np.array(sample)
        return (1, arr.shape[0], arr.shape[1])
