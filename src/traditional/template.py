"""
Template for implementing traditional denoising methods

Copy this file and modify it to implement your own traditional method.
Example: cp template.py nlm.py, then implement NonLocalMeans class
"""

from src.base_model import TraditionalMethod
import torch
import numpy as np


class TraditionalMethodTemplate(TraditionalMethod):
    """
    Template for traditional denoising methods

    Traditional methods:
    - Don't require training (parameters are set, not learned)
    - Often based on signal processing techniques
    - Usually work on CPU (can be slower than GPU-based deep learning)
    - Examples: NLM, Bilateral Filter, Wavelet, Median Filter
    """

    def __init__(self, param1=1.0, param2=5, **kwargs):
        """
        Initialize the traditional method

        Args:
            param1: Example parameter (e.g., sigma for Gaussian)
            param2: Example parameter (e.g., kernel size)
            **kwargs: Additional parameters
        """
        super().__init__(
            name="Template Method",
            description="Template for traditional denoising methods",
        )

        # Store method parameters
        self.param1 = param1
        self.param2 = param2

        # Any additional setup
        self._setup()

    def _setup(self):
        """Additional setup if needed"""
        pass

    def forward(self, x):
        """
        Denoise a batch of images

        Args:
            x: Tensor of shape [B, C, H, W] with noisy images
               Values in range [0, 1]

        Returns:
            Tensor of shape [B, C, H, W] with denoised images
            Values in range [0, 1]
        """
        # Process each image in the batch
        denoised_batch = []

        for img in x:
            # Convert from tensor [C, H, W] to numpy [H, W, C]
            img_np = img.permute(1, 2, 0).cpu().numpy()

            # Apply your denoising algorithm
            denoised_np = self._denoise_single_image(img_np)

            # Convert back to tensor [C, H, W]
            denoised = torch.from_numpy(denoised_np).permute(2, 0, 1)
            denoised_batch.append(denoised)

        # Stack batch and move to same device as input
        return torch.stack(denoised_batch).to(x.device)

    def _denoise_single_image(self, image_np):
        """
        Denoise a single image (numpy array)

        Args:
            image_np: Numpy array of shape [H, W, C] in range [0, 1]

        Returns:
            Denoised numpy array of shape [H, W, C] in range [0, 1]
        """
        # IMPLEMENT YOUR DENOISING ALGORITHM HERE
        #
        # Example using OpenCV Gaussian blur:
        # import cv2
        # denoised = cv2.GaussianBlur(image_np, (self.param2, self.param2), self.param1)

        # For now, just return the input (no denoising)
        denoised = image_np

        # Ensure output is in valid range
        denoised = np.clip(denoised, 0.0, 1.0)

        return denoised


# ============================================================================
# EXAMPLE IMPLEMENTATIONS
# ============================================================================


class GaussianFilter(TraditionalMethod):
    """Example: Gaussian blur denoising"""

    def __init__(self, sigma=1.0, kernel_size=5):
        super().__init__(
            name="Gaussian Filter", description="Gaussian blur for denoising"
        )
        self.sigma = sigma
        self.kernel_size = kernel_size

    def forward(self, x):
        """Apply Gaussian blur to batch"""
        import cv2

        denoised_batch = []
        for img in x:
            # Convert to numpy [H, W, C]
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # Apply Gaussian blur
            denoised = cv2.GaussianBlur(
                img_np, (self.kernel_size, self.kernel_size), self.sigma
            )

            # Convert back to tensor [C, H, W]
            denoised = torch.from_numpy(denoised / 255.0).permute(2, 0, 1).float()
            denoised_batch.append(denoised)

        return torch.stack(denoised_batch).to(x.device)


class MedianFilter(TraditionalMethod):
    """Example: Median filter (good for salt & pepper noise)"""

    def __init__(self, kernel_size=5):
        super().__init__(
            name="Median Filter", description="Median filter for impulse noise"
        )
        self.kernel_size = kernel_size

    def forward(self, x):
        """Apply median filter to batch"""
        import cv2

        denoised_batch = []
        for img in x:
            # Convert to numpy [H, W, C]
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # Apply median filter
            denoised = cv2.medianBlur(img_np, self.kernel_size)

            # Convert back to tensor [C, H, W]
            denoised = torch.from_numpy(denoised / 255.0).permute(2, 0, 1).float()
            denoised_batch.append(denoised)

        return torch.stack(denoised_batch).to(x.device)


# ============================================================================
# HOW TO ADD TO REGISTRY
# ============================================================================

"""
After implementing your method, add it to config.py MODEL_REGISTRY:

"gaussian_filter": {
    "name": "Gaussian Filter",
    "type": "traditional",
    "class": "GaussianFilter",
    "module": "src.traditional.template",  # Change to your module
    "description": "Gaussian blur denoising",
    "requires_training": False,
    "default_params": {
        "sigma": 1.0,
        "kernel_size": 5,
    },
},

Then use it:

from config import MODEL_REGISTRY
from src.registry import get_model_from_registry

# Get the model
model = get_model_from_registry("gaussian_filter", MODEL_REGISTRY)

# Use it (no training needed!)
denoised = model.denoise(noisy_images)
"""


if __name__ == "__main__":
    # Test the template
    print("Testing traditional method template...")

    # Create test image
    test_img = torch.randn(2, 3, 32, 32)  # Batch of 2 images

    # Test Gaussian filter
    gaussian = GaussianFilter(sigma=1.0, kernel_size=5)
    print(f"\nModel: {gaussian.name}")
    print(f"Type: {gaussian.model_type}")

    denoised = gaussian(test_img)
    print(f"Input: {test_img.shape}, Output: {denoised.shape}")

    # Test Median filter
    median = MedianFilter(kernel_size=5)
    print(f"\nModel: {median.name}")
    print(f"Type: {median.model_type}")

    denoised = median(test_img)
    print(f"Input: {test_img.shape}, Output: {denoised.shape}")

    print("\n✓ Template works! Now implement your own methods.")
