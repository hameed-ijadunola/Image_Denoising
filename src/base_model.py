"""
Base classes for denoising models and techniques
This enables easy addition of new models and traditional methods
"""

from abc import ABC, abstractmethod
import torch.nn as nn


class BaseDenoisingModel(ABC, nn.Module):
    """
    Abstract base class for all denoising models and techniques.

    This class provides a common interface for:
    - Deep learning models (U-Net, DnCNN, etc.)
    - Traditional methods (NLM, BM3D, Wavelet, etc.)
    """

    def __init__(self, name: str, model_type: str, description: str = ""):
        """
        Args:
            name: Name of the model/technique
            model_type: Type - 'deep_learning' or 'traditional'
            description: Brief description of the method
        """
        super().__init__()
        self.name = name
        self.model_type = model_type
        self.description = description
        self._trained = False

    @abstractmethod
    def forward(self, x):
        """
        Forward pass for denoising

        Args:
            x: Noisy input tensor [B, C, H, W]

        Returns:
            Denoised output tensor [B, C, H, W]
        """
        pass

    def denoise(self, x):
        """
        Convenience method for denoising (calls forward)
        Can be overridden for traditional methods that don't need forward()

        Args:
            x: Noisy input tensor

        Returns:
            Denoised output tensor
        """
        return self.forward(x)

    def get_info(self):
        """Get model information"""
        return {
            "name": self.name,
            "type": self.model_type,
            "description": self.description,
            "num_parameters": self.count_parameters()
            if self.model_type == "deep_learning"
            else 0,
            "trained": self._trained,
        }

    def count_parameters(self):
        """Count trainable parameters (for deep learning models)"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def mark_trained(self):
        """Mark model as trained"""
        self._trained = True


class DeepLearningModel(BaseDenoisingModel):
    """
    Base class for deep learning denoising models
    These models need training
    """

    def __init__(self, name: str, description: str = ""):
        super().__init__(name=name, model_type="deep_learning", description=description)


class TraditionalMethod(BaseDenoisingModel):
    """
    Base class for traditional denoising methods
    These methods are parameter-based (no training needed)
    """

    def __init__(self, name: str, description: str = ""):
        super().__init__(name=name, model_type="traditional", description=description)
        self._trained = True  # Traditional methods don't need training

    def parameters(self):
        """Traditional methods have no trainable parameters"""
        return []
