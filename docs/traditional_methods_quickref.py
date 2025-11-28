"""
Quick Reference: How to Use Traditional Denoising Methods
===========================================================

All traditional methods are now available in the MODEL_REGISTRY and can be
used with the same interface as deep learning models.

BASIC USAGE
-----------

1. Non-Local Means:
   python main.py --mode train --model nlm --dataset cifar10 --batch-size 16

2. Bilateral Filter:
   python main.py --mode train --model bilateral --dataset cifar10 --batch-size 16

3. Wavelet Denoising:
   python main.py --mode train --model wavelet --dataset cifar10 --batch-size 16

Note: Despite using --mode train, traditional methods skip training and
      go directly to evaluation.


PROGRAMMATIC USAGE
------------------

from config import MODEL_REGISTRY
from src.registry import get_model_from_registry
import torch

# Load any traditional method from registry
model = get_model_from_registry('nlm', MODEL_REGISTRY)

# Use it directly (no training needed!)
noisy_images = torch.rand(4, 3, 32, 32)  # Batch of images
denoised = model(noisy_images)

# That's it! No optimizer, no training loop needed.


CUSTOMIZING PARAMETERS
----------------------

Option 1: Modify config.py MODEL_REGISTRY default_params

Option 2: Pass parameters directly
model = get_model_from_registry('nlm', MODEL_REGISTRY, h=15, template_window_size=9)


TESTING
-------

# Test all traditional methods
python test_traditional_methods.py

# Test individual method implementations
python src/traditional/nlm.py
python src/traditional/bilateral.py
python src/traditional/wavelet.py


COMPARING WITH DEEP LEARNING
-----------------------------

# Run traditional method (instant evaluation)
python main.py --model nlm --dataset cifar10

# Run deep learning method (requires training)
python main.py --model unet --dataset cifar10 --epochs 10

# Compare results in ./results/ directory


KEY DIFFERENCES: Traditional vs Deep Learning
----------------------------------------------

Traditional Methods:
  ✓ No training required
  ✓ Work immediately out-of-the-box
  ✓ Fixed parameters (no learning)
  ✓ Faster to evaluate (no GPU needed)
  ✓ Parameters have clear physical meaning
  ✗ Generally lower PSNR than trained models
  ✗ Not adaptive to specific noise types

Deep Learning Methods:
  ✓ Learn optimal denoising from data
  ✓ Higher PSNR after training
  ✓ Adaptive to training data characteristics
  ✗ Require training (time + GPU)
  ✗ Need large datasets
  ✗ Parameters less interpretable


WHEN TO USE WHICH
-----------------

Use Traditional Methods:
  • Quick baseline comparisons
  • No training data available
  • Interpretable parameters needed
  • CPU-only environment
  • Real-time processing requirements

Use Deep Learning Methods:
  • Have training data and GPU
  • Need best possible quality
  • Can afford training time
  • Want adaptive performance


AVAILABLE TRADITIONAL METHODS
------------------------------

1. Non-Local Means (nlm)
   - Best for: Texture preservation
   - Speed: Moderate
   - Params: h, template_window_size, search_window_size

2. Bilateral Filter (bilateral)
   - Best for: Edge preservation
   - Speed: Fast
   - Params: d, sigma_color, sigma_space

3. Wavelet (wavelet)
   - Best for: Multi-scale features
   - Speed: Fast
   - Params: wavelet, mode, level, sigma


TROUBLESHOOTING
---------------

Q: "ValueError: optimizer got an empty parameter list"
A: You're trying to train a traditional method. They have no parameters
   to optimize. The code should automatically skip training, but if not,
   check that requires_training: False in config.py

Q: "ModuleNotFoundError: No module named 'pywt'"
A: Install missing dependency: pip install PyWavelets

Q: "ModuleNotFoundError: No module named 'cv2'"
A: Install missing dependency: pip install opencv-python

Q: Traditional method gives poor results
A: Try adjusting parameters in config.py MODEL_REGISTRY.
   Different noise types/levels need different parameters.


MORE INFORMATION
----------------

See docs/traditional_methods_summary.md for full implementation details.
"""

if __name__ == "__main__":
    print(__doc__)
