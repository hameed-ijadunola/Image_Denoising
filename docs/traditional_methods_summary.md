# Traditional Denoising Methods - Implementation Summary

## Overview

Successfully implemented **three traditional denoising methods** for the Image Denoising project:

1. **Non-Local Means (NLM)** - Buades et al., 2005
2. **Bilateral Filter** - Tomasi & Manduchi, 1998  
3. **Wavelet Denoising** - Donoho, 1995

All methods are now fully integrated with the registry system and can be used via command-line.

## Implementation Details

### 1. Non-Local Means (NLM)
**File:** `src/traditional/nlm.py`

**Description:** Edge-preserving filter that searches for similar patches across the entire image and averages them to denoise.

**Key Parameters:**
- `h` (float): Filter strength (default: 10.0)
- `template_window_size` (int): Patch size (default: 7)
- `search_window_size` (int): Search area size (default: 21)

**Implementation:** Uses OpenCV's `fastNlMeansDenoisingColored()`

**Usage:**
```bash
python main.py --mode train --model nlm --dataset cifar10
```

---

### 2. Bilateral Filter
**File:** `src/traditional/bilateral.py`

**Description:** Edge-preserving smoothing filter that combines spatial proximity and intensity similarity.

**Key Parameters:**
- `d` (int): Pixel neighborhood diameter (default: 9)
- `sigma_color` (float): Color space sigma (default: 75)
- `sigma_space` (float): Coordinate space sigma (default: 75)

**Implementation:** Uses OpenCV's `bilateralFilter()`

**Usage:**
```bash
python main.py --mode train --model bilateral --dataset cifar10
```

---

### 3. Wavelet Denoising
**File:** `src/traditional/wavelet.py`

**Description:** Decomposes images using wavelet transform, applies thresholding, and reconstructs the denoised image.

**Key Parameters:**
- `wavelet` (str): Wavelet type - 'db1', 'db4', 'sym4', etc. (default: 'db1')
- `mode` (str): Thresholding mode - 'soft' or 'hard' (default: 'soft')
- `level` (int): Decomposition level (default: None = automatic)
- `sigma` (float): Noise std estimate (default: None = automatic via MAD)

**Implementation:** Uses PyWavelets library with universal threshold

**Usage:**
```bash
python main.py --mode train --model wavelet --dataset cifar10
```

---

## Registry Integration

All methods are registered in `config.py`:

```python
MODEL_REGISTRY = {
    "unet": {...},  # Deep learning model
    "nlm": {...},    # Traditional method
    "bilateral": {...},  # Traditional method
    "wavelet": {...},    # Traditional method
}
```

Each entry includes:
- `requires_training: False` - No gradient-based optimization needed
- `default_params` - Sensible defaults for each method
- `paper` - Reference to original publication

---

## Key Modifications

### 1. Updated `main.py`
- Added logic to detect traditional methods via `requires_training` flag
- Skip Trainer initialization for traditional methods
- Go directly to evaluation without training loop
- Handle missing training losses in results

### 2. Updated `requirements.txt`
Added dependencies:
```
PyWavelets>=1.4.0  # For wavelet-based denoising
opencv-python>=4.8.0  # For NLM and bilateral filtering
```

### 3. Created Test Suite
**File:** `test_traditional_methods.py`

Comprehensive test script that:
- Loads each method from registry
- Creates synthetic noisy images
- Applies denoising
- Validates outputs (shape, range, no NaN/Inf)
- Calculates metrics (MSE, PSNR)

**Run tests:**
```bash
python test_traditional_methods.py
```

---

## Design Principles Followed

### ✅ Fail-Fast Philosophy
- All methods validate inputs immediately with assertions
- No silent error handling - exceptions propagate naturally
- Explicit precondition checks at function entry
- Range validation for all parameters

**Example:**
```python
assert h > 0, f"h must be positive, got {h}"
assert x.dim() == 4, f"Expected 4D tensor [B,C,H,W], got {x.dim()}D"
assert 0 <= x.min() and x.max() <= 1.0, f"Expected values in [0,1]"
```

### ✅ No Optional Dependencies
- All dependencies are REQUIRED (PyWavelets, opencv-python)
- Direct imports with no try/except
- Fail immediately if dependency missing

### ✅ Type Conversions for JSON
- All metrics converted to Python native types
- No numpy.float32 or torch.Tensor in JSON output

---

## Testing Results

All three methods pass comprehensive tests:

```
✓ nlm: PASSED
✓ bilateral: PASSED  
✓ wavelet: PASSED

✓ ALL TESTS PASSED!
```

Each method:
- Produces valid output shapes
- Maintains value ranges [0, 1]
- Contains no NaN or Inf values
- Can be loaded from registry
- Works with the main training/evaluation pipeline

---

## Performance Characteristics

### Non-Local Means
- **Best for:** General noise reduction with texture preservation
- **Speed:** Moderate (patch searching is computationally intensive)
- **Edge preservation:** Excellent

### Bilateral Filter
- **Best for:** Edge-preserving smoothing, fast denoising
- **Speed:** Fast (local operations only)
- **Edge preservation:** Very good

### Wavelet Denoising
- **Best for:** Gaussian noise, multi-scale features
- **Speed:** Fast (FFT-based transforms)
- **Edge preservation:** Good (depends on wavelet and threshold)

---

## Usage Examples

### Compare all methods on CIFAR-10:
```bash
# Test NLM
python main.py --mode train --model nlm --dataset cifar10

# Test Bilateral
python main.py --mode train --model bilateral --dataset cifar10

# Test Wavelet
python main.py --mode train --model wavelet --dataset cifar10

# Compare with U-Net
python main.py --mode train --model unet --dataset cifar10 --epochs 5
```

### Custom parameters:
Modify `config.py` MODEL_REGISTRY default_params, or extend the registry utility to accept parameter overrides.

---

## Future Enhancements

Potential additions:
1. **Median Filter** - Good for salt & pepper noise
2. **Total Variation (TV) Denoising** - Preserves edges
3. **BM3D** - State-of-art traditional method
4. **Anisotropic Diffusion** - Edge-preserving PDE-based
5. **Dictionary Learning** - Sparse coding approaches

All can follow the same template pattern in `src/traditional/template.py`.

---

## File Structure

```
src/traditional/
├── __init__.py
├── template.py          # Template and examples
├── nlm.py              # Non-Local Means ✅
├── bilateral.py        # Bilateral Filter ✅
└── wavelet.py          # Wavelet Denoising ✅

test_traditional_methods.py  # Comprehensive test suite ✅
```

---

## Conclusion

All traditional denoising methods are fully implemented, tested, and integrated with the project's registry system. They can be used alongside deep learning methods like U-Net, providing a comprehensive toolkit for image denoising research and comparison.

**Key Achievement:** Zero-training methods that work out-of-the-box with the same interface as deep learning models, following all project conventions and fail-fast principles.
