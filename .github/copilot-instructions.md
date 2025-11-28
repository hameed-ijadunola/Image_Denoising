# GitHub Copilot Instructions for Image Denoising Project

## Core Principles

### Fail-Fast Philosophy
- **NEVER use try/except blocks** unless absolutely critical for resource cleanup (e.g., file handles, GPU memory)
- Let exceptions propagate naturally to surface issues immediately during development
- Use explicit precondition checks with assert statements for development-time validation
- Validate inputs at function entry and fail immediately with clear error messages
- Prefer explicit type hints and let mypy/type checkers catch errors early

### Error Handling
```python
# ❌ AVOID - Silently catching errors
try:
    result = process_image(img)
except Exception as e:
    print(f"Error: {e}")
    return None

# ✅ PREFER - Fail fast with clear error messages
assert img is not None, "Input image cannot be None"
assert img.shape[-1] == 3, f"Expected 3-channel image, got {img.shape[-1]} channels"
result = process_image(img)  # Let it fail if something is wrong
```

### Input Validation
- Check preconditions explicitly at function entry
- Use assertions for invariants that should never fail in correct code
- Use explicit if-checks with ValueError/TypeError for user-provided inputs
- Validate tensor shapes, dtypes, and ranges immediately

```python
# ✅ GOOD - Explicit validation
def denoise_batch(images: torch.Tensor, noise_std: float) -> torch.Tensor:
    assert images.dim() == 4, f"Expected 4D tensor [B,C,H,W], got {images.dim()}D"
    assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
    if not 0 < noise_std <= 1.0:
        raise ValueError(f"noise_std must be in (0, 1], got {noise_std}")
    
    # Process without error handling - fail fast if something breaks
    return model(images)
```

## Project-Specific Conventions

### Model Registration
- All models must be registered in `config.py` MODEL_REGISTRY
- Use registry pattern from `src/registry.py` for loading models
- Fail immediately if model not found in registry
- No fallback to default models

### Dataset Handling
- Inherit from `BaseDataset` in `src/base_dataset.py`
- Validate image shapes and normalization at dataset construction
- Assert consistent tensor dimensions in `__getitem__`
- No silent skipping of corrupted images - fail and report

### Training Loop
- Use `src/train.py` Trainer class for all training
- Validate hyperparameters at Trainer initialization
- Assert gradient sanity (not NaN/Inf) after each backward pass
- Fail training immediately on detection of anomalies

```python
# ✅ GOOD - Fail fast on gradient issues
loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
optimizer.step()
```

### Configuration
- All experimental configs in `config.py`
- No magic numbers - use named constants
- Validate config dictionaries at load time
- Missing required keys should raise KeyError immediately

### Metrics and Evaluation
- Use `src/utils.py` metric functions (PSNR, SSIM, etc.)
- Assert image value ranges before metric calculation
- No silent clipping or normalization - fail if ranges are wrong
- Validate metric results are in expected ranges
- **Always convert numpy/torch types to Python native types** for JSON serialization

```python
# ✅ GOOD - Explicit range validation and type conversion
def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    assert img1.shape == img2.shape, "Images must have same shape"
    assert 0 <= img1.min() and img1.max() <= 1.0, "img1 must be in [0, 1]"
    assert 0 <= img2.min() and img2.max() <= 1.0, "img2 must be in [0, 1]"
    
    mse = torch.mean((img1 - img2) ** 2)
    assert mse >= 0, "MSE cannot be negative"
    
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

# ✅ GOOD - Convert all metrics to Python float for JSON serialization
def calculate_all_metrics(img1, img2, lpips_model=None):
    metrics = {
        "psnr": float(calculate_psnr(img1, img2)),  # Ensure Python float
        "ssim": float(calculate_ssim(img1, img2)),  # Not numpy.float32
        "mse": float(calculate_mse(img1, img2)),
        "mae": float(calculate_mae(img1, img2)),
    }
    return metrics
```

## Code Style

### Type Hints
- Always use type hints for function signatures
- Use torch.Tensor, not just Tensor
- Specify shapes in docstrings when relevant

### Assertions
- Use assertions for invariants and developer errors
- Use exceptions (ValueError, TypeError) for user/input errors
- Include descriptive messages in all assertions

### Device Management
- Always be explicit about device (cpu/cuda)
- Assert tensors are on expected device before operations
- No silent device transfers
- When passing tensors to models, ensure they're on the same device

```python
# ✅ GOOD - Assert device consistency
def forward(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    assert x.device == device, f"Input on {x.device}, expected {device}"
    return self.model(x)

# ✅ GOOD - Check model device matches input device
def calculate_metric(img1: torch.Tensor, img2: torch.Tensor, model):
    model_device = next(model.parameters()).device
    assert img1.device == model_device, (
        f"Input on {img1.device} but model on {model_device}"
    )
    return model(img1, img2)
```

### Type Conversions for Serialization
- **CRITICAL**: Convert numpy/torch types to Python native types before JSON serialization
- Use `float()` for numeric values, `int()` for integers, `list()` for arrays
- Do NOT use try/except to handle serialization errors - fix the types!

```python
# ✅ GOOD - Explicit type conversion
import json
import numpy as np

metrics = {
    "psnr": float(psnr_value),      # Convert numpy.float32 → float
    "epoch": int(epoch_num),         # Convert numpy.int64 → int
    "losses": [float(x) for x in losses],  # Convert all elements
}

with open("results.json", "w") as f:
    json.dump(metrics, f)

# ❌ BAD - Silent error handling
try:
    json.dump(metrics, f)
except TypeError:
    # Don't hide the problem!
    pass
```

### Logging
- Use print() for critical information during development
- Use wandb for experiment tracking (required dependency)
- Log hyperparameters and metrics, but never catch and hide training errors
- All logging dependencies must be installed - no optional imports

## No Optional Dependencies

**CRITICAL: This project does not allow optional dependencies.**

All dependencies in `requirements.txt` are REQUIRED and must be installed:
- `wandb` - Required for experiment tracking
- `lpips` - Required for perceptual loss metrics
- All other dependencies must be available

### Import Pattern - ALWAYS Use Direct Imports

```python
# ✅ CORRECT - Direct import, fail fast if not available
import wandb
import lpips

# ❌ NEVER DO THIS - No try/except for imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
```

### Why No Optional Dependencies?

1. **Fail Fast**: If a dependency is missing, we want to know immediately, not discover it deep in execution
2. **Consistency**: All developers and deployment environments must have identical dependencies
3. **No Silent Degradation**: Missing features due to optional dependencies lead to confusing behavior
4. **Reproducibility**: Experiments must be fully reproducible with exact same dependencies

### When Dependencies Are Truly Missing

If a required dependency is not installed, the import will fail with a clear error:
- This is GOOD - it tells the user exactly what to install
- Use `requirements.txt` to document all required packages
- Run `pip install -r requirements.txt` to install everything

## When Try/Except IS Acceptable

Only use try/except for:
1. **File I/O** - when you need to clean up resources
2. **External API calls** - network requests, cloud storage  
3. **Explicit user-facing error recovery** - CLI tools with multiple attempts
4. **Testing** - test files verifying expected exceptions

```python
# ✅ ACCEPTABLE - Resource cleanup
checkpoint_path = None
try:
    checkpoint_path = save_checkpoint(model, path)
finally:
    if checkpoint_path and os.path.exists(checkpoint_path + '.tmp'):
        os.remove(checkpoint_path + '.tmp')

# ❌ NEVER ACCEPTABLE - Optional dependency pattern
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
```

## Testing
- Tests in `test_*.py` files MAY use try/except to verify expected failures
- Production code should not catch exceptions from test code
- Assert expected values, don't just check for absence of errors

## Anti-Patterns to Avoid

```python
# ❌ NEVER DO THIS
try:
    model = load_model(path)
except:
    model = UNet()  # Silent fallback

# ❌ NEVER DO THIS  
try:
    loss = criterion(output, target)
except Exception as e:
    loss = torch.tensor(0.0)  # Hiding training errors

# ❌ NEVER DO THIS
def process(x):
    try:
        return complicated_operation(x)
    except:
        return None  # What went wrong? Who knows!
```

## Command Patterns

When generating code for:
- **Training**: Use `python main.py --mode train --epochs N`
- **Evaluation**: Use `python main.py --mode eval`
- **Experiments**: Use `python run_experiment.py` with config.py modifications
- **Testing**: Use `python test_*.py` or `pytest`

## Remember
- Fail fast, fail loud, fail with context
- Exceptions are for exceptional circumstances, not control flow
- Every assertion should have a descriptive message
- Type hints and assertions are your friends for catching bugs early
- If something is wrong, it should be obvious immediately
