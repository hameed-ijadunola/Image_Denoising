# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### 1. Conda Not Found
**Problem**: `conda: command not found`

**Solution**:
```bash
# Install Miniconda (recommended)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Or install Anaconda (larger, includes more packages)
# Visit: https://www.anaconda.com/download

# After installation, restart your terminal or run:
source ~/.bashrc
```

#### 2. PyTorch Installation Fails
**Problem**: `pip install torch` fails or takes too long

**Solution with Conda (Recommended)**:
```bash
# Activate your conda environment first
conda activate image_denoising

# Install PyTorch with conda (faster and more reliable)
# For CPU-only:
conda install pytorch torchvision cpuonly -c pytorch

# For CUDA 11.8:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1:
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Solution with pip**:
```bash
# For CPU-only version (faster to install)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8 (if you have NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Permission Denied on setup.sh
**Problem**: `bash: ./setup.sh: Permission denied`

**Solution**:
```bash
chmod +x setup.sh
./setup.sh
```

#### 4. Conda Environment Activation Fails
**Problem**: `conda activate` not working

**Solution**:
```bash
# Initialize conda for your shell
conda init bash  # or zsh, fish, etc.

# Restart terminal or run:
source ~/.bashrc

# Then try again:
conda activate image_denoising
```

### Runtime Issues

#### 3. CUDA Out of Memory
**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size
python main.py --batch-size 8  # or even 4

# Or force CPU usage
python main.py --cpu
```

#### 4. Too Slow on CPU
**Problem**: Training takes hours

**Solution**:
- Reduce number of epochs for testing:
  ```bash
  python demo.py  # Only 1 epoch
  ```
- Reduce batch size won't help with speed, but you can reduce dataset size in code
- Consider using Google Colab (free GPU): https://colab.research.google.com/

#### 5. "Import torch could not be resolved"
**Problem**: VS Code shows import errors but code runs fine

**Solution**:
- This is just a linting issue, the code will still run
- Select the correct Python interpreter in VS Code (Ctrl+Shift+P → "Python: Select Interpreter")
- Make sure you activated the virtual environment

#### 6. Dataset Download Fails
**Problem**: CIFAR-10 download times out or fails

**Solution**:
```bash
# Manually download CIFAR-10
mkdir -p data/cifar-10-batches-py
cd data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
```

### Training Issues

#### 7. Loss Not Decreasing
**Problem**: Loss stays high or increases

**Possible Causes & Solutions**:
- Learning rate too high:
  ```bash
  python main.py --lr 0.0001
  ```
- Learning rate too low:
  ```bash
  python main.py --lr 0.01
  ```
- Wrong optimizer for your setup:
  ```bash
  python main.py --optimizer adam  # Usually best
  ```

#### 8. NaN Loss
**Problem**: Loss becomes NaN during training

**Solution**:
- Reduce learning rate:
  ```bash
  python main.py --lr 0.0001
  ```
- Check your PyTorch installation:
  ```bash
  python -c "import torch; print(torch.__version__)"
  ```

#### 9. "No module named 'src'"
**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Make sure you're running from the project root
cd /home/hijaduno/Documents/Image_Denoising
python main.py

# Or for demo.py and run_experiment.py, they handle paths automatically
```

### Output Issues

#### 10. No Plots Showing
**Problem**: Plots don't display (common in SSH/remote sessions)

**Solution**:
- Plots are automatically saved to `./results/` directory
- View them later with any image viewer
- For Jupyter: use `%matplotlib inline`
- For remote server: plots save but won't display (check results folder)

#### 11. Permission Denied Writing Files
**Problem**: Can't save models or results

**Solution**:
```bash
# Check permissions
ls -la models/ results/

# Fix permissions
chmod 755 models/ results/

# Or specify different directories
python main.py --save-dir ~/my_models --results-dir ~/my_results
```

### Performance Issues

#### 12. Training Too Slow
**Expected Times**:
- CPU: ~1 hour per epoch (normal)
- GPU: ~5-10 minutes per epoch (normal)

**If slower than this**:
- Check CPU usage: `top` or `htop`
- Reduce num_workers: `--num-workers 0`
- Close other applications
- Use GPU if available

#### 13. High Memory Usage
**Problem**: System runs out of RAM

**Solution**:
```bash
# Reduce batch size
python main.py --batch-size 8

# Reduce number of workers
python main.py --num-workers 0
```

### Evaluation Issues

#### 14. Low PSNR Improvement
**Problem**: PSNR improvement is less than expected

**Possible Causes**:
- Model not trained enough (train more epochs)
- Noise level too high (reduce --noise-param)
- Learning rate not optimal (try different values)
- Need to train longer:
  ```bash
  python main.py --epochs 20
  ```

#### 15. Checkpoint Loading Fails
**Problem**: Can't load saved model

**Solution**:
```bash
# Check file exists
ls -la models/*/best_model.pth

# Use full path
python main.py --mode eval --checkpoint $(pwd)/models/gaussian_adam_lr0.001/best_model.pth
```

## Platform-Specific Issues

### Windows

#### Issue: setup.sh won't run
**Solution**: Use manual installation:
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Issue: Multiprocessing errors
**Solution**: 
```bash
python main.py --num-workers 0
```

### macOS

#### Issue: MPS (Metal) errors with M1/M2
**Solution**:
```bash
# Force CPU
python main.py --cpu

# Or update PyTorch to latest version
pip install --upgrade torch torchvision
```

### Linux

#### Issue: "Too many open files"
**Solution**:
```bash
# Reduce number of workers
python main.py --num-workers 1

# Or increase file limit
ulimit -n 4096
```

## Debugging Tips

### Check Installation
```python
python -c "import torch; import torchvision; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Test Individual Components
```bash
# Test U-Net
cd src && python unet.py

# Test dataset
cd src && python dataset.py

# Test training (without actually training)
cd src && python train.py --help
```

### Verbose Output
```bash
# Run with Python's verbose mode
python -v main.py --mode train --epochs 1
```

### Memory Profiling
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## Getting Help

If you're still stuck:

1. **Check error message carefully** - Often tells you exactly what's wrong
2. **Search for the exact error** on Google or Stack Overflow
3. **Verify file paths** - Use absolute paths when in doubt
4. **Check Python version** - Needs Python 3.8+
   ```bash
   python --version
   ```
5. **Reinstall dependencies**:
   
   **With Conda (Recommended)**:
   ```bash
   # Remove and recreate environment
   conda deactivate
   conda env remove -n image_denoising
   conda create -n image_denoising python=3.10 -y
   conda activate image_denoising
   pip install -r requirements.txt
   ```
   
   **With venv**:
   ```bash
   pip uninstall -y torch torchvision
   pip install -r requirements.txt
   ```

## Quick Diagnostics

Run this to check your setup:
```python
import sys
import torch
import torchvision
import numpy as np
import matplotlib

print("="*50)
print("System Diagnostics")
print("="*50)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("="*50)
print("✅ All imports successful!")
```

Save this as `check_setup.py` and run it to verify everything works.

## Still Having Issues?

Make sure you:
- [ ] Activated conda environment (`conda activate image_denoising`) or venv
- [ ] Installed all requirements
- [ ] Running from project root directory
- [ ] Have enough disk space (need ~5GB)
- [ ] Have enough RAM (need ~4GB minimum)
- [ ] Using Python 3.8 or higher

**Most common fix with Conda (Recommended)**:
```bash
# Remove the environment completely
conda deactivate
conda env remove -n image_denoising -y

# Recreate from scratch
conda create -n image_denoising python=3.10 -y
conda activate image_denoising
pip install --upgrade pip
pip install -r requirements.txt
```

**Alternative fix with venv**:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
