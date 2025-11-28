# Conda Quick Reference

This project uses **Conda** for environment management. Here's a quick reference guide.

## Setup Commands

### First Time Setup
```bash
# Run the automated setup script
./setup.sh
```

This creates a conda environment named `image_denoising` with Python 3.10.

### Manual Setup
```bash
# Create environment
conda create -n image_denoising python=3.10 -y

# Activate environment
conda activate image_denoising

# Install dependencies
pip install -r requirements.txt
```

## Daily Usage

### Activate Environment (Required before running scripts)
```bash
conda activate image_denoising
```

### Deactivate Environment
```bash
conda deactivate
```

### Check Active Environment
```bash
conda env list
# or
conda info --envs
```

The active environment will have an asterisk (*) next to it.

## Package Management

### Install Additional Packages
```bash
# Activate environment first
conda activate image_denoising

# Install with pip
pip install package_name

# Or install with conda (often faster for scientific packages)
conda install package_name
```

### List Installed Packages
```bash
conda activate image_denoising
conda list
# or
pip list
```

### Update Packages
```bash
conda activate image_denoising
pip install --upgrade package_name
```

## Environment Management

### Remove Environment (Clean Reinstall)
```bash
# Deactivate if currently active
conda deactivate

# Remove environment
conda env remove -n image_denoising -y

# Recreate
./setup.sh
```

### Export Environment (Share with Others)
```bash
conda activate image_denoising
conda env export > environment.yml
```

### Create from Exported Environment
```bash
conda env create -f environment.yml
```

### Clone Environment
```bash
conda create --name new_env --clone image_denoising
```

## PyTorch-Specific

### Install PyTorch with CUDA (GPU Support)
```bash
conda activate image_denoising

# For CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Install PyTorch CPU-Only (Faster, No GPU)
```bash
conda activate image_denoising
conda install pytorch torchvision cpuonly -c pytorch
```

### Check PyTorch Installation
```bash
conda activate image_denoising
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Troubleshooting

### "conda: command not found"
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Initialize conda
conda init bash
source ~/.bashrc
```

### "conda activate" not working
```bash
# Initialize conda for your shell
conda init bash  # or zsh, fish, etc.

# Restart terminal or reload
source ~/.bashrc

# Try again
conda activate image_denoising
```

### Environment conflicts or errors
```bash
# Clean reinstall
conda deactivate
conda env remove -n image_denoising -y
conda clean --all -y
./setup.sh
```

### Slow package installation
```bash
# Use mamba (faster conda alternative)
conda install mamba -c conda-forge
mamba install package_name

# Or use libmamba solver (conda 23.10+)
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

## Useful Commands

### See conda configuration
```bash
conda config --show
```

### Update conda itself
```bash
conda update -n base -c defaults conda
```

### Clean up cached packages
```bash
conda clean --all -y
```

### Get help
```bash
conda --help
conda install --help
conda env --help
```

## Project-Specific Workflow

### Complete Workflow Example
```bash
# 1. Setup (first time only)
./setup.sh

# 2. Activate environment (every time you work)
conda activate image_denoising

# 3. Verify setup
python check_setup.py

# 4. Run your code
python demo.py
# or
python main.py --mode train --epochs 5

# 5. Deactivate when done
conda deactivate
```

## Why Conda vs venv?

**Advantages of Conda:**
- ✅ Better package dependency resolution
- ✅ Can install non-Python packages (CUDA, compilers, etc.)
- ✅ Faster installation for scientific packages
- ✅ Better isolation (completely separate Python installations)
- ✅ Cross-platform consistency
- ✅ Easy Python version management

**When to use venv instead:**
- You want a lighter solution
- You only need Python packages
- You have a specific venv workflow

## Resources

- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html)
- [Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

---

**Quick Tip**: Add this to your shell configuration to always show the active conda environment in your prompt:
```bash
# Add to ~/.bashrc or ~/.zshrc
conda config --set changeps1 true
```
