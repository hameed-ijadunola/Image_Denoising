#!/bin/bash
# Quick start script for Image Denoising project

echo "=================================="
echo "Image Denoising Setup (Conda)"
echo "=================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Conda version:"
conda --version

# Environment name
ENV_NAME="image_denoising"

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Using existing environment..."
        conda activate ${ENV_NAME}
        pip install -r requirements.txt
        echo ""
        echo "=================================="
        echo "Setup complete!"
        echo "=================================="
        echo ""
        echo "Environment updated. To start training:"
        echo "  conda activate ${ENV_NAME}"
        echo "  python main.py --mode train --epochs 5"
        echo ""
        exit 0
    fi
fi

# Create conda environment
echo "Creating conda environment '${ENV_NAME}' with Python 3.10..."
conda create -n ${ENV_NAME} python=3.10 -y

# Activate environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "To start training:"
echo "  conda activate ${ENV_NAME}"
echo "  python main.py --mode train --epochs 5"
echo ""
echo "For help:"
echo "  python main.py --help"
echo ""
echo "To deactivate environment:"
echo "  conda deactivate"
echo ""
