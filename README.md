# Image Denoising - Multiple Approaches

A comprehensive PyTorch implementation of image denoising using both deep learning and traditional methods, trained on the CIFAR-10 dataset.

## Overview

This project demonstrates various approaches to image denoising tasks, from traditional techniques to state-of-the-art deep learning and reinforcement learning methods.

## Implemented Models

### Deep Learning Models

1. **U-Net** - Classic encoder-decoder architecture with skip connections
   - Paper: Ronneberger et al., 2015
   - Better denoising with preserved edge information
   - Efficient with automatic feature learning

2. **R3L** - Residual Recovery using Reinforcement Learning ⭐ NEW
   - Paper: Zhang et al., 2021
   - Uses A3C (Actor-Critic) framework
   - Learns pixel-wise denoising through reinforcement learning
   - Better generalization to noise level variations
   - See [docs/R3L_implementation.md](docs/R3L_implementation.md) for details

### Traditional Methods

3. **Non-Local Means (NLM)** - Non-local averaging denoising
4. **Bilateral Filter** - Edge-preserving spatial filtering
5. **Wavelet Denoising** - Transform domain denoising with soft thresholding

## Features

- **Multiple Models**: U-Net, R3L, NLM, Bilateral, Wavelet denoising
- **Multiple Noise Types**: Support for Gaussian, Poisson, and Salt & Pepper noise
- **Multiple Optimizers**: Comparison of Adam, RMSprop, and SGD optimizers
- **Comprehensive Metrics**: PSNR, SSIM, MSE, MAE, and LPIPS for thorough quality evaluation
- **Flexible Training**: Configurable hyperparameters including learning rate, batch size, epochs, and dropout
- **Weights & Biases Integration**: Experiment tracking, visualization, and model versioning with wandb
- **Extensible Registry**: Easy addition of new models through MODEL_REGISTRY

## Image Quality Metrics

This project evaluates denoising performance using **5 comprehensive metrics**:

1. **PSNR** (Peak Signal-to-Noise Ratio) - Classic metric, measures pixel-wise accuracy
2. **SSIM** (Structural Similarity Index) - Measures structural similarity, better perceptual correlation
3. **MSE** (Mean Squared Error) - Average squared pixel differences
4. **MAE** (Mean Absolute Error) - Average absolute pixel differences
5. **LPIPS** (Learned Perceptual Image Patch Similarity) - Deep learning-based perceptual metric

See [METRICS_GUIDE.md](METRICS_GUIDE.md) for detailed information about each metric, interpretation guidelines, and usage examples.

## Project Structure

```
Image_Denoising/
├── docs/
│   └── unet_denoising.pdf          # Original paper/documentation
├── src/
│   ├── unet.py                     # U-Net model architecture
│   ├── dataset.py                  # CIFAR-10 dataset with noise injection
│   ├── train.py                    # Training utilities and Trainer class
│   └── utils.py                    # Evaluation and visualization functions
├── models/                         # Saved model checkpoints
├── results/                        # Training results and visualizations
├── main.py                         # Main execution script
└── requirements.txt                # Project dependencies
```

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

This will create a conda environment named `image_denoising` and install all dependencies.

### Option 2: Manual Setup with Conda

```bash
# Create conda environment
conda create -n image_denoising python=3.10 -y

# Activate environment
conda activate image_denoising

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Manual Setup with venv (Alternative)

```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: After installation, always activate the environment before running scripts:
```bash
conda activate image_denoising  # For conda
# or
source venv/bin/activate        # For venv
```

## Usage

### Quick Start with Weights & Biases

For experiment tracking and visualization:

```bash
# First time: login to wandb
wandb login

# Train with wandb tracking
python main.py --use-wandb --epochs 10

# Run experiments with wandb
python run_experiment.py optimizer_comparison --use-wandb
```

📊 **See [WANDB_GUIDE.md](WANDB_GUIDE.md) for complete wandb integration documentation.**

### Training Different Models

**Train U-Net (default):**
```bash
python main.py --mode train \
    --model unet \
    --dataset cifar10 \
    --epochs 10 \
    --batch-size 16 \
    --lr 0.001
```

**Train R3L (Reinforcement Learning):**
```bash
python main.py --mode train \
    --model r3l \
    --dataset cifar10 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.0001 \
    --noise-type gaussian \
    --noise-param 25
```

**Use Traditional Methods (No Training):**
```bash
# Non-Local Means
python main.py --mode eval \
    --model nlm \
    --dataset cifar10

# Bilateral Filter
python main.py --mode eval \
    --model bilateral \
    --dataset cifar10

# Wavelet Denoising
python main.py --mode eval \
    --model wavelet \
    --dataset cifar10
```

### Available Models

Use the `--model` flag to select a denoising model:

| Model | Key | Type | Training Required | Description |
|-------|-----|------|-------------------|-------------|
| U-Net | `unet` | Deep Learning | Yes | Encoder-decoder with skip connections |
| R3L | `r3l` | Deep Learning (RL) | Yes | Reinforcement learning with A3C |
| Non-Local Means | `nlm` | Traditional | No | Non-local averaging filter |
| Bilateral Filter | `bilateral` | Traditional | No | Edge-preserving spatial filter |
| Wavelet | `wavelet` | Traditional | No | Transform domain denoising |

**R3L Specific Notes:**
- Uses custom trainer with separate policy and value optimizers
- Recommended learning rate: 1e-4 for policy, 1e-3 for value (automatic)
- Requires more epochs (50+) than supervised models
- See [docs/R3L_implementation.md](docs/R3L_implementation.md) for details

### Training Parameters
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size`: Batch size for training (default: 16)
- `--lr`: Learning rate (default: 0.001)
- `--optimizer`: Optimizer choice - `adam`, `rmsprop`, or `sgd` (default: adam)
- `--dropout`: Dropout rate (default: 0.2)

**Noise Parameters:**
- `--noise-type`: Type of noise - `gaussian`, `poisson`, or `salt_pepper` (default: gaussian)
- `--noise-param`: Noise parameter value (default: 0.05)
  - For Gaussian: sigma (standard deviation)
  - For Poisson: lambda (scaling factor)
  - For Salt & Pepper: probability

**Other Options:**
- `--cpu`: Force CPU usage even if GPU is available
- `--num-workers`: Number of data loading workers (default: 2)
- `--save-dir`: Directory to save models (default: ./models)
- `--results-dir`: Directory to save results (default: ./results)

**Weights & Biases Options:**
- `--use-wandb`: Enable Weights & Biases logging
- `--wandb-project`: Project name (default: image-denoising-unet)
- `--wandb-entity`: Entity/team name (optional)
- `--wandb-log-interval`: Log metrics every N batches (default: 1)
- `--wandb-log-images`: Log sample images to wandb
- `--wandb-log-model`: Save model checkpoints to wandb

### Evaluation

Evaluate a saved model:
```bash
python main.py --mode eval \
    --checkpoint ./models/gaussian_adam_lr0.001/best_model.pth \
    --noise-type gaussian \
    --noise-param 0.05
```

## Experiments

### Optimizer Comparison

The project allows comparison of different optimizers. Based on the original paper:

1. **Adam** - Typically provides the lowest loss
2. **RMSprop** - Good alternative with adaptive learning rate
3. **SGD** - Traditional optimizer with momentum

Run experiments with different optimizers:
```bash
# Adam
python main.py --mode train --optimizer adam --epochs 5

# RMSprop
python main.py --mode train --optimizer rmsprop --epochs 5

# SGD
python main.py --mode train --optimizer sgd --epochs 5
```

### Noise Type Comparison

Compare denoising performance on different noise types:
```bash
# Gaussian noise
python main.py --mode train --noise-type gaussian --noise-param 0.05

# Poisson noise
python main.py --mode train --noise-type poisson --noise-param 1.0

# Salt and Pepper noise
python main.py --mode train --noise-type salt_pepper --noise-param 0.05
```

## Results

After training, the following outputs are generated:

1. **Model Checkpoints**: Saved in `./models/`
   - `best_model.pth`: Best performing model based on validation loss
   - `checkpoint_epoch_X.pth`: Checkpoint for each epoch
   - `training_history.json`: Training and validation loss history

2. **Visualizations**: Saved in `./results/`
   - Loss curves over epochs
   - PSNR comparison (noisy vs denoised)
   - Sample denoised images

3. **Metrics**:
   - PSNR improvement in dB
   - Training and validation loss
   - Per-sample PSNR values

### Expected Performance

Based on the original paper with 1 epoch of training:
- **Noisy PSNR**: ~20-25 dB (depends on noise level)
- **Denoised PSNR**: ~28-32 dB (improvement of 5-10 dB)
- **Adam optimizer loss**: ~0.003-0.005 (after 5 iterations)

## Model Architecture

The U-Net consists of:
- **Encoder (Contracting Path)**: 4 downsampling blocks with double convolution
- **Bottleneck**: Deepest layer with dropout regularization
- **Decoder (Expanding Path)**: 4 upsampling blocks with skip connections
- **Output**: 1x1 convolution to produce denoised image

Key features:
- Skip connections between encoder and decoder for preserving spatial information
- Batch normalization for stable training
- Dropout for regularization
- Total parameters: ~31 million

## Dataset

**CIFAR-10**: 
- Training: 50,000 images
- Testing: 10,000 images
- Image size: 32×32 pixels
- 3 color channels (RGB)
- 10 classes (not used for denoising task)

The dataset is automatically downloaded on first run.

## Performance Tips

1. **GPU Usage**: Training on GPU significantly speeds up the process
   ```bash
   # Check if GPU is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Batch Size**: Increase for faster training if you have sufficient memory
   ```bash
   python main.py --batch-size 32  # If memory allows
   ```

3. **Number of Workers**: Increase for faster data loading
   ```bash
   python main.py --num-workers 4
   ```

4. **Training Time**: 
   - CPU: ~1 hour per epoch
   - GPU: ~5-10 minutes per epoch

## Future Improvements

Based on the original paper's discussion:

1. **Traditional Method Comparison**: Compare with BM3D/CBM3D algorithms
2. **Hyperparameter Tuning**: Extended grid search with more epochs
3. **Super-Resolution**: Extend to image super-resolution tasks
4. **Different Datasets**: Test on medical imaging datasets
5. **Advanced Architectures**: Implement residual U-Net or attention mechanisms

## 🆕 Model and Dataset Tracking System

This project now includes a comprehensive tracking system for managing multiple models, datasets, and techniques!

### Key Features

- ✅ **Model Registry**: Track deep learning models (U-Net, DnCNN) and traditional methods (NLM, Bilateral, Wavelet)
- ✅ **Dataset Registry**: Manage multiple datasets (CIFAR-10, BSD68, custom datasets)
- ✅ **Experiment Tracking**: Complete metadata for all experiments
- ✅ **Easy Extensibility**: Add new models/datasets by editing a config file

### Quick Start

```python
from config import MODEL_REGISTRY, DATASET_REGISTRY
from src.registry import get_model_from_registry, get_dataset_from_registry

# Get a model
model = get_model_from_registry("unet", MODEL_REGISTRY)

# Get a dataset
dataset = get_dataset_from_registry("cifar10", DATASET_REGISTRY, train=True)
```

### Documentation

- **[QUICKSTART_TRACKING.md](QUICKSTART_TRACKING.md)** - Quick introduction and basic usage
- **[TRACKING_SYSTEM.md](TRACKING_SYSTEM.md)** - Complete system overview
- **[REGISTRY_GUIDE.md](REGISTRY_GUIDE.md)** - Detailed examples for adding models and datasets
- **[src/traditional/template.py](src/traditional/template.py)** - Template for traditional methods

### Test the System

```bash
python test_registry.py
```

All tests should pass with: `🎉 All tests PASSED!`

### Adding New Models

**Deep Learning** (requires training):
1. Create model class inheriting from `DeepLearningModel`
2. Add to `MODEL_REGISTRY` in `config.py`
3. Use with the registry system

**Traditional Methods** (no training):
1. Create method class inheriting from `TraditionalMethod`
2. Add to `MODEL_REGISTRY` in `config.py`
3. Use directly (no training needed!)

See `REGISTRY_GUIDE.md` for complete examples.

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation". arXiv:1505.04597
2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet Classification with Deep Convolutional Neural Networks". NIPS.
3. Dua, P. "Image Denoising Using a U-net". Stanford University.

## License

This project is for educational purposes.

## Acknowledgments

- Original paper by Paavani Dua (Stanford University)
- U-Net architecture by Ronneberger et al.
- CIFAR-10 dataset by Alex Krizhevsky

## Contact

For questions or issues, please open an issue on the repository.

---

**Note**: This implementation is designed for educational purposes and demonstrates the effectiveness of U-Net for image denoising tasks. The training hyperparameters can be adjusted based on available computational resources.
