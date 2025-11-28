# Image Denoising Using U-Net

A PyTorch implementation of image denoising using U-Net architecture, trained on the CIFAR-10 dataset. This project replicates the work described in "Image Denoising Using a U-net" by Paavani Dua.

## Overview

This project demonstrates the effectiveness of deep convolutional neural networks (CNNs) for image denoising tasks. Unlike traditional denoising techniques such as spatial filtering, wavelet thresholding, and transform domain filtering, the U-Net architecture provides:

- Better denoising results with preserved edge information
- More computational efficiency
- Automatic feature learning without manual parameter tuning

## Features

- **U-Net Architecture**: Implementation of the classic U-Net with encoder-decoder structure and skip connections
- **Multiple Noise Types**: Support for Gaussian, Poisson, and Salt & Pepper noise
- **Multiple Optimizers**: Comparison of Adam, RMSprop, and SGD optimizers
- **Comprehensive Evaluation**: PSNR (Peak Signal-to-Noise Ratio) metrics and visualizations
- **Flexible Training**: Configurable hyperparameters including learning rate, batch size, epochs, and dropout
- **Weights & Biases Integration**: Experiment tracking, visualization, and model versioning with wandb

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

### Training

Train a model with default settings (Gaussian noise, Adam optimizer):
```bash
python main.py --mode train
```

Train with custom parameters:
```bash
python main.py --mode train \
    --epochs 10 \
    --batch-size 16 \
    --lr 0.001 \
    --optimizer adam \
    --noise-type gaussian \
    --noise-param 0.05 \
    --dropout 0.2
```

### Available Options

**Training Parameters:**
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
