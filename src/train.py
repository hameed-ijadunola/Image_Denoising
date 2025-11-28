"""
Training script for U-Net image denoising
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json
from datetime import datetime
from .utils import calculate_all_metrics
import wandb
import lpips


class Trainer:
    """
    Trainer class for U-Net denoising model
    """

    def __init__(
        self,
        model,
        device,
        train_loader,
        test_loader,
        optimizer_name="adam",
        lr=0.001,
        save_dir="./models",
        use_wandb=False,
        wandb_config=None,
        model_metadata=None,
        dataset_metadata=None,
    ):
        """
        Args:
            model: U-Net model
            device: torch device (cpu or cuda)
            train_loader: Training data loader
            test_loader: Test data loader
            optimizer_name: Name of optimizer ('adam', 'rmsprop', 'sgd')
            lr: Learning rate
            save_dir: Directory to save models
            use_wandb: Whether to use Weights & Biases logging
            wandb_config: Dictionary with wandb configuration options
            model_metadata: Dictionary with model information (name, type, etc.)
            dataset_metadata: Dictionary with dataset information (name, type, etc.)
        """
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.save_dir = save_dir
        self.lr = lr
        self.use_wandb = use_wandb

        # Store metadata
        self.model_metadata = model_metadata or {}
        self.dataset_metadata = dataset_metadata or {}

        # Wandb configuration
        if self.use_wandb and wandb_config is None:
            wandb_config = {}
        self.wandb_config = wandb_config or {}

        # Loss function (MSE for denoising)
        self.criterion = nn.MSELoss()

        # Setup optimizer
        self.optimizer_name = optimizer_name.lower()
        if self.optimizer_name == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        elif self.optimizer_name == "rmsprop":
            self.optimizer = optim.RMSprop(model.parameters(), lr=lr)
        elif self.optimizer_name == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Training history
        self.train_losses = []
        self.test_losses = []
        self.test_metrics = []  # Store metrics for each epoch
        self.best_loss = float("inf")

        # Initialize LPIPS model
        self.lpips_model = lpips.LPIPS(net="alex").to(device)

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (noisy_images, clean_images) in enumerate(pbar):
            noisy_images = noisy_images.to(self.device)
            clean_images = clean_images.to(self.device)

            # Forward pass
            outputs = self.model(noisy_images)
            loss = self.criterion(outputs, clean_images)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

            # Log to wandb
            if self.use_wandb and self.wandb_config.get("log_interval", 1) > 0:
                if batch_idx % self.wandb_config.get("log_interval", 1) == 0:
                    wandb.log(
                        {
                            "batch_loss": loss.item(),
                            "batch": batch_idx
                            + len(self.train_loader) * (len(self.train_losses)),
                        }
                    )

        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """Validate on test set"""
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for noisy_images, clean_images in tqdm(self.test_loader, desc="Validating"):
                noisy_images = noisy_images.to(self.device)
                clean_images = clean_images.to(self.device)

                outputs = self.model(noisy_images)
                loss = self.criterion(outputs, clean_images)
                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(self.test_loader)
        return avg_loss

    def calculate_validation_metrics(self, num_batches=5):
        """
        Calculate comprehensive metrics on validation set

        Args:
            num_batches: Number of batches to evaluate (to save time)

        Returns:
            Dictionary with average metrics
        """
        self.model.eval()

        # Initialize metric lists
        metrics_lists = {
            "psnr": [],
            "ssim": [],
            "mse": [],
            "mae": [],
            "lpips": [],
        }

        with torch.no_grad():
            for batch_idx, (noisy_images, clean_images) in enumerate(self.test_loader):
                if batch_idx >= num_batches:
                    break

                noisy_images = noisy_images.to(self.device)
                clean_images = clean_images.to(self.device)

                # Denoise images
                denoised_images = self.model(noisy_images)

                # Calculate metrics for each image in batch
                for i in range(noisy_images.size(0)):
                    metrics = calculate_all_metrics(
                        denoised_images[i], clean_images[i], self.lpips_model
                    )

                    # Append to lists
                    for key in metrics:
                        if metrics[key] is not None:
                            metrics_lists[key].append(metrics[key])

        # Calculate averages
        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in metrics_lists.items()
            if values
        }

        return avg_metrics

    def save_checkpoint(self, epoch, is_best=False, is_last=False):
        """Save model checkpoint with metadata"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "test_metrics": self.test_metrics,
            "optimizer_name": self.optimizer_name,
            "lr": self.lr,
            # Add metadata for tracking
            "model_metadata": self.model_metadata,
            "dataset_metadata": self.dataset_metadata,
            "model_info": self.model.get_info()
            if hasattr(self.model, "get_info")
            else {},
        }

        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (loss: {self.test_losses[-1]:.5f})")

        # Save last checkpoint (overwrite each time)
        if is_last:
            last_path = os.path.join(self.save_dir, "last_checkpoint.pth")
            torch.save(checkpoint, last_path)

    def train(self, num_epochs, log_metrics_every=5):
        """
        Train the model for specified number of epochs

        Args:
            num_epochs: Number of epochs to train
            log_metrics_every: Calculate and log comprehensive metrics every N epochs
        """
        print(f"\n{'=' * 60}")
        print("Training Configuration:")
        print(f"  Optimizer: {self.optimizer_name.upper()}")
        print(f"  Learning Rate: {self.lr}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Device: {self.device}")
        print(f"  Wandb Logging: {'Enabled' if self.use_wandb else 'Disabled'}")
        print(f"  Metrics Logging: Every {log_metrics_every} epochs")
        print(f"{'=' * 60}\n")

        # Log model and dataset metadata to wandb at the start of training
        if self.use_wandb:
            metadata_log = {}

            # Add model metadata
            for key, value in self.model_metadata.items():
                metadata_log[f"model/{key}"] = value

            # Add dataset metadata
            for key, value in self.dataset_metadata.items():
                metadata_log[f"dataset/{key}"] = value

            # Log to wandb
            if metadata_log:
                wandb.log(metadata_log)
                print("✓ Logged model and dataset metadata to wandb\n")

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            test_loss = self.validate()
            self.test_losses.append(test_loss)

            print(f"Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f}")

            # Calculate comprehensive metrics periodically
            metrics = None
            if epoch % log_metrics_every == 0 or epoch == num_epochs:
                print("Calculating comprehensive metrics...")
                metrics = self.calculate_validation_metrics()
                self.test_metrics.append({"epoch": epoch, "metrics": metrics})

                # Print metrics
                print("\nValidation Metrics:")
                for key, value in metrics.items():
                    print(f"  {key.upper()}: {value:.4f}")

            # Log to wandb
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "learning_rate": self.lr,
                }

                # Add metrics if calculated
                if metrics:
                    for key, value in metrics.items():
                        log_dict[f"val_{key}"] = value

                wandb.log(log_dict)

            # Save checkpoint
            is_best = test_loss < self.best_loss
            if is_best:
                self.best_loss = test_loss

            # Save best and last checkpoint only
            self.save_checkpoint(epoch, is_best=is_best, is_last=True)

            # Log best model to wandb
            if self.use_wandb and is_best and self.wandb_config.get("log_model", True):
                best_path = os.path.join(self.save_dir, "best_model.pth")
                wandb.save(best_path)

        # Save training history
        history_path = os.path.join(self.save_dir, "training_history.json")

        # Ensure all numeric values are JSON-serializable Python types
        history_data = {
            "train_losses": [float(x) for x in self.train_losses],
            "test_losses": [float(x) for x in self.test_losses],
            "test_metrics": self.test_metrics,  # Already converted in calculate_all_metrics
            "optimizer": self.optimizer_name,
            "lr": float(self.lr),
            "best_loss": float(self.best_loss),
        }

        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=4)

        print(f"\n{'=' * 60}")
        print("Training Complete!")
        print(f"  Best Test Loss: {self.best_loss:.5f}")
        print(f"  Final Train Loss: {self.train_losses[-1]:.5f}")

        # Print final metrics if available
        if self.test_metrics:
            final_metrics = self.test_metrics[-1]["metrics"]
            print(f"\nFinal Validation Metrics:")
            for key, value in final_metrics.items():
                print(f"  {key.upper()}: {value:.4f}")

        print(f"\n  Model saved to: {self.save_dir}")
        print(f"{'=' * 60}\n")

        return self.train_losses, self.test_losses


def load_checkpoint(model, checkpoint_path, device):
    """
    Load a saved checkpoint

    Args:
        model: U-Net model
        checkpoint_path: Path to checkpoint file
        device: torch device

    Returns:
        model, optimizer, epoch, train_losses, test_losses
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Recreate optimizer
    optimizer_name = checkpoint.get("optimizer_name", "adam")
    lr = checkpoint.get("lr", 0.001)

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (
        model,
        optimizer,
        checkpoint["epoch"],
        checkpoint["train_losses"],
        checkpoint["test_losses"],
    )


if __name__ == "__main__":
    from unet import UNet
    from dataset import get_dataloaders

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader = get_dataloaders(batch_size=16, noise_type="gaussian")

    # Create model
    model = UNet(n_channels=3, n_classes=3, dropout_rate=0.2)

    # Create trainer
    trainer = Trainer(model, device, train_loader, test_loader, optimizer_name="adam")

    # Train
    trainer.train(num_epochs=1)
