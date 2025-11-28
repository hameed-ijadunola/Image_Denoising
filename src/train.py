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
        """
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.save_dir = save_dir
        self.lr = lr

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
        self.best_loss = float("inf")

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0

        pbar = tqdm(self.train_loader, desc="Training")
        for noisy_images, clean_images in pbar:
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

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "optimizer_name": self.optimizer_name,
            "lr": self.lr,
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (loss: {self.test_losses[-1]:.5f})")

    def train(self, num_epochs):
        """
        Train the model for specified number of epochs

        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\n{'=' * 60}")
        print(f"Training Configuration:")
        print(f"  Optimizer: {self.optimizer_name.upper()}")
        print(f"  Learning Rate: {self.lr}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Device: {self.device}")
        print(f"{'=' * 60}\n")

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

            # Save checkpoint
            is_best = test_loss < self.best_loss
            if is_best:
                self.best_loss = test_loss

            self.save_checkpoint(epoch, is_best)

        # Save training history
        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(
                {
                    "train_losses": self.train_losses,
                    "test_losses": self.test_losses,
                    "optimizer": self.optimizer_name,
                    "lr": self.lr,
                    "best_loss": self.best_loss,
                },
                f,
                indent=4,
            )

        print(f"\n{'=' * 60}")
        print(f"Training Complete!")
        print(f"  Best Test Loss: {self.best_loss:.5f}")
        print(f"  Final Train Loss: {self.train_losses[-1]:.5f}")
        print(f"  Model saved to: {self.save_dir}")
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
