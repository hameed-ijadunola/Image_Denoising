"""
Custom trainer for R3L model using reinforcement learning.

This trainer implements the A3C training algorithm as described in the paper,
which is different from standard supervised learning.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict
import wandb
from tqdm import tqdm

from src.r3l import R3L
from src.utils import calculate_psnr, calculate_ssim


class R3LTrainer:
    """
    Trainer for R3L model using reinforcement learning.

    Unlike standard supervised learning, R3L uses:
    - Separate optimizers for policy and value networks
    - Stochastic reward signals
    - Multi-stage recurrent training
    """

    def __init__(
        self,
        model: R3L,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate_policy: float = 1e-4,
        learning_rate_value: float = 1e-3,
        num_epochs: int = 100,
        save_dir: str = "models/r3l",
        use_wandb: bool = True,
        **kwargs,
    ):
        """
        Args:
            model: R3L model instance
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to train on (cpu/cuda)
            learning_rate_policy: Learning rate for policy network
            learning_rate_value: Learning rate for value network
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            use_wandb: Whether to use wandb for logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.use_wandb = use_wandb

        # Separate optimizers for policy and value networks
        # Policy network includes encoder + policy head
        policy_params = list(model.encoder.parameters()) + list(
            model.policy_net.parameters()
        )
        self.optimizer_policy = torch.optim.Adam(policy_params, lr=learning_rate_policy)

        # Value network parameters
        value_params = list(model.value_net.parameters())
        self.optimizer_value = torch.optim.Adam(value_params, lr=learning_rate_value)

        # Training state
        self.current_epoch = 0
        self.best_val_psnr = 0.0
        self.history = {
            "train_policy_loss": [],
            "train_value_loss": [],
            "train_reward": [],
            "train_psnr": [],
            "val_psnr": [],
            "val_ssim": [],
        }

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch using reinforcement learning.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_reward = 0.0
        total_psnr = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}"
        )

        for batch in pbar:
            noisy, clean = batch
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # Perform one training step
            metrics = self.model.train_step(
                noisy_images=noisy,
                clean_images=clean,
                optimizer_policy=self.optimizer_policy,
                optimizer_value=self.optimizer_value,
                stage=0,  # Single-stage training
            )

            # Accumulate metrics
            total_policy_loss += metrics["policy_loss"]
            total_value_loss += metrics["value_loss"]
            total_reward += metrics["mean_reward"]
            total_psnr += metrics["psnr_after"]
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "p_loss": f"{metrics['policy_loss']:.4f}",
                    "v_loss": f"{metrics['value_loss']:.4f}",
                    "reward": f"{metrics['mean_reward']:.4f}",
                    "psnr": f"{metrics['psnr_after']:.2f}",
                }
            )

        # Average metrics
        avg_metrics = {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "mean_reward": total_reward / num_batches,
            "train_psnr": total_psnr / num_batches,
        }

        return avg_metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                noisy, clean = batch
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                # Inference mode: full recursive denoising
                denoised = self.model(noisy, training=False)

                # Calculate metrics
                psnr = calculate_psnr(denoised, clean)
                ssim = calculate_ssim(denoised, clean)

                total_psnr += float(psnr)
                total_ssim += float(ssim)
                num_batches += 1

        avg_metrics = {
            "val_psnr": total_psnr / num_batches,
            "val_ssim": total_ssim / num_batches,
        }

        return avg_metrics

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            filename: Name of checkpoint file
            is_best: Whether this is the best model so far
        """
        import os

        os.makedirs(self.save_dir, exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_policy_state_dict": self.optimizer_policy.state_dict(),
            "optimizer_value_state_dict": self.optimizer_value.state_dict(),
            "best_val_psnr": self.best_val_psnr,
            "history": self.history,
        }

        filepath = f"{self.save_dir}/{filename}"
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")

        if is_best:
            best_filepath = f"{self.save_dir}/best_model.pth"
            torch.save(checkpoint, best_filepath)
            print(f"Saved best model to {best_filepath}")

    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_policy.load_state_dict(checkpoint["optimizer_policy_state_dict"])
        self.optimizer_value.load_state_dict(checkpoint["optimizer_value_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_psnr = checkpoint["best_val_psnr"]
        self.history = checkpoint["history"]

        print(f"Loaded checkpoint from {filepath} (epoch {self.current_epoch})")

    def train(self):
        """
        Run full training loop.
        """
        print(f"\nStarting R3L training for {self.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Policy LR: {self.optimizer_policy.param_groups[0]['lr']}")
        print(f"Value LR: {self.optimizer_value.param_groups[0]['lr']}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update history
            self.history["train_policy_loss"].append(train_metrics["policy_loss"])
            self.history["train_value_loss"].append(train_metrics["value_loss"])
            self.history["train_reward"].append(train_metrics["mean_reward"])
            self.history["train_psnr"].append(train_metrics["train_psnr"])
            self.history["val_psnr"].append(val_metrics["val_psnr"])
            self.history["val_ssim"].append(val_metrics["val_ssim"])

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(
                f"Train - Policy Loss: {train_metrics['policy_loss']:.4f}, "
                f"Value Loss: {train_metrics['value_loss']:.4f}, "
                f"Reward: {train_metrics['mean_reward']:.4f}, "
                f"PSNR: {train_metrics['train_psnr']:.2f}"
            )
            print(
                f"Val   - PSNR: {val_metrics['val_psnr']:.2f}, "
                f"SSIM: {val_metrics['val_ssim']:.4f}"
            )

            # Log to wandb
            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/policy_loss": train_metrics["policy_loss"],
                        "train/value_loss": train_metrics["value_loss"],
                        "train/reward": train_metrics["mean_reward"],
                        "train/psnr": train_metrics["train_psnr"],
                        "val/psnr": val_metrics["val_psnr"],
                        "val/ssim": val_metrics["val_ssim"],
                    }
                )

            # Save checkpoints
            is_best = val_metrics["val_psnr"] > self.best_val_psnr
            if is_best:
                self.best_val_psnr = val_metrics["val_psnr"]

            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth", is_best=is_best)

        print(f"\nTraining complete! Best validation PSNR: {self.best_val_psnr:.2f}")

        return self.history
