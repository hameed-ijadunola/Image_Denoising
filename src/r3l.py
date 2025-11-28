"""
R3L: Residual Recovery using Reinforcement Learning for Image Denoising

Implementation based on:
"Connecting Deep Reinforcement Learning to Recurrent Neural Networks
for Image Denoising via Residual Recovery"
Zhang et al., 2021

This implements an A3C (Asynchronous Advantage Actor-Critic) framework
with FCN encoder for pixel-wise residual recovery in image denoising.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from src.base_model import BaseDenoisingModel


class FCNEncoder(nn.Module):
    """
    Fully Convolutional Network encoder shared by policy and value networks.

    Architecture from Table 2 in the paper:
    - Conv+ReLU: 3x3, dilation=1, 64 channels
    - Conv+ReLU: 3x3, dilation=2, 64 channels
    - Conv+ReLU: 3x3, dilation=3, 64 channels
    - Conv+ReLU: 3x3, dilation=4, 64 channels
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=3, dilation=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor [B, C, H, W]

        Returns:
            Encoded features [B, 64, H, W]
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x


class PolicyNetwork(nn.Module):
    """
    Policy network π that outputs probability distribution over residual values.

    Architecture from Table 2:
    - Conv+ReLU: 3x3, dilation=3, 64 channels
    - Conv+ReLU+Softmax: 3x3, dilation=1, |A| channels
    """

    def __init__(self, action_space_size: int = 27):
        """
        Args:
            action_space_size: Size of discrete action space (default: 27 for [-13, 13])
        """
        super().__init__()

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=3, dilation=3)
        self.conv2 = nn.Conv2d(
            64, action_space_size, kernel_size=3, padding=1, dilation=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Encoded state features [B, 64, H, W]

        Returns:
            Action probabilities [B, |A|, H, W]
        """
        x = self.relu(self.conv1(state))
        x = self.conv2(x)
        # Apply softmax over action dimension
        x = F.softmax(x, dim=1)
        return x


class ValueNetwork(nn.Module):
    """
    Value network V that estimates expected long-term discounted rewards.

    Architecture from Table 2:
    - Conv+ReLU: 3x3, dilation=3, 64 channels
    - Conv: 3x3, dilation=1, 1 channel
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=3, dilation=3)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1, dilation=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Encoded state features [B, 64, H, W]

        Returns:
            Value estimates [B, 1, H, W]
        """
        x = self.relu(self.conv1(state))
        x = self.conv2(x)
        return x


class R3L(BaseDenoisingModel):
    """
    R3L: Residual Recovery using Reinforcement Learning

    This model implements the A3C framework for image denoising by learning
    to predict residuals pixel-wise through reinforcement learning.

    Key features:
    - FCN encoder shared between policy and value networks
    - Discrete action space for residual values
    - Recurrent inference with T stages
    - Trained with stochastic rewards (pixel-wise MSE reduction)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_stages: int = 5,
        action_range: Tuple[int, int] = (-13, 13),
        gamma: float = 0.95,
        **kwargs,
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            num_stages: Number of recursive denoising stages T (default: 5)
            action_range: Range of discrete residual values (default: [-13, 13])
            gamma: Discount factor for future rewards (default: 0.95)
        """
        super().__init__(
            name="R3L",
            model_type="deep_learning",
            description="Residual Recovery using Reinforcement Learning",
        )

        self.in_channels = in_channels
        self.num_stages = num_stages
        self.gamma = gamma

        # Define discrete action space
        self.action_min, self.action_max = action_range
        assert self.action_min < self.action_max, "Invalid action range"
        self.action_space_size = self.action_max - self.action_min + 1

        # Create action value lookup (discrete values normalized to [-1, 1])
        self.register_buffer(
            "action_values",
            torch.linspace(
                self.action_min / 255.0, self.action_max / 255.0, self.action_space_size
            ),
        )

        # Shared encoder
        self.encoder = FCNEncoder(in_channels)

        # Policy and value networks
        self.policy_net = PolicyNetwork(self.action_space_size)
        self.value_net = ValueNetwork()

    def encode_state(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode input image to state representation.

        Args:
            image: Input image [B, C, H, W]

        Returns:
            State features [B, 64, H, W]
        """
        return self.encoder(image)

    def get_policy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probability distribution from policy network.

        Args:
            state: State features [B, 64, H, W]

        Returns:
            Action probabilities [B, |A|, H, W]
        """
        return self.policy_net(state)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate from value network.

        Args:
            state: State features [B, 64, H, W]

        Returns:
            Value estimates [B, 1, H, W]
        """
        return self.value_net(state)

    def sample_action(
        self, policy: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy distribution.

        Args:
            policy: Action probabilities [B, |A|, H, W]
            training: If True, sample stochastically; if False, use greedy selection

        Returns:
            action_indices: Sampled action indices [B, 1, H, W]
            action_values: Corresponding residual values [B, 1, H, W]
        """
        B, A, H, W = policy.shape

        if training:
            # Stochastic sampling during training
            # Reshape for categorical sampling
            policy_flat = policy.permute(0, 2, 3, 1).reshape(-1, A)
            action_indices_flat = torch.multinomial(policy_flat, 1)
            action_indices = action_indices_flat.reshape(B, H, W, 1).permute(0, 3, 1, 2)
        else:
            # Greedy selection during inference
            action_indices = torch.argmax(policy, dim=1, keepdim=True)

        # Convert action indices to actual residual values
        action_values = self.action_values[action_indices.squeeze(1)].unsqueeze(1)

        return action_indices, action_values

    def forward(
        self,
        x: torch.Tensor,
        training: bool = False,
        ground_truth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through R3L model.

        During training, this is called once per stage to get policy and value.
        During inference, this runs the full recursive denoising process.

        Args:
            x: Noisy input image [B, C, H, W]
            training: Whether in training mode
            ground_truth: Clean image for computing rewards (training only)

        Returns:
            If training=False: Denoised image [B, C, H, W]
            If training=True: Returns last denoised image (use train_step for full training)
        """
        assert x.dim() == 4, f"Expected 4D tensor [B,C,H,W], got {x.dim()}D"
        assert x.shape[1] == self.in_channels, (
            f"Expected {self.in_channels} channels, got {x.shape[1]}"
        )

        if not training:
            # Inference mode: Run full recursive denoising with greedy action selection
            current_image = x.clone()

            for t in range(self.num_stages):
                # Encode current state
                state = self.encode_state(current_image)

                # Get policy
                policy = self.get_policy(state)

                # Greedy action selection
                _, residual = self.sample_action(policy, training=False)

                # Update image by adding residual
                current_image = current_image + residual

                # Clamp to valid range [0, 1]
                current_image = torch.clamp(current_image, 0.0, 1.0)

            return current_image
        else:
            # Training mode: Return components for one stage
            # The actual training loop should be in train_step or external trainer
            state = self.encode_state(x)
            policy = self.get_policy(state)
            value = self.get_value(state)

            # For compatibility, return a simple forward pass result
            _, residual = self.sample_action(policy, training=True)
            denoised = torch.clamp(x + residual, 0.0, 1.0)

            return denoised

    def compute_reward(
        self,
        current_image: torch.Tensor,
        next_image: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reward as defined in equation (5) of the paper:
        r_i^t = (x_i - I_i^{t-1})^2 - (x_i - I_i^t)^2

        This is the reduction in MSE per pixel.

        Args:
            current_image: Image before action [B, C, H, W]
            next_image: Image after action [B, C, H, W]
            ground_truth: Clean image [B, C, H, W]

        Returns:
            Pixel-wise rewards [B, 1, H, W]
        """
        # MSE before action
        mse_before = (ground_truth - current_image) ** 2

        # MSE after action
        mse_after = (ground_truth - next_image) ** 2

        # Reward is reduction in MSE
        reward = mse_before - mse_after

        # Average over channels to get per-pixel reward
        reward = reward.mean(dim=1, keepdim=True)

        return reward

    def train_step(
        self,
        noisy_images: torch.Tensor,
        clean_images: torch.Tensor,
        optimizer_policy: torch.optim.Optimizer,
        optimizer_value: torch.optim.Optimizer,
        stage: int = 0,
    ) -> dict:
        """
        Perform one training step for R3L using A3C algorithm.

        This implements the gradient calculations from equations (6) in the paper.

        Args:
            noisy_images: Noisy input images [B, C, H, W]
            clean_images: Ground truth clean images [B, C, H, W]
            optimizer_policy: Optimizer for policy network
            optimizer_value: Optimizer for value network
            stage: Current training stage (for multi-stage training)

        Returns:
            Dictionary with training metrics
        """
        assert noisy_images.shape == clean_images.shape
        assert 0 <= noisy_images.min() and noisy_images.max() <= 1.0
        assert 0 <= clean_images.min() and clean_images.max() <= 1.0

        self.train()

        # Current image state
        current_image = noisy_images.clone()

        # Encode state
        state = self.encode_state(current_image)

        # Get policy and value
        policy = self.get_policy(state)
        value = self.get_value(state)

        # Sample action
        action_indices, residual = self.sample_action(policy, training=True)

        # Apply action
        next_image = torch.clamp(current_image + residual, 0.0, 1.0)

        # Compute reward (equation 5)
        reward = self.compute_reward(current_image, next_image, clean_images)

        # Get next state and value for bootstrapping
        with torch.no_grad():
            next_state = self.encode_state(next_image)
            next_value = self.get_value(next_state)

        # Compute discounted reward (equation 6)
        # R_i^t = r_i^t + γ * V(s^{t+1})
        discounted_reward = reward + self.gamma * next_value

        # Compute advantage
        advantage = discounted_reward - value

        # Value loss: MSE between predicted value and discounted reward
        value_loss = F.mse_loss(value, discounted_reward.detach())

        # Policy loss: negative log probability weighted by advantage
        # -∇ log P(a|s) * (R - V(s))
        B, A, H, W = policy.shape

        # Gather log probabilities of selected actions
        action_indices_flat = action_indices.view(B, 1, H, W).expand(-1, A, -1, -1)
        selected_log_probs = torch.log(policy + 1e-8).gather(1, action_indices_flat)[
            :, 0:1
        ]

        # Policy loss (negative because we want to maximize reward)
        policy_loss = -(selected_log_probs * advantage.detach()).mean()

        # Update value network
        optimizer_value.zero_grad()
        value_loss.backward(retain_graph=True)

        # Check for NaN/Inf gradients
        for name, param in self.value_net.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient in value_net.{name}"
                )

        optimizer_value.step()

        # Update policy network
        optimizer_policy.zero_grad()
        policy_loss.backward()

        # Check for NaN/Inf gradients
        for name, param in self.policy_net.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient in policy_net.{name}"
                )

        for name, param in self.encoder.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient in encoder.{name}"
                )

        optimizer_policy.step()

        # Compute metrics
        with torch.no_grad():
            psnr_before = -10 * torch.log10(F.mse_loss(current_image, clean_images))
            psnr_after = -10 * torch.log10(F.mse_loss(next_image, clean_images))
            mean_reward = reward.mean()
            mean_advantage = advantage.mean()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "mean_reward": float(mean_reward.item()),
            "mean_advantage": float(mean_advantage.item()),
            "psnr_before": float(psnr_before.item()),
            "psnr_after": float(psnr_after.item()),
        }
