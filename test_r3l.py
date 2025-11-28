"""
Test R3L (Residual Recovery using Reinforcement Learning) implementation
"""

import torch
import torch.nn.functional as F
from src.r3l import R3L, FCNEncoder, PolicyNetwork, ValueNetwork


def test_fcn_encoder():
    """Test FCN encoder forward pass"""
    print("Testing FCN Encoder...")

    encoder = FCNEncoder(in_channels=3)

    # Test single image
    x = torch.randn(1, 3, 32, 32)
    output = encoder(x)

    assert output.shape == (1, 64, 32, 32), (
        f"Expected shape (1, 64, 32, 32), got {output.shape}"
    )
    assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    # Test batch
    x_batch = torch.randn(8, 3, 32, 32)
    output_batch = encoder(x_batch)

    assert output_batch.shape == (8, 64, 32, 32), (
        f"Expected shape (8, 64, 32, 32), got {output_batch.shape}"
    )
    assert torch.isfinite(output_batch).all(), "Batch output contains NaN or Inf"

    print("  ✓ FCN Encoder tests passed")


def test_policy_network():
    """Test policy network forward pass"""
    print("Testing Policy Network...")

    policy_net = PolicyNetwork(action_space_size=27)

    # Create encoded state
    state = torch.randn(1, 64, 32, 32)
    policy = policy_net(state)

    assert policy.shape == (1, 27, 32, 32), (
        f"Expected shape (1, 27, 32, 32), got {policy.shape}"
    )
    assert torch.isfinite(policy).all(), "Policy contains NaN or Inf"

    # Check probabilities sum to 1 over action dimension
    prob_sum = policy.sum(dim=1)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), (
        "Probabilities don't sum to 1"
    )

    # Check probabilities are non-negative
    assert (policy >= 0).all(), "Negative probabilities found"

    print("  ✓ Policy Network tests passed")


def test_value_network():
    """Test value network forward pass"""
    print("Testing Value Network...")

    value_net = ValueNetwork()

    # Create encoded state
    state = torch.randn(1, 64, 32, 32)
    value = value_net(state)

    assert value.shape == (1, 1, 32, 32), (
        f"Expected shape (1, 1, 32, 32), got {value.shape}"
    )
    assert torch.isfinite(value).all(), "Value contains NaN or Inf"

    print("  ✓ Value Network tests passed")


def test_r3l_initialization():
    """Test R3L model initialization"""
    print("Testing R3L Initialization...")

    model = R3L(in_channels=3, num_stages=5, action_range=(-13, 13), gamma=0.95)

    assert model.in_channels == 3
    assert model.num_stages == 5
    assert model.gamma == 0.95
    assert model.action_space_size == 27

    # Check action values are properly initialized
    assert model.action_values.shape == (27,)
    assert model.action_values.min() >= -13 / 255.0
    assert model.action_values.max() <= 13 / 255.0

    print("  ✓ R3L Initialization tests passed")


def test_r3l_forward_inference():
    """Test R3L forward pass in inference mode"""
    print("Testing R3L Forward Pass (Inference)...")

    model = R3L(in_channels=3, num_stages=5)
    model.eval()

    # Test single image
    x = torch.rand(1, 3, 32, 32)  # Noisy image in [0, 1]

    with torch.no_grad():
        output = model(x, training=False)

    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    assert torch.isfinite(output).all(), "Output contains NaN or Inf"
    assert output.min() >= 0.0 and output.max() <= 1.0, (
        f"Output range [{output.min()}, {output.max()}] not in [0, 1]"
    )

    # Test batch
    x_batch = torch.rand(4, 3, 32, 32)

    with torch.no_grad():
        output_batch = model(x_batch, training=False)

    assert output_batch.shape == x_batch.shape
    assert torch.isfinite(output_batch).all()
    assert output_batch.min() >= 0.0 and output_batch.max() <= 1.0

    print("  ✓ R3L Forward Pass (Inference) tests passed")


def test_r3l_forward_training():
    """Test R3L forward pass in training mode"""
    print("Testing R3L Forward Pass (Training)...")

    model = R3L(in_channels=3, num_stages=5)
    model.train()

    x = torch.rand(2, 3, 32, 32)
    ground_truth = torch.rand(2, 3, 32, 32)

    output = model(x, training=True, ground_truth=ground_truth)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert output.min() >= 0.0 and output.max() <= 1.0

    print("  ✓ R3L Forward Pass (Training) tests passed")


def test_sample_action():
    """Test action sampling"""
    print("Testing Action Sampling...")

    model = R3L(in_channels=3, num_stages=5)

    # Create dummy policy
    policy = torch.softmax(torch.randn(2, 27, 32, 32), dim=1)

    # Test stochastic sampling (training)
    action_indices, action_values = model.sample_action(policy, training=True)

    assert action_indices.shape == (2, 1, 32, 32)
    assert action_values.shape == (2, 1, 32, 32)
    assert (action_indices >= 0).all() and (action_indices < 27).all()
    assert torch.isfinite(action_values).all()

    # Test greedy sampling (inference)
    action_indices_greedy, action_values_greedy = model.sample_action(
        policy, training=False
    )

    assert action_indices_greedy.shape == (2, 1, 32, 32)
    assert action_values_greedy.shape == (2, 1, 32, 32)

    # Greedy should select max probability action
    expected_indices = torch.argmax(policy, dim=1, keepdim=True)
    assert torch.equal(action_indices_greedy, expected_indices)

    print("  ✓ Action Sampling tests passed")


def test_compute_reward():
    """Test reward computation"""
    print("Testing Reward Computation...")

    model = R3L(in_channels=3, num_stages=5)

    # Create test images
    ground_truth = torch.rand(2, 3, 32, 32)
    current_image = ground_truth + 0.1 * torch.randn_like(ground_truth)
    next_image = ground_truth + 0.05 * torch.randn_like(ground_truth)

    # Clamp to valid range
    current_image = torch.clamp(current_image, 0, 1)
    next_image = torch.clamp(next_image, 0, 1)

    reward = model.compute_reward(current_image, next_image, ground_truth)

    assert reward.shape == (2, 1, 32, 32)
    assert torch.isfinite(reward).all()

    # Reward should be positive if next_image is closer to ground_truth
    mse_current = F.mse_loss(current_image, ground_truth)
    mse_next = F.mse_loss(next_image, ground_truth)

    if mse_next < mse_current:
        assert reward.mean() > 0, "Reward should be positive when image improves"

    print("  ✓ Reward Computation tests passed")


def test_train_step():
    """Test one training step"""
    print("Testing Training Step...")

    model = R3L(in_channels=3, num_stages=5)

    # Create dummy data
    noisy = torch.rand(2, 3, 32, 32)
    clean = torch.rand(2, 3, 32, 32)

    # Create optimizers
    policy_params = list(model.encoder.parameters()) + list(
        model.policy_net.parameters()
    )
    optimizer_policy = torch.optim.Adam(policy_params, lr=1e-4)

    value_params = list(model.value_net.parameters())
    optimizer_value = torch.optim.Adam(value_params, lr=1e-3)

    # Perform training step
    metrics = model.train_step(
        noisy_images=noisy,
        clean_images=clean,
        optimizer_policy=optimizer_policy,
        optimizer_value=optimizer_value,
        stage=0,
    )

    # Check metrics are returned
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "mean_reward" in metrics
    assert "mean_advantage" in metrics
    assert "psnr_before" in metrics
    assert "psnr_after" in metrics

    # Check all metrics are finite
    for key, value in metrics.items():
        assert isinstance(value, float), f"{key} should be float"
        assert not torch.isnan(torch.tensor(value)), f"{key} is NaN"
        assert not torch.isinf(torch.tensor(value)), f"{key} is Inf"

    print("  ✓ Training Step tests passed")


def test_gradients():
    """Test that gradients flow properly"""
    print("Testing Gradient Flow...")

    model = R3L(in_channels=3, num_stages=5)

    # Create dummy data
    noisy = torch.rand(2, 3, 32, 32)
    clean = torch.rand(2, 3, 32, 32)

    # Create optimizers
    policy_params = list(model.encoder.parameters()) + list(
        model.policy_net.parameters()
    )
    optimizer_policy = torch.optim.Adam(policy_params, lr=1e-4)

    value_params = list(model.value_net.parameters())
    optimizer_value = torch.optim.Adam(value_params, lr=1e-3)

    # Get initial parameters
    initial_policy_param = next(model.policy_net.parameters()).clone()
    initial_value_param = next(model.value_net.parameters()).clone()

    # Perform training step
    model.train_step(
        noisy_images=noisy,
        clean_images=clean,
        optimizer_policy=optimizer_policy,
        optimizer_value=optimizer_value,
        stage=0,
    )

    # Check parameters updated
    updated_policy_param = next(model.policy_net.parameters())
    updated_value_param = next(model.value_net.parameters())

    assert not torch.equal(initial_policy_param, updated_policy_param), (
        "Policy parameters not updated"
    )
    assert not torch.equal(initial_value_param, updated_value_param), (
        "Value parameters not updated"
    )

    print("  ✓ Gradient Flow tests passed")


def run_all_tests():
    """Run all R3L tests"""
    print("\n" + "=" * 70)
    print("R3L MODEL TESTS")
    print("=" * 70 + "\n")

    test_fcn_encoder()
    test_policy_network()
    test_value_network()
    test_r3l_initialization()
    test_r3l_forward_inference()
    test_r3l_forward_training()
    test_sample_action()
    test_compute_reward()
    test_train_step()
    test_gradients()

    print("\n" + "=" * 70)
    print("ALL R3L TESTS PASSED ✓")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
