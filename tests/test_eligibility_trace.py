"""
Unit tests for EligibilityTrace and TracedLinear.

These tests verify the stateful behavior, shape handling, decay, and
reward-modulated weight updates. Since state management is the riskiest
part of the design, the tests here are deliberately explicit about what
"correct" means.
"""

from __future__ import annotations

import math

import pytest
import torch

from spiking_rl.learning.eligibility_trace import (
    EligibilityTrace,
    TracedLinear,
)


# ----------------------------------------------------------------------
# EligibilityTrace
# ----------------------------------------------------------------------


def test_initial_state_is_uninitialized():
    e = EligibilityTrace(in_features=8, out_features=4, tau_e=50.0)
    assert e.eligibility is None
    assert e.diagnostics()["initialized"] is False


def test_first_update_initializes_matrix_with_correct_shape():
    e = EligibilityTrace(in_features=8, out_features=4, tau_e=50.0)
    pre = torch.randn(2, 8)
    post = torch.randn(2, 4)
    e.update(pre, post)
    assert e.eligibility is not None
    assert e.eligibility.shape == (4, 8)


def test_decay_matches_expected_value():
    tau_e = 50.0
    e = EligibilityTrace(in_features=2, out_features=2, tau_e=tau_e)
    # First update: E = contribution (no prior).
    pre = torch.ones(1, 2)
    post = torch.ones(1, 2)
    e.update(pre, post)
    E1 = e.eligibility.clone()
    # Second update with zero traces: E should be decay * E1.
    pre_zero = torch.zeros(1, 2)
    post_zero = torch.zeros(1, 2)
    e.update(pre_zero, post_zero)
    E2 = e.eligibility.clone()
    expected_decay = math.exp(-1.0 / tau_e)
    assert torch.allclose(E2, expected_decay * E1, atol=1e-6)


def test_zero_traces_leave_trace_decaying_to_zero():
    e = EligibilityTrace(in_features=2, out_features=2, tau_e=10.0)
    pre = torch.ones(1, 2)
    post = torch.ones(1, 2)
    e.update(pre, post)
    initial_norm = e.eligibility.norm().item()
    for _ in range(100):
        e.update(torch.zeros(1, 2), torch.zeros(1, 2))
    final_norm = e.eligibility.norm().item()
    assert final_norm < initial_norm * 1e-3


def test_reset_zeroes_eligibility():
    e = EligibilityTrace(in_features=4, out_features=4, tau_e=50.0)
    e.update(torch.ones(1, 4), torch.ones(1, 4))
    assert not e.diagnostics()["all_zero"]
    e.reset()
    assert e.diagnostics()["all_zero"]
    assert e.diagnostics()["reset_count"] == 1


def test_shape_mismatch_raises():
    e = EligibilityTrace(in_features=8, out_features=4, tau_e=50.0)
    with pytest.raises(ValueError, match="pre_trace has"):
        e.update(torch.randn(2, 5), torch.randn(2, 4))
    with pytest.raises(ValueError, match="post_trace has"):
        e.update(torch.randn(2, 8), torch.randn(2, 3))


def test_batch_size_mismatch_raises():
    e = EligibilityTrace(in_features=4, out_features=4, tau_e=50.0)
    with pytest.raises(ValueError, match="Batch sizes differ"):
        e.update(torch.randn(2, 4), torch.randn(3, 4))


def test_outer_product_is_correct():
    """Manually verify the outer-product computation."""
    e = EligibilityTrace(in_features=3, out_features=2, tau_e=1e9)
    # With tau_e huge, decay ≈ 1, so E after one update ≈ contribution.
    pre = torch.tensor([[1.0, 2.0, 3.0]])
    post = torch.tensor([[4.0, 5.0]])
    e.update(pre, post)
    # Expected contribution: post^T @ pre = [[4*1, 4*2, 4*3],
    #                                       [5*1, 5*2, 5*3]]
    expected = torch.tensor([[4.0, 8.0, 12.0], [5.0, 10.0, 15.0]])
    assert torch.allclose(e.eligibility, expected, atol=1e-4)


def test_diagnostics_reflect_state():
    e = EligibilityTrace(in_features=4, out_features=3, tau_e=20.0)
    e.update(torch.randn(2, 4), torch.randn(2, 3))
    diag = e.diagnostics()
    assert diag["initialized"] is True
    assert diag["shape"] == (3, 4)
    assert diag["update_count"] == 1
    assert diag["all_zero"] is False


# ----------------------------------------------------------------------
# TracedLinear
# ----------------------------------------------------------------------


def test_traced_linear_forward_shape():
    block = TracedLinear(in_features=6, out_features=4)
    x = torch.randn(2, 6)
    spikes = block(x)
    assert spikes.shape == (2, 4)


def test_traced_linear_eligibility_shape():
    block = TracedLinear(in_features=6, out_features=4)
    block(torch.randn(2, 6))
    assert block.eligibility.eligibility.shape == (4, 6)


def test_traced_linear_reset_clears_state():
    torch.manual_seed(0)  # deterministic init so spiking is guaranteed
    block = TracedLinear(in_features=6, out_features=4)
    # Strong input to guarantee the LIF neurons fire.
    x = torch.ones(2, 6) * 20.0
    for _ in range(20):
        block(x)

    # Sanity: eligibility should be nonzero after warmup.
    diag = block.eligibility.diagnostics()
    assert not diag["all_zero"], (
        f"Expected nonzero eligibility after 20 steps of strong input. "
        f"Diagnostics: {diag}"
    )

    block.reset()

    assert block.eligibility.diagnostics()["all_zero"]
    assert torch.allclose(
        block.neuron.pre_trace, torch.zeros_like(block.neuron.pre_trace)
    )
    assert torch.allclose(
        block.neuron.post_trace, torch.zeros_like(block.neuron.post_trace)
    )
    assert torch.allclose(
        block._input_trace, torch.zeros_like(block._input_trace)
    )


def test_apply_reward_updates_weights_in_expected_direction():
    torch.manual_seed(0)
    block = TracedLinear(in_features=4, out_features=2)
    x = torch.ones(1, 4) * 5.0  # scaled to drive spikes
    for _ in range(20):
        block(x)
    w_before = block.linear.weight.clone()
    delta_norm = block.apply_reward(reward_signal=1.0, lr=0.01)
    assert delta_norm > 0.0
    w_after = block.linear.weight.clone()
    assert not torch.allclose(w_before, w_after)


def test_apply_reward_with_zero_reward_is_noop():
    block = TracedLinear(in_features=4, out_features=2)
    block(torch.ones(1, 4))
    w_before = block.linear.weight.clone()
    block.apply_reward(reward_signal=0.0, lr=0.01)
    assert torch.allclose(block.linear.weight, w_before)


def test_apply_reward_before_forward_is_noop():
    block = TracedLinear(in_features=4, out_features=2)
    w_before = block.linear.weight.clone()
    result = block.apply_reward(reward_signal=1.0, lr=0.01)
    assert result == 0.0
    assert torch.allclose(block.linear.weight, w_before)


def test_weight_clipping_prevents_runaway():
    block = TracedLinear(in_features=4, out_features=2)
    for _ in range(100):
        block(torch.ones(1, 4) * 10.0)
        block.apply_reward(reward_signal=1.0, lr=1.0, clip_weights=2.0)
    assert block.linear.weight.abs().max().item() <= 2.0 + 1e-5


def test_positive_and_negative_rewards_push_opposite_directions():
    torch.manual_seed(0)
    block_a = TracedLinear(in_features=4, out_features=2)
    block_b = TracedLinear(in_features=4, out_features=2)
    block_b.load_state_dict(block_a.state_dict())

    x = torch.ones(1, 4) * 5.0  # scaled to drive spikes
    w_original = block_a.linear.weight.clone()

    for _ in range(20):
        block_a(x)
        block_b(x)

    block_a.apply_reward(reward_signal=+1.0, lr=0.1)
    block_b.apply_reward(reward_signal=-1.0, lr=0.1)

    a_change = block_a.linear.weight - w_original
    b_change = block_b.linear.weight - w_original
    assert torch.allclose(a_change, -b_change, atol=1e-6)
    assert a_change.abs().sum().item() > 0.0
    
def test_normalize_weights_sets_target_norm():
    layer = TracedLinear(in_features=8, out_features=4, init_snn=False)
    layer.linear.weight.data = torch.randn(4, 8) * 10.0  # large weights
    layer.normalize_weights(target_norm=1.0)
    norms = layer.linear.weight.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_weight_norm_diagnostics_shape():
    layer = TracedLinear(in_features=8, out_features=4)
    diag = layer.weight_norm_diagnostics()
    assert "weight_norm_mean" in diag
    assert "weight_norm_max" in diag
    assert "weight_norm_min" in diag


def test_use_adaptive_false_uses_standard_lif():
    from spiking_rl.models.lif import LIFNodeWithTrace, AdaptiveLIFNode
    layer_adaptive = TracedLinear(4, 2, use_adaptive=True)
    layer_standard = TracedLinear(4, 2, use_adaptive=False)
    assert isinstance(layer_adaptive.neuron, AdaptiveLIFNode)
    assert isinstance(layer_standard.neuron, LIFNodeWithTrace)
    assert not isinstance(layer_standard.neuron, AdaptiveLIFNode)
    
def test_traced_linear_handles_batch_shape_changes():
    """Forward passes with changing batch sizes should not crash."""
    layer = TracedLinear(in_features=6, out_features=4, init_snn=False)
    layer(torch.ones(1, 6) * 5.0)
    layer(torch.ones(64, 6) * 5.0)
    layer(torch.ones(2, 6) * 5.0)
    # Eligibility and input trace should still be sane.
    assert layer.eligibility.eligibility is not None
    assert layer._input_trace.shape == (2, 6)