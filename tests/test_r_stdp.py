"""
Unit tests for RewardBaseline and ThreeFactorLearner.

The tests deliberately populate eligibility traces directly (rather than
running forward passes through the spiking neurons) so that these tests
exercise the learner's coordination logic in isolation. This decouples
the learning rule from the spiking dynamics, which is tested elsewhere.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spiking_rl.learning.eligibility_trace import TracedLinear
from spiking_rl.learning.r_stdp import (
    RewardBaseline,
    ThreeFactorLearner,
    collect_traced_layers,
    split_actor_critic_layers,
)


# ----------------------------------------------------------------------
# RewardBaseline
# ----------------------------------------------------------------------


def test_baseline_initial_value():
    b = RewardBaseline(alpha=0.1, initial=0.5)
    assert b.value == 0.5
    assert b.diagnostics()["count"] == 0


def test_baseline_update_moves_toward_reward():
    b = RewardBaseline(alpha=0.1, initial=0.0)
    b.update(10.0)
    # value = 0.9 * 0 + 0.1 * 10 = 1.0
    assert b.value == pytest.approx(1.0)


def test_baseline_converges_to_constant_reward():
    b = RewardBaseline(alpha=0.1, initial=0.0)
    for _ in range(200):
        b.update(5.0)
    assert b.value == pytest.approx(5.0, abs=0.01)


def test_baseline_modulate_uses_current_value():
    b = RewardBaseline(alpha=0.1, initial=1.0)
    assert b.modulate(3.0) == pytest.approx(2.0)
    assert b.modulate(-1.0) == pytest.approx(-2.0)


def test_baseline_modulate_does_not_update():
    b = RewardBaseline(alpha=0.1, initial=1.0)
    before = b.value
    b.modulate(100.0)
    assert b.value == before
    assert b.diagnostics()["count"] == 0


def test_baseline_reset():
    b = RewardBaseline(alpha=0.1, initial=0.0)
    b.update(5.0)
    b.reset()
    assert b.value == 0.0
    assert b.diagnostics()["count"] == 0


def test_baseline_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        RewardBaseline(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        RewardBaseline(alpha=1.5)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def test_collect_traced_layers():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = TracedLinear(4, 8)
            self.l2 = TracedLinear(8, 3)
            self.other = nn.Linear(3, 1)

    net = Net()
    layers = collect_traced_layers(net)
    assert len(layers) == 2
    assert layers[0] is net.l1
    assert layers[1] is net.l2


def test_split_actor_critic_layers():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(TracedLinear(4, 8), TracedLinear(8, 2))
            self.critic = nn.Sequential(TracedLinear(4, 8), TracedLinear(8, 1))

    net = Net()
    actor, critic = split_actor_critic_layers(net)
    assert len(actor) == 2
    assert len(critic) == 2
    # Correct partition.
    assert actor[0] is net.actor[0]
    assert critic[0] is net.critic[0]


# ----------------------------------------------------------------------
# ThreeFactorLearner
# ----------------------------------------------------------------------


def _populate_eligibility(layer: TracedLinear, in_f: int, out_f: int) -> None:
    """Directly fill a layer's eligibility with a synthetic nonzero value."""
    pre = torch.randn(2, in_f)
    post = torch.randn(2, out_f).abs()
    layer.eligibility.update(pre, post)


def test_learner_requires_at_least_one_layer():
    with pytest.raises(ValueError, match="at least one"):
        ThreeFactorLearner([])


def test_learner_rejects_nonpositive_lr():
    l1 = TracedLinear(4, 2)
    with pytest.raises(ValueError, match="lr"):
        ThreeFactorLearner([l1], lr=0.0)


def test_learner_updates_all_layers():
    torch.manual_seed(0)
    l1 = TracedLinear(4, 8)
    l2 = TracedLinear(8, 3)
    _populate_eligibility(l1, 4, 8)
    _populate_eligibility(l2, 8, 3)

    w1_before = l1.linear.weight.clone()
    w2_before = l2.linear.weight.clone()

    learner = ThreeFactorLearner([l1, l2], lr=0.01, use_baseline=False)
    total = learner.step(reward=1.0)

    assert total > 0.0
    assert not torch.allclose(l1.linear.weight, w1_before)
    assert not torch.allclose(l2.linear.weight, w2_before)
    assert learner.diagnostics()["update_count"] == 1


def test_learner_zero_reward_is_noop_when_baseline_off():
    torch.manual_seed(0)
    l1 = TracedLinear(4, 2)
    _populate_eligibility(l1, 4, 2)
    w_before = l1.linear.weight.clone()

    learner = ThreeFactorLearner([l1], lr=0.01, use_baseline=False)
    total = learner.step(reward=0.0)

    assert total == 0.0
    assert torch.allclose(l1.linear.weight, w_before)


def test_learner_baseline_subtracts_reward():
    torch.manual_seed(0)
    l1 = TracedLinear(4, 2)
    _populate_eligibility(l1, 4, 2)

    # Baseline initialized to 1.0; a reward of 1.0 gives modulation ~0.
    learner = ThreeFactorLearner(
        [l1], lr=0.01, use_baseline=True,
        baseline_alpha=1e-9, baseline_initial=1.0,
    )
    w_before = l1.linear.weight.clone()
    total = learner.step(reward=1.0)
    # Modulation is ~0 (reward minus baseline), so update magnitude ~0.
    assert total == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(l1.linear.weight, w_before, atol=1e-7)


def test_learner_apply_modulation_ignores_baseline():
    torch.manual_seed(0)
    l1 = TracedLinear(4, 2)
    _populate_eligibility(l1, 4, 2)

    learner = ThreeFactorLearner(
        [l1], lr=0.01, use_baseline=True, baseline_initial=100.0
    )
    w_before = l1.linear.weight.clone()
    # Even though baseline=100, apply_modulation uses the given value.
    total = learner.apply_modulation(1.0)
    assert total > 0.0
    assert not torch.allclose(l1.linear.weight, w_before)
    # Baseline should be unchanged by apply_modulation.
    assert learner.baseline.value == 100.0


def test_learner_positive_and_negative_modulation_move_oppositely():
    torch.manual_seed(0)
    l1 = TracedLinear(4, 2)
    l2 = TracedLinear(4, 2)
    # Identical eligibility.
    _populate_eligibility(l1, 4, 2)
    l2.eligibility._E = l1.eligibility._E.clone()

    # Snapshot each layer's OWN initial weights.
    w_orig_a = l1.linear.weight.clone()
    w_orig_b = l2.linear.weight.clone()

    learner_a = ThreeFactorLearner([l1], lr=0.1, use_baseline=False)
    learner_b = ThreeFactorLearner([l2], lr=0.1, use_baseline=False)

    learner_a.apply_modulation(+1.0)
    learner_b.apply_modulation(-1.0)

    a_change = l1.linear.weight - w_orig_a
    b_change = l2.linear.weight - w_orig_b
    # The changes should be mirror images (same eligibility, opposite
    # modulation signs).
    assert torch.allclose(a_change, -b_change, atol=1e-7)


def test_learner_reset_layers_clears_eligibility():
    torch.manual_seed(0)
    l1 = TracedLinear(4, 2)
    # Use the real forward path so both eligibility and _input_trace
    # are initialized. Strong input to guarantee spiking.
    x = torch.ones(1, 4) * 20.0
    for _ in range(20):
        l1(x)

    assert not l1.eligibility.diagnostics()["all_zero"]
    assert l1._input_trace is not None

    learner = ThreeFactorLearner([l1], lr=0.01)
    learner.reset_layers()

    assert l1.eligibility.diagnostics()["all_zero"]
    assert torch.allclose(l1._input_trace, torch.zeros_like(l1._input_trace))


def test_learner_reset_diagnostics_does_not_touch_weights_or_baseline():
    l1 = TracedLinear(4, 2)
    _populate_eligibility(l1, 4, 2)
    learner = ThreeFactorLearner([l1], lr=0.01, baseline_initial=0.0)

    learner.step(reward=1.0)
    w_after = l1.linear.weight.clone()
    baseline_after = learner.baseline.value

    learner.reset_diagnostics()

    assert learner.diagnostics()["update_count"] == 0
    assert torch.allclose(l1.linear.weight, w_after)
    assert learner.baseline.value == baseline_after


def test_learner_weight_clipping():
    torch.manual_seed(0)
    l1 = TracedLinear(4, 2)
    _populate_eligibility(l1, 4, 2)
    # Huge modulation and LR to force runaway.
    learner = ThreeFactorLearner(
        [l1], lr=10.0, use_baseline=False, clip_weights=2.0
    )
    for _ in range(20):
        learner.apply_modulation(100.0)
    assert l1.linear.weight.abs().max().item() <= 2.0 + 1e-6


def test_learner_apply_modulation_rejects_non_scalar():
    l1 = TracedLinear(4, 2)
    learner = ThreeFactorLearner([l1])
    with pytest.raises(TypeError, match="scalar"):
        learner.apply_modulation(torch.tensor([1.0, 2.0]))


def test_learner_diagnostics_shape():
    l1 = TracedLinear(4, 2)
    l2 = TracedLinear(4, 2)
    learner = ThreeFactorLearner([l1, l2], lr=0.01)
    diag = learner.diagnostics()
    assert diag["num_layers"] == 2
    assert len(diag["last_updates"]) in (0, 2)
    assert len(diag["cumulative_updates"]) == 2
    assert "baseline" in diag