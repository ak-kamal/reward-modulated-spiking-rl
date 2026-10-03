"""Tests for AdaptiveLIFNode homeostatic threshold adaptation."""

from __future__ import annotations

import torch
import pytest

from spiking_rl.models.lif import AdaptiveLIFNode


def test_forward_returns_spikes_same_shape_as_input():
    neuron = AdaptiveLIFNode(tau=2.0)
    x = torch.randn(4, 8) * 5.0
    spikes = neuron(x)
    assert spikes.shape == (4, 8)


def test_threshold_changes_over_time():
    neuron = AdaptiveLIFNode(tau=2.0, eta_threshold=0.1, target_rate=0.5)
    x = torch.ones(4, 8) * 20.0
    threshold_before = neuron.v_threshold
    for _ in range(20):
        neuron(x)
    threshold_after = neuron.v_threshold
    assert threshold_before != threshold_after


def test_threshold_rises_when_firing_above_target():
    """With strong input, the neuron fires above target and threshold rises."""
    neuron = AdaptiveLIFNode(tau=2.0, eta_threshold=0.1, target_rate=0.01)
    initial = neuron.v_threshold
    x = torch.ones(4, 8) * 50.0
    for _ in range(50):
        neuron(x)
    assert neuron.v_threshold > initial


def test_threshold_falls_when_firing_below_target():
    """With weak input, firing stays below target and threshold falls."""
    neuron = AdaptiveLIFNode(tau=2.0, eta_threshold=0.1, target_rate=0.9)
    initial = neuron.v_threshold
    x = torch.ones(4, 8) * 0.01
    for _ in range(50):
        neuron(x)
    assert neuron.v_threshold < initial


def test_threshold_respects_min_and_max_bounds():
    neuron = AdaptiveLIFNode(
        tau=2.0, eta_threshold=1.0, target_rate=0.5,
        threshold_min=0.5, threshold_max=1.5,
    )
    x = torch.ones(2, 4) * 50.0
    for _ in range(100):
        neuron(x)
    assert neuron.threshold_min <= neuron.v_threshold <= neuron.threshold_max


def test_forward_accepts_same_batch_shape_repeatedly():
    """Sanity: same batch size across calls works (baseline)."""
    neuron = AdaptiveLIFNode(tau=2.0)
    neuron(torch.ones(1, 8) * 5.0)
    neuron(torch.ones(1, 8) * 5.0)
    assert neuron.rate_ema.shape == (8,)


def test_forward_accepts_larger_batch_after_small():
    """1 -> 64 transition: cached v/traces were (1, 8), now (64, 8)."""
    neuron = AdaptiveLIFNode(tau=2.0)
    neuron(torch.ones(1, 8) * 5.0)
    neuron(torch.ones(64, 8) * 5.0)
    assert neuron.rate_ema.shape == (8,)


def test_forward_accepts_smaller_batch_after_large():
    """64 -> 2 transition: the case that was failing."""
    neuron = AdaptiveLIFNode(tau=2.0)
    neuron(torch.ones(64, 8) * 5.0)
    neuron(torch.ones(2, 8) * 5.0)
    assert neuron.rate_ema.shape == (8,)


def test_forward_accepts_arbitrary_batch_transitions():
    """Cycle through several batch sizes; all should work."""
    neuron = AdaptiveLIFNode(tau=2.0)
    for batch in [1, 64, 2, 32, 4, 1]:
        neuron(torch.ones(batch, 8) * 5.0)
        assert neuron.rate_ema.shape == (8,), (
            f"rate_ema shape wrong after batch={batch}: "
            f"{tuple(neuron.rate_ema.shape)}"
        )


def test_reset_zeros_rate_estimate():
    neuron = AdaptiveLIFNode(tau=2.0)
    neuron(torch.ones(1, 8) * 20.0)
    assert neuron.rate_ema.abs().sum().item() > 0
    neuron.reset()
    assert neuron.rate_ema.abs().sum().item() == 0.0


def test_homeostatic_diagnostics_returns_expected_keys():
    neuron = AdaptiveLIFNode(tau=2.0)
    neuron(torch.ones(1, 8) * 5.0)
    diag = neuron.homeostatic_diagnostics()
    assert "v_threshold" in diag
    assert "target_rate" in diag
    assert "rate_ema_mean" in diag
    

def test_state_dict_save_load_preserves_v_threshold():
    """Trained threshold must survive save/load.

    This test would have failed before the get_extra_state override,
    because v_threshold was a plain instance attribute.
    """
    torch.manual_seed(0)
    neuron = AdaptiveLIFNode(tau=2.0, eta_threshold=0.5, target_rate=0.9)
    x = torch.ones(1, 8) * 0.01  # low input → threshold falls
    for _ in range(20):
        neuron(x)

    threshold_before = float(neuron.v_threshold)
    assert threshold_before < 1.0, (
        f"Expected threshold to fall below 1.0, got {threshold_before}"
    )

    saved = neuron.state_dict()
    neuron2 = AdaptiveLIFNode(tau=2.0, eta_threshold=0.5, target_rate=0.9)
    neuron2.load_state_dict(saved)

    assert abs(neuron2.v_threshold - threshold_before) < 1e-6, (
        f"v_threshold not preserved: {neuron2.v_threshold} vs {threshold_before}"
    )


def test_state_dict_save_load_preserves_rate_ema():
    """Running rate estimate must survive save/load."""
    torch.manual_seed(0)
    neuron = AdaptiveLIFNode(tau=2.0)
    x = torch.ones(1, 8) * 5.0
    for _ in range(10):
        neuron(x)

    assert neuron.rate_ema is not None
    assert neuron.rate_ema.abs().sum().item() > 0
    rate_before = neuron.rate_ema.clone()

    saved = neuron.state_dict()
    neuron2 = AdaptiveLIFNode(tau=2.0)
    neuron2.load_state_dict(saved)

    assert neuron2.rate_ema is not None
    assert torch.allclose(neuron2.rate_ema, rate_before), (
        f"rate_ema not preserved"
    )


def test_loaded_neuron_reproduces_spiking_behavior():
    """A trained neuron saved and loaded must produce identical spikes."""
    torch.manual_seed(0)
    neuron = AdaptiveLIFNode(tau=2.0, eta_threshold=0.1, target_rate=0.3)
    x = torch.ones(1, 8) * 3.0
    for _ in range(30):
        neuron(x)

    # Save.
    saved = neuron.state_dict()

    # Reload into a fresh neuron.
    neuron2 = AdaptiveLIFNode(tau=2.0, eta_threshold=0.1, target_rate=0.3)
    neuron2.load_state_dict(saved)

    # Compare loaded state.
    assert abs(neuron2.v_threshold - neuron.v_threshold) < 1e-6
    assert torch.allclose(neuron2.rate_ema, neuron.rate_ema)

    # Now verify behavior on fresh inputs after reset.
    neuron.reset()
    neuron2.reset()
    test_x = torch.ones(1, 8) * 2.0
    for _ in range(10):
        s1 = neuron(test_x)
        s2 = neuron2(test_x)
        assert torch.equal(s1, s2), "Loaded neuron produces different spikes"