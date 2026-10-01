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