"""Tests for SNN-specific weight initialization."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from spiking_rl.models.initialization import snn_weight_init, snn_weight_std


def test_snn_weight_std_basic():
    std = snn_weight_std(in_features=100, v_threshold=1.0)
    # P(u > 1.0) ≈ 0.1587
    # var = 1 / (100 * 0.1587) ≈ 0.063
    # std ≈ 0.251
    assert 0.2 < std < 0.35


def test_snn_weight_std_decreases_with_width():
    s1 = snn_weight_std(100, 1.0)
    s2 = snn_weight_std(400, 1.0)
    assert s2 < s1


def test_snn_weight_std_increases_with_threshold():
    s1 = snn_weight_std(100, 0.5)
    s2 = snn_weight_std(100, 1.0)
    assert s2 > s1


def test_snn_weight_std_rejects_bad_inputs():
    with pytest.raises(ValueError, match="positive"):
        snn_weight_std(0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        snn_weight_std(100, 0.0)


def test_snn_weight_init_applies_correct_std():
    torch.manual_seed(0)
    linear = nn.Linear(100, 50)
    snn_weight_init(linear, v_threshold=1.0)
    expected_std = snn_weight_std(100, 1.0)
    # Sample std over 5000 weights should be close.
    observed_std = linear.weight.std().item()
    assert observed_std == pytest.approx(expected_std, rel=0.1)


def test_snn_weight_init_zeroes_bias():
    linear = nn.Linear(10, 5)
    snn_weight_init(linear, v_threshold=1.0)
    assert torch.allclose(linear.bias, torch.zeros_like(linear.bias))


def test_snn_weight_init_rejects_non_linear():
    with pytest.raises(TypeError, match="nn.Linear"):
        snn_weight_init("not a layer")  # type: ignore