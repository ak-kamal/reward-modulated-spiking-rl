"""
SNN-specific weight initialization.

Standard ANN initialization schemes (Kaiming, Xavier) are designed for
continuous activations like ReLU. When applied to spiking neural networks,
they cause information loss and vanishing spikes in deeper layers, because
SNNs binarize their activations via the firing threshold.

This module implements the weight initialization method derived in:

    Micheli, Booij, van Gemert, Tomen (2025),
    "Deep activity propagation via weight initialization in spiking
    neural networks",
    IEEE Conference Publication / arXiv:2410.00580.

The method conserves the variance of membrane potentials across layers
by accounting for the quantization operation. The key equation is:

    Var[w_l] = 1 / (n_l * P(u_{l-1} > theta))

where n_l is the number of input neurons to layer l, and P(u > theta)
is the probability that a standard normal variable exceeds the threshold.

This is a general, architecture-agnostic fix: it can be applied to any
spiking layer without changing the learning rule, the eligibility trace
mechanism, or the network structure.
"""

from __future__ import annotations

import math

import torch.nn as nn
from scipy.stats import norm


def snn_weight_std(
    in_features: int,
    v_threshold: float = 1.0,
) -> float:
    """Compute the weight standard deviation for SNN-specific init.

    Parameters
    ----------
    in_features : int
        Number of input features (fan-in).
    v_threshold : float
        Firing threshold of the LIF neuron.

    Returns
    -------
    float
        The standard deviation for the weight distribution.
    """
    if in_features <= 0:
        raise ValueError(f"in_features must be positive, got {in_features}")
    if v_threshold <= 0:
        raise ValueError(f"v_threshold must be positive, got {v_threshold}")

    # P(u > theta) for u ~ N(0, 1) = 1 - Phi(theta) = Phi(-theta)
    p_above = norm.sf(v_threshold)  # sf = survival function = 1 - cdf

    # Guard against extremely small probabilities (extremely high threshold),
    # which would produce enormous weights.
    if p_above < 1e-8:
        raise ValueError(
            f"P(u > {v_threshold}) = {p_above:.2e} is too small. "
            f"The threshold is too high relative to the unit variance "
            f"assumption. Consider a smaller v_threshold."
        )

    var_w = 1.0 / (in_features * p_above)
    return math.sqrt(var_w)


def snn_weight_init(
    linear: nn.Linear,
    v_threshold: float = 1.0,
) -> None:
    """Initialize a Linear layer with SNN-specific weight distribution.

    Applies:
        W ~ N(0, 1 / (n_in * P(u > theta)))
        b = 0

    Parameters
    ----------
    linear : nn.Linear
        The linear layer to initialize.
    v_threshold : float
        Firing threshold of the LIF neuron that follows this layer.
    """
    if not isinstance(linear, nn.Linear):
        raise TypeError(
            f"Expected nn.Linear, got {type(linear).__name__}"
        )

    std = snn_weight_std(linear.in_features, v_threshold)
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
        
def snn_target_weight_norm(v_threshold: float = 1.0) -> float:
    """Return the per-row L2 norm produced by SNN weight initialization.

    For a Linear layer with in_features=n and firing threshold θ, the
    Micheli et al. (2025) initialization sets per-weight std to
    sqrt(1 / (n * P(u > θ))). The per-row L2 norm is therefore:

        sqrt(n) * sqrt(1 / (n * P(u > θ))) = sqrt(1 / P(u > θ))

    which is independent of n. For θ=1.0 this is ≈ 2.51.

    This is the natural target for output balancing: it lets weights
    vary in *direction* (which is what learning should change) while
    preventing runaway growth in *magnitude*.
    """
    from scipy.stats import norm as _norm
    p_above = _norm.sf(v_threshold)
    return float((1.0 / p_above) ** 0.5)