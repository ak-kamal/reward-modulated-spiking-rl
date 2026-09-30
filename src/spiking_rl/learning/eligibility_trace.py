"""
Eligibility trace module for reward-modulated spiking RL.

This module provides the stateful eligibility trace machinery for the
three-factor R-STDP learning rule. The eligibility trace is the "memory"
of recent pre×post-synaptic coincidences at each synapse. It decays over
time but persists long enough to bridge the gap between a spiking event
and a delayed reward signal.

Design
------
Two classes are provided:

1. ``EligibilityTrace`` — a per-synapse eligibility matrix E of shape
   ``(out_features, in_features)``. At each time step it decays E
   exponentially and adds a new contribution equal to the outer product
   of the neuron's pre- and post-synaptic traces, averaged over the
   batch.

2. ``TracedLinear`` — a composite module that combines ``nn.Linear`` +
   ``LIFNodeWithTrace`` + ``EligibilityTrace`` into a single block.
   Forward passes update the spiking state and accumulate the
   eligibility. A separate ``apply_reward()`` method performs the actual
   R-STDP weight update.

The separation between forward (spike + trace accumulation) and
apply_reward (weight update) is deliberate. Forward runs at every time
step; reward may arrive only occasionally. The eligibility trace holds
the "credit" until a reward signal is available, which is exactly what
solves the distal reward problem.

State management is the riskiest part of this design. To make debugging
fast, ``EligibilityTrace`` maintains explicit counters, exposes a
``diagnostics()`` method, and performs shape/device checks on every
update. If something is wrong with the state, the error message tells
us exactly what.

References
----------
- Bellec et al. (2020), "A solution to the learning dilemma for
  recurrent networks of spiking neurons", Nature Communications 11:3625.
  (e-prop: factorized eligibility traces)
- Izhikevich (2007), "Solving the distal reward problem through linkage
  of STDP and dopamine signaling", Cerebral Cortex 17(10):2443-2452.
  (three-factor rule with eligibility trace)
- Frémaux & Gerstner (2016), "Neuromodulated STDP and theory of
  three-factor learning rules", Frontiers in Neural Circuits 9:85.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
import torch.nn as nn

from spiking_rl.models.lif import LIFNodeWithTrace


# ======================================================================
# EligibilityTrace
# ======================================================================


class EligibilityTrace:
    """A per-synapse eligibility matrix with exponential decay.

    The eligibility matrix ``E`` has the same shape as the weight matrix
    of the preceding ``nn.Linear`` layer: ``(out_features, in_features)``.
    At each time step:

        E ← decay * E + outer_product(post_trace, pre_trace).mean(batch)

    where ``decay = exp(-1 / tau_e)``.

    Parameters
    ----------
    in_features : int
        Number of input features (columns of E).
    out_features : int
        Number of output features (rows of E).
    tau_e : float, optional
        Time constant for the eligibility decay, in units of time steps.
        Larger values mean longer memory. Default is 50.0.
    record_history : bool, optional
        If True, keep a list of snapshots of E at every update. Useful
        for debugging but memory-intensive. Default is False.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_e: float = 50.0,
        record_history: bool = False,
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"in_features and out_features must be positive, "
                f"got ({in_features}, {out_features})"
            )
        if tau_e <= 0:
            raise ValueError(f"tau_e must be positive, got {tau_e}")

        self.in_features = in_features
        self.out_features = out_features
        self.tau_e = tau_e
        self.record_history = record_history

        # Exponential decay factor per time step.
        # tau_e=50 -> decay ≈ 0.9802
        self._decay = math.exp(-1.0 / tau_e)

        # The eligibility matrix itself. Lazily initialized on first update.
        self._E: torch.Tensor | None = None

        # Counters for diagnostics.
        self._update_count: int = 0
        self._reset_count: int = 0

        # Optional history for debugging.
        self._history: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(
        self,
        pre_trace: torch.Tensor,
        post_trace: torch.Tensor,
    ) -> None:
        """Update the eligibility matrix with new trace values.

        Parameters
        ----------
        pre_trace : torch.Tensor
            Shape ``(batch, in_features)``. Pre-synaptic trace from the
            neuron (already maintained by ``LIFNodeWithTrace``).
        post_trace : torch.Tensor
            Shape ``(batch, out_features)``. Post-synaptic trace.

        Raises
        ------
        ValueError
            If the shapes of the traces do not match the configured
            ``in_features`` and ``out_features``.
        RuntimeError
            If the trace tensors are on different devices, or if the
            eligibility matrix was previously created on a different
            device.
        """
        # --- Shape checks ------------------------------------------------
        if pre_trace.dim() != 2 or post_trace.dim() != 2:
            raise ValueError(
                f"Traces must be 2D (batch, features). Got shapes "
                f"{tuple(pre_trace.shape)} and {tuple(post_trace.shape)}"
            )
        if pre_trace.shape[-1] != self.in_features:
            raise ValueError(
                f"pre_trace has {pre_trace.shape[-1]} features but "
                f"EligibilityTrace expects {self.in_features}"
            )
        if post_trace.shape[-1] != self.out_features:
            raise ValueError(
                f"post_trace has {post_trace.shape[-1]} features but "
                f"EligibilityTrace expects {self.out_features}"
            )
        if pre_trace.shape[0] != post_trace.shape[0]:
            raise ValueError(
                f"Batch sizes differ: pre={pre_trace.shape[0]}, "
                f"post={post_trace.shape[0]}"
            )

        # --- Device checks ----------------------------------------------
        if pre_trace.device != post_trace.device:
            raise RuntimeError(
                f"pre_trace and post_trace on different devices: "
                f"{pre_trace.device} vs {post_trace.device}"
            )

        # Detach to avoid graph accumulation: R-STDP updates are manual
        # (no_grad), and holding onto the graph would leak memory.
        pre = pre_trace.detach()
        post = post_trace.detach()

        # --- Lazy initialization ----------------------------------------
        if self._E is None:
            self._E = torch.zeros(
                self.out_features,
                self.in_features,
                device=post.device,
                dtype=post.dtype,
            )
        elif self._E.device != post.device:
            raise RuntimeError(
                f"Eligibility matrix is on {self._E.device} but new traces "
                f"are on {post.device}. Call .to(device) explicitly before "
                f"changing devices."
            )
        elif self._E.dtype != post.dtype:
            # Cast E to match incoming traces. This happens once.
            self._E = self._E.to(dtype=post.dtype)

        # --- The actual update ------------------------------------------
        # Outer product averaged over the batch:
        #   post: (batch, out) -> (batch, out, 1)
        #   pre:  (batch, in)  -> (batch, 1, in)
        #   product: (batch, out, in) -> mean over batch -> (out, in)
        contribution = (
            post.unsqueeze(-1) * pre.unsqueeze(-2)
        ).mean(dim=0)

        self._E = self._decay * self._E + contribution

        self._update_count += 1

        if self.record_history:
            self._history.append(self._E.clone())

    def reset(self) -> None:
        """Zero the eligibility matrix.

        Call this at episode boundaries. The matrix keeps its shape and
        device; only the values are reset. This is cheaper than
        reallocating and avoids accidental shape changes.
        """
        if self._E is not None:
            self._E.zero_()
        self._reset_count += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def eligibility(self) -> torch.Tensor | None:
        """The current eligibility matrix, or None if never updated."""
        return self._E

    def to(self, device: torch.device | str) -> "EligibilityTrace":
        """Move the eligibility matrix to a device. Returns self."""
        if self._E is not None:
            self._E = self._E.to(device)
        return self

    def diagnostics(self) -> dict[str, Any]:
        """Return a dict of useful numbers for debugging.

        Use this when you suspect the eligibility is not doing what you
        expect. The keys are:

        - ``initialized``: whether E has been created yet.
        - ``shape``: E's shape.
        - ``device``, ``dtype``: E's tensor properties.
        - ``mean``, ``abs_mean``, ``abs_max``: summary statistics.
        - ``frac_nonzero``: fraction of entries that are not exactly 0.
        - ``update_count``, ``reset_count``: counters.
        - ``all_zero``: True if every entry is exactly 0 (a red flag:
          the network is not generating any eligibility, which means no
          learning will happen).
        """
        info: dict[str, Any] = {
            "initialized": self._E is not None,
            "shape": tuple(self._E.shape) if self._E is not None else None,
            "device": str(self._E.device) if self._E is not None else None,
            "dtype": str(self._E.dtype) if self._E is not None else None,
            "tau_e": self.tau_e,
            "decay": self._decay,
            "update_count": self._update_count,
            "reset_count": self._reset_count,
        }
        if self._E is not None:
            info["mean"] = float(self._E.mean().item())
            info["abs_mean"] = float(self._E.abs().mean().item())
            info["abs_max"] = float(self._E.abs().max().item())
            info["frac_nonzero"] = float(
                (self._E != 0).float().mean().item()
            )
            info["all_zero"] = bool((self._E == 0).all().item())
        return info


# ======================================================================
# TracedLinear
# ======================================================================


class TracedLinear(nn.Module):
    """A Linear + LIF block with eligibility trace accumulation.

    This module is a drop-in replacement for the standard SpikingJelly
    pattern::

        nn.Sequential(nn.Linear(in_f, out_f), neuron.LIFNode(tau=2.0))

    The forward pass returns spikes of shape ``(batch, out_features)``,
    identical to the standard pattern. In addition, it accumulates an
    eligibility trace on every call, which can later be used to apply an
    R-STDP weight update via ``apply_reward()``.

    Notes
    -----
    The eligibility matrix has shape ``(out_features, in_features)``,
    matching the linear layer's weight matrix. To compute it, we need:

    - A trace of the **linear input** ``x`` (shape ``(batch, in)``).
      This is maintained by this module directly.
    - A trace of the **spike output** (shape ``(batch, out)``). This is
      maintained by the LIF neuron as ``post_trace``.

    The neuron's own ``pre_trace`` is a trace of the neuron's *input*,
    which is the linear output, not the linear input. That trace is not
    used here.

    Parameters
    ----------
    in_features : int
        Input dimension.
    out_features : int
        Output dimension.
    tau : float, optional
        LIF membrane time constant. Default 2.0.
    tau_trace : float, optional
        Time constant for the pre/post traces. Default 20.0.
    tau_e : float, optional
        Time constant for the eligibility decay. Default 50.0.
    bias : bool, optional
        Whether to include a bias in the linear layer. Default True.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: float = 2.0,
        tau_trace: float = 20.0,
        tau_e: float = 50.0,
        bias: bool = True,
        v_threshold: float = 1.0,
        init_snn: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau_trace = tau_trace
        self._trace_decay = 1.0 - (1.0 / tau_trace)

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.neuron = LIFNodeWithTrace(tau=tau, tau_trace=tau_trace, v_threshold=v_threshold)
        self.eligibility = EligibilityTrace(
            in_features=in_features,
            out_features=out_features,
            tau_e=tau_e,
        )
        
        # Apply SNN-specific weight initialization if requested.
        if init_snn:
            from spiking_rl.models.initialization import snn_weight_init
            snn_weight_init(self.linear, v_threshold=v_threshold)

        # Pre-synaptic input trace (trace of the linear input x).
        # Lazily initialized on the first forward pass.
        self._input_trace: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: linear -> LIF, and update the eligibility trace.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, in_features)``.

        Returns
        -------
        torch.Tensor
            Spikes of shape ``(batch, out_features)``.
        """
        # Linear pre-activation (input to the neuron).
        pre_current = self.linear(x)

        # Spiking dynamics. Also updates the neuron's post_trace.
        spikes = self.neuron(pre_current)

        # Maintain the input trace ourselves, using x directly.
        # Detach to avoid holding the graph across steps.
        x_detached = x.detach()
        if self._input_trace is None:
            self._input_trace = torch.zeros_like(x_detached)
        self._input_trace = (
            self._trace_decay * self._input_trace
            + (1.0 - self._trace_decay) * x_detached
        )

        # Eligibility update: outer(input_trace, post_trace).
        self.eligibility.update(
            self._input_trace, self.neuron.post_trace
        )

        return spikes

    def reset(self) -> None:
        """Reset neuron state, traces, and eligibility matrix."""
        self.neuron.reset()
        self.eligibility.reset()
        if self._input_trace is not None:
            self._input_trace.zero_()

    @torch.no_grad()
    def apply_reward(
        self,
        reward_signal: float | torch.Tensor,
        lr: float,
        baseline: float = 0.0,
        clip_weights: float | None = 5.0,
    ) -> float:
        """Apply the R-STDP weight update: Δw = lr * (R - baseline) * E."""
        if self.eligibility.eligibility is None:
            return 0.0

        if isinstance(reward_signal, torch.Tensor):
            r = float(reward_signal.detach().item())
        else:
            r = float(reward_signal)

        effective_reward = r - baseline

        E = self.eligibility.eligibility
        delta = lr * effective_reward * E

        self.linear.weight.add_(delta)

        if clip_weights is not None:
            self.linear.weight.clamp_(-clip_weights, clip_weights)

        return float(delta.norm().item())

    def diagnostics(self) -> dict[str, Any]:
        """Return per-layer diagnostics for debugging."""
        info = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "weight_mean": float(self.linear.weight.mean().item()),
            "weight_abs_mean": float(self.linear.weight.abs().mean().item()),
            "weight_abs_max": float(self.linear.weight.abs().max().item()),
            "eligibility": self.eligibility.diagnostics(),
        }
        if self._input_trace is not None:
            info["input_trace_abs_mean"] = float(
                self._input_trace.abs().mean().item()
            )
        return info