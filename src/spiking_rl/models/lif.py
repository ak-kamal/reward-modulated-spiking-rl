"""
LIF neuron wrappers for reward-modulated spiking RL.

This module provides two classes built on top of SpikingJelly's
``neuron.LIFNode``:

1. ``LIFNodeWithTrace`` — a standard LIF neuron that additionally maintains
   decaying pre-synaptic and post-synaptic traces. These traces are the
   building blocks for the eligibility trace used in the three-factor
   R-STDP learning rule.

2. ``NonSpikingLIFNode`` — a LIF neuron that never fires. It returns its
   final membrane potential instead of spikes. This is the standard
   SpikingJelly pattern for producing continuous outputs (Q-values, action
   logits, value estimates) from a spiking network.

The design follows the SpikingJelly convention: we inherit from the
official ``LIFNode`` and extend it minimally, rather than reimplementing
LIF dynamics from scratch. This gives us correct membrane dynamics,
surrogate gradients, and state management for free.

References
----------
- SpikingJelly neuron tutorial (clock-driven neurons):
  https://spikingjelly.readthedocs.io/zh-cn/latest/legacy_tutorials/en/0_neuron.html
- SpikingJelly A2C actor-critic example (NonSpikingLIFNode pattern):
  https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.6/clock_driven_en/7_a2c_cart_pole.html
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate


class LIFNodeWithTrace(neuron.LIFNode):
    """A LIF neuron that maintains decaying pre- and post-synaptic traces.

    The eligibility trace for a synapse ``(i, j)`` is approximated as the
    product of a pre-synaptic trace (activity of neuron ``i``) and a
    post-synaptic trace (spikes of neuron ``j``), each decaying
    exponentially. This factorization is a standard approximation in
    three-factor learning rules (see e-prop, Bellec et al. 2020).

    Parameters
    ----------
    tau : float, optional
        Membrane time constant. Passed through to ``neuron.LIFNode``.
    tau_trace : float, optional
        Time constant for the pre- and post-synaptic trace decay.
        Default is 20.0 (20 time steps).
    v_threshold : float, optional
        Firing threshold. Default 1.0.
    v_reset : float, optional
        Reset voltage after a spike. Default 0.0.
    surrogate_function : callable, optional
        Surrogate gradient function for backpropagation. Default is
        ``surrogate.ATan()``.
    **kwargs
        Additional keyword arguments passed to ``neuron.LIFNode``.

    Attributes
    ----------
    pre_trace : torch.Tensor or None
        Decaying trace of the neuron's input (pre-synaptic activity).
        Shape matches the input to ``forward``. ``None`` before the first
        forward pass.
    post_trace : torch.Tensor or None
        Decaying trace of the neuron's spike output (post-synaptic
        activity). Shape matches the output of ``forward``.
    """

    def __init__(
        self,
        tau: float = 2.0,
        tau_trace: float = 20.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_function: Any = None,
        **kwargs: Any,
    ) -> None:
        if surrogate_function is None:
            surrogate_function = surrogate.ATan()

        super().__init__(
            tau=tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            **kwargs,
        )

        self.tau_trace = tau_trace

        # Trace buffers. Initialized to None; created on the first forward
        # pass once we know the input shape.
        self.pre_trace: torch.Tensor | None = None
        self.post_trace: torch.Tensor | None = None

        # Precompute the decay factor for efficiency.
        self._trace_decay = 1.0 - (1.0 / tau_trace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: integrate input, emit spikes, update traces.

        Parameters
        ----------
        x : torch.Tensor
            Input to the neuron (pre-synaptic current).

        Returns
        -------
        torch.Tensor
            Spike output (0 or 1) with the same shape as ``x``.
        """
        self._ensure_batch_shape(x)
        
        # Standard LIF dynamics and spike generation.
        spikes = super().forward(x)

        # Initialize traces on the first forward pass.
        if self.pre_trace is None:
            self.pre_trace = torch.zeros_like(x)
            self.post_trace = torch.zeros_like(spikes)

        # Update pre-synaptic trace: exponential decay + new input.
        # Using a simple leaky integrator.
        self.pre_trace = (
            self._trace_decay * self.pre_trace
            + (1.0 - self._trace_decay) * x
        )

        # Update post-synaptic trace: exponential decay + new spikes.
        self.post_trace = (
            self._trace_decay * self.post_trace
            + (1.0 - self._trace_decay) * spikes
        )

        return spikes

    def reset_traces(self) -> None:
        """Reset pre- and post-synaptic traces to zero.

        Call this between episodes or when the environment resets.
        """
        if self.pre_trace is not None:
            self.pre_trace = torch.zeros_like(self.pre_trace)
        if self.post_trace is not None:
            self.post_trace = torch.zeros_like(self.post_trace)

    def get_eligibility(self) -> torch.Tensor | None:
        """Compute the eligibility trace as the outer product of traces.

        Returns
        -------
        torch.Tensor or None
            Eligibility trace of shape ``(out_features, in_features)``,
            or ``None`` if no forward pass has been made yet.
        """
        if self.pre_trace is None or self.post_trace is None:
            return None

        # Outer product: (out_features,) x (in_features,) -> (out, in)
        # Then average over the batch dimension.
        pre = self.pre_trace  # shape: (batch, in_features)
        post = self.post_trace  # shape: (batch, out_features)

        # Expand and multiply, then average over batch.
        # pre: (batch, 1, in), post: (batch, out, 1)
        eligibility = (post.unsqueeze(-1) * pre.unsqueeze(-2)).mean(dim=0)

        return eligibility
    
    def reset(self) -> None:
        """Reset both the membrane potential and the eligibility traces.

        This overrides the parent ``LIFNode.reset()`` so that a single
        ``reset()`` call cleans up everything related to this neuron's
        state. Without this override, ``reset()`` would zero the voltage
        but leave ``pre_trace`` and ``post_trace`` at their last values,
        silently leaking state across episodes.
        """
        super().reset()
        self.reset_traces()
        
    def _ensure_batch_shape(self, x: torch.Tensor) -> None:
        """Reinitialize any cached batch-shaped state that no longer matches.

        SpikingJelly caches ``self.v`` across calls, and this class caches
        ``pre_trace`` and ``post_trace``. Each is created with the batch
        shape of its first forward call. If a later call has a different
        batch size, the cached tensors become incompatible with the new
        input and downstream arithmetic fails. This method reinitializes
        any such tensor to zeros of the correct shape, which is the
        semantically correct behavior — the previous batch's state is
        meaningless for a different batch.
        """
        if not isinstance(self.v, torch.Tensor) or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)

        if self.pre_trace is not None and self.pre_trace.shape != x.shape:
            self.pre_trace = torch.zeros_like(x)

        if self.post_trace is not None and self.post_trace.shape != x.shape:
            self.post_trace = torch.zeros_like(x)


class NonSpikingLIFNode(neuron.LIFNode):
    """A LIF neuron that never fires; returns membrane potential instead.

    This is the standard SpikingJelly pattern for producing continuous
    outputs from a spiking network. The neuron integrates input and
    updates its membrane potential, but it does not emit spikes and does
    not reset its voltage. The membrane potential is returned directly.

    This is used for the actor and critic output layers, where we need
    continuous values (action logits, state value) rather than binary
    spikes.

    Parameters
    ----------
    tau : float, optional
        Membrane time constant. Default 2.0.
    **kwargs
        Additional keyword arguments passed to ``neuron.LIFNode``.
    """

    def __init__(self, tau: float = 2.0, **kwargs: Any) -> None:
        # Use a very large finite threshold instead of inf, to avoid
        # type-casting issues in some SpikingJelly code paths.
        kwargs.setdefault("v_threshold", 1e6)
        super().__init__(tau=tau, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: integrate input, return membrane potential.

        Parameters
        ----------
        x : torch.Tensor
            Input to the neuron.

        Returns
        -------
        torch.Tensor
            Membrane potential after integration.
        """
        # Lazily initialize v as a tensor matching the input shape.
        # SpikingJelly's LIFNode starts with self.v = 0.0 (a float);
        # neuronal_charge requires a Tensor.
        if not isinstance(self.v, torch.Tensor):
            self.v = torch.zeros_like(x)
        
        # Charge the membrane potential.
        self.neuronal_charge(x)
        # Skip neuronal_fire() and neuronal_reset().
        return self.v


class NonSpikingLIFNodeWithTrace(NonSpikingLIFNode):
    """Non-spiking LIF neuron that also maintains eligibility traces.

    This combines the continuous-output behavior of ``NonSpikingLIFNode``
    with the trace maintenance of ``LIFNodeWithTrace``. Used for output
    layers that need both continuous values and trace-based learning.
    """

    def __init__(
        self,
        tau: float = 2.0,
        tau_trace: float = 20.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(tau=tau, **kwargs)
        self.tau_trace = tau_trace
        self.pre_trace: torch.Tensor | None = None
        self.post_trace: torch.Tensor | None = None
        self._trace_decay = 1.0 - (1.0 / tau_trace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: integrate, update traces, return membrane potential."""
        output = super().forward(x)  # membrane potential (continuous)

        if self.pre_trace is None:
            self.pre_trace = torch.zeros_like(x)
            self.post_trace = torch.zeros_like(output)

        self.pre_trace = (
            self._trace_decay * self.pre_trace
            + (1.0 - self._trace_decay) * x
        )
        self.post_trace = (
            self._trace_decay * self.post_trace
            + (1.0 - self._trace_decay) * output
        )

        return output

    def reset_traces(self) -> None:
        """Reset traces to zero."""
        if self.pre_trace is not None:
            self.pre_trace = torch.zeros_like(self.pre_trace)
        if self.post_trace is not None:
            self.post_trace = torch.zeros_like(self.post_trace)

    def get_eligibility(self) -> torch.Tensor | None:
        """Compute eligibility trace as outer product of pre and post traces."""
        if self.pre_trace is None or self.post_trace is None:
            return None
        pre = self.pre_trace
        post = self.post_trace
        return (post.unsqueeze(-1) * pre.unsqueeze(-2)).mean(dim=0)

    def reset(self) -> None:
        """Reset both the membrane potential and the eligibility traces."""
        super().reset()
        self.reset_traces()
        
class AdaptiveLIFNode(LIFNodeWithTrace):
    """LIF neuron with homeostatic threshold adaptation.

    The firing threshold slowly rises when the neuron fires more than a
    target rate and falls when it fires less. This is a well-documented
    homeostatic mechanism that prevents neurons from becoming permanently
    silent (a known failure mode in reward-modulated STDP networks).

    The adaptation is deliberately slow (much slower than the membrane
    time constant) to avoid oscillations, following the principle from
    the homeostatic plasticity literature that homeostatic processes
    must operate on a slower timescale than Hebbian plasticity.

    References
    ----------
    - Skorheim, Lonjers & Bazhenov (2014), PLoS ONE 9(3): e90821.
    - Geng & Li (2023), HoSNN, arXiv:2308.10373.
    - Hertag & Sprekeler (2020), Nature Communications 11:3358.
    """

    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        tau_trace: float = 20.0,
        tau_homeo: float = 2000.0,
        target_rate: float = 0.1,
        eta_threshold: float = 1e-4,
        threshold_min: float = 0.3,
        threshold_max: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tau=tau,
            tau_trace=tau_trace,
            v_threshold=v_threshold,
            **kwargs,
        )
        self.tau_homeo = tau_homeo
        self.target_rate = target_rate
        self.eta_threshold = eta_threshold
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max

        # Per-neuron firing-rate estimate, shape (out_features,).
        # Independent of batch size.
        self.rate_ema: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with homeostatic threshold adaptation."""
        # LIFNodeWithTrace.forward handles batch-shape consistency.
        spikes = super().forward(x)

        # Per-neuron firing rate for this batch: average over batch dim.
        batch_rate = spikes.mean(dim=0)  # shape (out_features,)

        if self.rate_ema is None:
            self.rate_ema = torch.zeros_like(batch_rate)

        # Slow exponential moving average.
        decay = 1.0 - (1.0 / self.tau_homeo)
        self.rate_ema = decay * self.rate_ema + (1.0 - decay) * batch_rate

        # Threshold adaptation: scalar update based on mean firing rate.
        rate = float(self.rate_ema.mean().item())
        delta = self.eta_threshold * (rate - self.target_rate)
        new_threshold = self.v_threshold + delta
        self.v_threshold = float(
            max(self.threshold_min, min(self.threshold_max, new_threshold))
        )

        return spikes

    def reset(self) -> None:
        """Reset membrane potential, traces, and rate estimate."""
        super().reset()
        if self.rate_ema is not None:
            self.rate_ema = torch.zeros_like(self.rate_ema)

    def homeostatic_diagnostics(self) -> dict[str, Any]:
        """Return threshold and rate diagnostics."""
        info: dict[str, Any] = {
            "v_threshold": self.v_threshold,
            "target_rate": self.target_rate,
            "tau_homeo": self.tau_homeo,
            "eta_threshold": self.eta_threshold,
        }
        if self.rate_ema is not None:
            info["rate_ema_mean"] = float(self.rate_ema.mean().item())
        return info
    
    def get_extra_state(self) -> dict[str, Any]:
        """Return state that PyTorch's default save/load doesn't capture.

        ``v_threshold`` and ``rate_ema`` are instance attributes that are
        modified during training but are not registered buffers. Without
        this override, saving a trained model loses the adapted threshold
        and the running rate estimate, and loading it silently reverts
        the network to its initial behavior.
        """
        return {
            "v_threshold": float(self.v_threshold),
            "rate_ema": (
                None if self.rate_ema is None else self.rate_ema.detach().clone()
            ),
        }

    def set_extra_state(self, state: Any) -> None:
        """Restore state saved by ``get_extra_state``."""
        if state is None:
            return
        self.v_threshold = float(state["v_threshold"])
        rate_ema = state["rate_ema"]
        self.rate_ema = None if rate_ema is None else rate_ema.clone()