"""
Reward-modulated three-factor learning rule for spiking RL.

This module provides the orchestrator that drives the actual weight
updates for both the actor (R-STDP, reward-modulated) and the critic
(TD-LTP, temporal-difference-modulated) of the spiking agent.

The update rule is the classical three-factor rule:

    Δw = lr * modulation_signal * E

where:

- ``E`` is the eligibility trace maintained by each ``TracedLinear``
  layer (see ``spiking_rl.learning.eligibility_trace``). It encodes the
  recent pre×post-synaptic coincidence history.
- ``modulation_signal`` is the third factor. It can be:

    * For the actor (R-STDP): ``reward - baseline``, where ``baseline``
      is an exponential moving average of recent rewards. Subtracting
      the baseline reduces variance and turns the raw reward into a
      reward-prediction-error-like signal.
    * For the critic (TD-LTP): the temporal-difference error
      ``r + γ V(s') - V(s)``, computed externally by the agent.

Both cases share the same update mechanism; only the source of the
modulation signal differs. This module cleanly separates the two
responsibilities:

1. ``RewardBaseline`` — tracks a running average of rewards.
2. ``ThreeFactorLearner`` — applies modulation signals to a set of
   ``TracedLinear`` layers and exposes diagnostics.

References
----------
- Izhikevich (2007), "Solving the distal reward problem through linkage
  of STDP and dopamine signaling", Cerebral Cortex 17(10):2443-2452.
- Frémaux & Gerstner (2016), "Neuromodulated STDP and theory of
  three-factor learning rules", Frontiers in Neural Circuits 9:85.
- Bellec et al. (2020), "A solution to the learning dilemma for
  recurrent networks of spiking neurons", Nature Communications 11:3625.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn

from spiking_rl.learning.eligibility_trace import TracedLinear


# ======================================================================
# RewardBaseline
# ======================================================================


class RewardBaseline:
    """Exponential moving average of rewards.

    Used as a subtractive baseline to reduce the variance of the
    R-STDP modulation signal. Without a baseline, a task with average
    reward ``μ`` would push all recent synapses in the same direction
    proportional to ``μ``, which biases learning toward the trivial
    "always do what I did" solution rather than toward "do what worked
    better than average".

    Parameters
    ----------
    alpha : float, optional
        Update rate of the moving average. ``alpha=0.01`` corresponds to
        a ~100-sample memory. Must be in ``(0, 1]``. Default is 0.01.
    initial : float, optional
        Initial baseline value. Default is 0.0.

    Attributes
    ----------
    value : float
        Current baseline estimate.
    """

    def __init__(self, alpha: float = 0.01, initial: float = 0.0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._value = float(initial)
        self._count = 0

    @property
    def value(self) -> float:
        return self._value

    def update(self, reward: float) -> None:
        """Fold a new reward into the moving average."""
        self._value = (1.0 - self.alpha) * self._value + self.alpha * float(reward)
        self._count += 1

    def modulate(self, reward: float) -> float:
        """Return ``reward - baseline`` (does not update the baseline)."""
        return float(reward) - self._value

    def reset(self) -> None:
        """Zero the baseline estimate and its counter."""
        self._value = 0.0
        self._count = 0

    def diagnostics(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "value": self._value,
            "count": self._count,
        }


# ======================================================================
# ThreeFactorLearner
# ======================================================================


class ThreeFactorLearner:
    """Applies three-factor weight updates to a set of TracedLinear layers.

    Given a scalar modulation signal ``m``, applies to every attached
    layer the update::

        Δw = lr * m * E

    where ``E`` is the layer's current eligibility matrix. Weights are
    updated in place (under ``torch.no_grad()``), and optionally clamped
    to a symmetric interval.

    The learner does not own any parameters itself; it orchestrates
    updates to layers that live inside the model. It is therefore a
    plain class, not an ``nn.Module``.

    Parameters
    ----------
    layers : iterable of TracedLinear
        The layers whose weights should be updated by this learner.
    lr : float, optional
        Learning rate. Default is 1e-3.
    use_baseline : bool, optional
        If True, ``step(reward)`` subtracts the running reward baseline
        before applying the update (R-STDP behavior). If False, the raw
        reward is used. Default is True.
    baseline_alpha : float, optional
        Update rate for the reward baseline. Default 0.01.
    baseline_initial : float, optional
        Initial baseline value. Default 0.0.
    clip_weights : float or None, optional
        If not None, clamp each layer's weight matrix to the symmetric
        interval ``[-clip_weights, clip_weights]`` after each update.
        Default is 5.0.

    Notes
    -----
    The learner deliberately maintains two ways of applying an update:

    - ``step(reward)`` — used by the actor. Handles baseline tracking
      internally.
    - ``apply_modulation(modulation)`` — used by the critic. Receives a
      pre-computed modulation signal (e.g., the TD error).

    Diagnostics are tracked per layer and aggregated. If a training run
    is not learning, ``diagnostics()`` will show whether updates are
    consistently zero (which would indicate zero eligibility, which in
    turn would indicate the neurons are not firing).
    """

    def __init__(
        self,
        layers: Iterable[TracedLinear],
        lr: float = 1e-3,
        use_baseline: bool = True,
        baseline_alpha: float = 0.01,
        baseline_initial: float = 0.0,
        clip_weights: float | None = 5.0,
    ) -> None:
        self.layers: list[TracedLinear] = list(layers)
        if not self.layers:
            raise ValueError("ThreeFactorLearner requires at least one layer")

        self.lr = float(lr)
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")

        self.use_baseline = bool(use_baseline)
        self.clip_weights = clip_weights

        self.baseline = RewardBaseline(
            alpha=baseline_alpha, initial=baseline_initial
        )

        # Diagnostics state.
        self._update_count = 0
        self._last_updates: list[float] = []
        self._cumulative_updates: list[float] = [0.0] * len(self.layers)

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def step(self, reward: float | torch.Tensor) -> float:
        """Process a reward and apply the corresponding updates.

        This is the actor-side entry point. It:

        1. Converts the reward to a Python float.
        2. Computes the modulation signal (``reward - baseline`` if
           ``use_baseline`` is True, otherwise ``reward``).
        3. Updates the baseline with the raw reward.
        4. Applies the modulation to all attached layers.

        Parameters
        ----------
        reward : float or scalar tensor
            The raw reward from the environment.

        Returns
        -------
        float
            Sum of the update magnitudes across all layers.
        """
        r = float(reward) if not isinstance(reward, torch.Tensor) else float(
            reward.detach().item()
        )

        # Compute modulation using the *current* baseline, then update
        # the baseline with the new reward.
        if self.use_baseline:
            modulation = self.baseline.modulate(r)
        else:
            modulation = r
        self.baseline.update(r)

        return self.apply_modulation(modulation)

    def apply_modulation(self, modulation: float) -> float:
        """Apply an explicit modulation signal to all layers.

        This is the critic-side entry point (used for TD errors) and also
        the low-level entry point for custom modulation signals.

        Parameters
        ----------
        modulation : float
            The modulation signal. For R-STDP this is ``reward -
            baseline``; for TD-LTP this is the TD error.

        Returns
        -------
        float
            Sum of the update magnitudes across all layers.
        """
        if not isinstance(modulation, (int, float)):
            raise TypeError(
                f"modulation must be a scalar float, got {type(modulation)}"
            )

        self._last_updates = []
        for i, layer in enumerate(self.layers):
            norm = layer.apply_reward(
                reward_signal=float(modulation),
                lr=self.lr,
                baseline=0.0,  # baseline handled at this level
                clip_weights=self.clip_weights,
            )
            self._last_updates.append(norm)
            self._cumulative_updates[i] += norm

        self._update_count += 1
        return float(sum(self._last_updates))

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_layers(self) -> None:
        """Reset neuron state and eligibility of all attached layers.

        Call this at episode boundaries. This zeros:

        - Each layer's neuron membrane potential and pre/post traces.
        - Each layer's input trace.
        - Each layer's eligibility matrix.

        The **weights** and the **baseline** are preserved.
        """
        for layer in self.layers:
            layer.reset()

    def reset_diagnostics(self) -> None:
        """Reset update counters without touching layer state."""
        self._update_count = 0
        self._last_updates = []
        self._cumulative_updates = [0.0] * len(self.layers)

    def reset_baseline(self) -> None:
        """Reset the reward baseline.

        Usually you do **not** want to call this between episodes, since
        the baseline is meant to track the long-run average reward. It is
        useful when starting a new phase of training or when switching
        to a task with a different reward scale.
        """
        self.baseline.reset()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        """Return a dict of useful numbers for debugging.

        Keys
        ----
        - ``lr``, ``use_baseline``, ``clip_weights``: configuration.
        - ``num_layers``: number of attached TracedLinear layers.
        - ``update_count``: how many times ``step`` or
          ``apply_modulation`` has been called.
        - ``last_updates``: per-layer magnitude of the most recent update.
          If all entries are 0.0, the eligibility is likely all-zero
          (neurons not firing).
        - ``cumulative_updates``: per-layer total update magnitude since
          the last diagnostics reset.
        - ``baseline``: baseline diagnostics dict.
        """
        return {
            "lr": self.lr,
            "use_baseline": self.use_baseline,
            "clip_weights": self.clip_weights,
            "num_layers": len(self.layers),
            "update_count": self._update_count,
            "last_updates": list(self._last_updates),
            "cumulative_updates": list(self._cumulative_updates),
            "baseline": self.baseline.diagnostics(),
        }


# ======================================================================
# Convenience helpers
# ======================================================================


def collect_traced_layers(model: nn.Module) -> list[TracedLinear]:
    """Walk a module tree and return every ``TracedLinear`` instance.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch module. Nested submodules are traversed.

    Returns
    -------
    list of TracedLinear
        Traced layers in the order they appear during tree traversal
        (which is the order they are registered as submodules).
    """
    return [m for m in model.modules() if isinstance(m, TracedLinear)]


def split_actor_critic_layers(
    model: nn.Module,
    actor_prefix: str = "actor",
    critic_prefix: str = "critic",
) -> tuple[list[TracedLinear], list[TracedLinear]]:
    """Split a model's TracedLinear layers into actor and critic groups.

    Layers are assigned to a group based on whether their fully
    qualified name starts with the given prefix.

    Parameters
    ----------
    model : nn.Module
        Model containing both actor and critic submodules.
    actor_prefix : str
        Prefix of names belonging to the actor.
    critic_prefix : str
        Prefix of names belonging to the critic.

    Returns
    -------
    actor_layers, critic_layers : tuple of list
        Two lists of TracedLinear instances.
    """
    actor_layers: list[TracedLinear] = []
    critic_layers: list[TracedLinear] = []

    for name, module in model.named_modules():
        if not isinstance(module, TracedLinear):
            continue
        if name.startswith(actor_prefix):
            actor_layers.append(module)
        elif name.startswith(critic_prefix):
            critic_layers.append(module)

    return actor_layers, critic_layers