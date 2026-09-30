"""
Actor-critic spiking network for reward-modulated RL.

Architecture
------------
Two separate spiking networks:

- **Actor** — stack of ``TracedLinear`` layers, ending in ``num_actions``
  output neurons. Actions are selected by softmax-sampling the spike
  counts accumulated over ``T`` internal time steps. All actor layers
  maintain eligibility traces and are updated by R-STDP.

- **Critic** — stack of ``TracedLinear`` hidden layers, ending in a
  ``Linear → NonSpikingLIFNode`` head that outputs a scalar value
  estimate. The critic is trained by backprop on the TD-error MSE loss.
  Its hidden layers also maintain eligibility, so a future switch to
  TD-LTP for the critic is possible without architectural change.

Why no shared encoder
---------------------
Sharing an encoder between actor and critic would require that encoder's
synapses be updated by two different signals (R-STDP from the actor and
backprop from the critic). That introduces a "which learning rule owns
this synapse" problem with no clean answer. Keeping the two networks
separate mirrors the SpikingJelly A2C example and the BSVogler actor-
critic framework, and avoids this issue entirely.

Why the actor's output is spiking
---------------------------------
For R-STDP to be meaningful, the actor's output layer must emit spikes
so that its post-synaptic trace reflects discrete events. The chosen
action is read out from the spike counts of the output neurons over T
steps.

Why the critic's output is non-spiking
--------------------------------------
The critic needs to output a continuous value estimate. ``NonSpikingLIFNode``
accumulates its membrane potential and returns it without ever firing,
which is exactly the pattern used in the SpikingJelly A2C example.

References
----------
- SpikingJelly A2C CartPole example (actor-critic structure).
- BSVogler/SNN-RL (actor-critic with spiking actor and value critic).
- Potjans, Diesmann, Morrison (2011), "A spiking neural network model
  of an actor-critic learning agent", Neural Computation 23(2):269-328.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spiking_rl.learning.eligibility_trace import TracedLinear
from spiking_rl.models.lif import NonSpikingLIFNode


# ======================================================================
# Actor
# ======================================================================


class Actor(nn.Module):
    """Spiking actor network that outputs action spike counts.

    Structure: ``obs → TracedLinear(...) → ... → TracedLinear(num_actions)``.

    On ``forward``, the observation is passed through the stack ``steps``
    times. The returned value is the spike count of each output neuron
    accumulated over those steps. Higher spike count = stronger vote for
    that action.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation vector.
    num_actions : int
        Number of discrete actions. The output layer has this many
        neurons, one per action.
    hidden_dims : tuple of int
        Hidden layer sizes. Default ``(64, 64)``.
    tau, tau_trace, tau_e : float
        Time constants passed to ``TracedLinear``.
    obs_scale : float
        Observation is multiplied by this before entering the network.
        Needed because LIF neurons with ``v_threshold=1.0`` require
        sufficiently strong input currents to fire at all.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dims: tuple[int, ...] = (64, 64),
        tau: float = 2.0,
        tau_trace: float = 20.0,
        tau_e: float = 50.0,
        obs_scale: float = 20.0,
        v_threshold: float = 1.0,
        output_v_threshold: float = 1.0,
        init_snn: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.obs_scale = obs_scale
        self.v_threshold = v_threshold
        self.output_v_threshold = output_v_threshold
        self.init_snn = init_snn

        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(
                TracedLinear(
                    prev, h,
                    tau=tau, tau_trace=tau_trace, tau_e=tau_e,
                    v_threshold=v_threshold, init_snn=init_snn,
                )
            )
            prev = h
        # Output layer: num_actions spiking neurons.
        layers.append(
            TracedLinear(
                prev, num_actions,
                tau=tau, tau_trace=tau_trace, tau_e=tau_e,
                v_threshold=output_v_threshold, init_snn=init_snn,
            )
        )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        obs: torch.Tensor,
        steps: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the actor for ``steps`` internal time steps.

        Parameters
        ----------
        obs : torch.Tensor
            Shape ``(batch, obs_dim)``.
        steps : int
            Number of internal time steps per decision.

        Returns
        -------
        spike_counts : torch.Tensor
            Shape ``(batch, num_actions)``. Total spikes per output
            neuron over the ``steps`` internal steps.
        last_spikes : torch.Tensor
            Shape ``(batch, num_actions)``. Spikes at the final step.
        """
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")

        x = obs * self.obs_scale
        spike_counts: torch.Tensor | None = None
        last_spikes: torch.Tensor | None = None
        for _ in range(steps):
            h = x
            for layer in self.layers:
                h = layer(h)
            last_spikes = h
            spike_counts = h if spike_counts is None else spike_counts + h
        return spike_counts, last_spikes

    def reset(self) -> None:
        """Reset every layer's neuron state, traces, and eligibility."""
        for layer in self.layers:
            layer.reset()

    def traced_layers(self) -> list[TracedLinear]:
        """Return the list of ``TracedLinear`` layers (for the learner)."""
        return list(self.layers)


# ======================================================================
# Critic
# ======================================================================


class Critic(nn.Module):
    """Value-network with spiking hidden layers and non-spiking output.

    Structure: ``obs → TracedLinear(...) → ... → Linear(1) → NonSpikingLIFNode``.

    The final non-spiking neuron accumulates membrane potential over the
    internal steps and returns it as the value estimate. The hidden
    ``TracedLinear`` layers maintain eligibility (available for future
    TD-LTP use), but the critic is trained by backprop on the TD-error
    MSE loss in our design.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation.
    hidden_dims : tuple of int
        Hidden layer sizes. Default ``(64, 64)``.
    tau, tau_trace, tau_e : float
        Time constants passed to ``TracedLinear``.
    obs_scale : float
        Same scaling as the actor.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: tuple[int, ...] = (64, 64),
        tau: float = 2.0,
        tau_trace: float = 20.0,
        tau_e: float = 50.0,
        obs_scale: float = 20.0,
        v_threshold: float = 1.0,
        init_snn: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.obs_scale = obs_scale
        self.v_threshold = v_threshold
        self.init_snn = init_snn

        hidden = []
        prev = obs_dim
        for h in hidden_dims:
            hidden.append(
                TracedLinear(
                    prev, h,
                    tau=tau, tau_trace=tau_trace, tau_e=tau_e,
                    v_threshold=v_threshold,
                    init_snn=init_snn,
                )
            )
            prev = h
        self.hidden = nn.ModuleList(hidden)

        self.output_linear = nn.Linear(prev, 1)
        self.output_neuron = NonSpikingLIFNode(tau=tau)

    def forward(
        self,
        obs: torch.Tensor,
        steps: int = 5,
    ) -> torch.Tensor:
        """Run the critic for ``steps`` internal time steps.

        Parameters
        ----------
        obs : torch.Tensor
            Shape ``(batch, obs_dim)``.
        steps : int
            Number of internal time steps.

        Returns
        -------
        value : torch.Tensor
            Shape ``(batch, 1)``. Membrane potential of the output
            neuron after the final internal step.
        """
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")

        x = obs * self.obs_scale
        v: torch.Tensor | None = None
        for _ in range(steps):
            h = x
            for layer in self.hidden:
                h = layer(h)
            v = self.output_neuron(self.output_linear(h))
        return v

    def reset(self) -> None:
        """Reset every hidden neuron and the output neuron."""
        for layer in self.hidden:
            layer.reset()
        self.output_neuron.reset()

    def traced_layers(self) -> list[TracedLinear]:
        """Return the hidden ``TracedLinear`` layers."""
        return list(self.hidden)


# ======================================================================
# ActorCritic (composite)
# ======================================================================


class ActorCritic(nn.Module):
    """Composite actor + critic with helpers for action selection.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation.
    num_actions : int
        Number of discrete actions.
    actor_hidden : tuple of int
        Actor hidden layer sizes. Default ``(64, 64)``.
    critic_hidden : tuple of int
        Critic hidden layer sizes. Default ``(64, 64)``.
    tau, tau_trace, tau_e : float
        Time constants passed to ``TracedLinear``.
    obs_scale : float
        Observation scaling factor. Default 20.0.
    decision_steps : int
        Number of internal steps per decision. Default 5.
    action_temperature : float
        Softmax temperature for sampling actions from spike counts.
        Lower = more greedy. Default 1.0.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        actor_hidden: tuple[int, ...] = (64, 64),
        critic_hidden: tuple[int, ...] = (64, 64),
        tau: float = 2.0,
        tau_trace: float = 20.0,
        tau_e: float = 50.0,
        obs_scale: float = 20.0,
        decision_steps: int = 5,
        action_temperature: float = 1.0,
        v_threshold: float = 1.0,
        output_v_threshold: float = 1.0,
        init_snn: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.decision_steps = decision_steps
        self.action_temperature = action_temperature

        self.actor = Actor(
            obs_dim=obs_dim,
            num_actions=num_actions,
            hidden_dims=actor_hidden,
            tau=tau,
            tau_trace=tau_trace,
            tau_e=tau_e,
            obs_scale=obs_scale,
            v_threshold=v_threshold,
            output_v_threshold=output_v_threshold,
            init_snn=init_snn,
        )
        self.critic = Critic(
            obs_dim=obs_dim,
            hidden_dims=critic_hidden,
            tau=tau,
            tau_trace=tau_trace,
            tau_e=tau_e,
            obs_scale=obs_scale,
            v_threshold=v_threshold,
            init_snn=init_snn,
        )

        # Bookkeeping for diagnostics.
        self._last_actor_spike_counts: torch.Tensor | None = None
        self._last_critic_value: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward_actor(
        self,
        obs: torch.Tensor,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Run the actor and return spike counts of shape (batch, num_actions)."""
        s = self.decision_steps if steps is None else steps
        spike_counts, _ = self.actor(obs, steps=s)
        self._last_actor_spike_counts = spike_counts.detach()
        return spike_counts

    def forward_critic(
        self,
        obs: torch.Tensor,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Run the critic and return value estimate of shape (batch, 1)."""
        s = self.decision_steps if steps is None else steps
        value = self.critic(obs, steps=s)
        self._last_critic_value = value.detach()
        return value

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def sample_action(
        self,
        obs: torch.Tensor,
        steps: int | None = None,
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from the actor's spike counts.

        Parameters
        ----------
        obs : torch.Tensor
            Shape ``(batch, obs_dim)``.
        steps : int or None
            Internal steps. If None, uses ``self.decision_steps``.
        temperature : float or None
            Softmax temperature. If None, uses ``self.action_temperature``.

        Returns
        -------
        actions : torch.LongTensor
            Shape ``(batch,)``. Sampled action indices.
        probs : torch.Tensor
            Shape ``(batch, num_actions)``. Action probabilities used
            for sampling.
        spike_counts : torch.Tensor
            Shape ``(batch, num_actions)``. Raw spike counts.
        """
        spike_counts = self.forward_actor(obs, steps=steps)
        t = self.action_temperature if temperature is None else temperature
        if t <= 0:
            raise ValueError(f"temperature must be positive, got {t}")

        # Softmax over spike counts / temperature.
        probs = torch.softmax(spike_counts / t, dim=-1)
        actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return actions, probs, spike_counts

    def greedy_action(
        self,
        obs: torch.Tensor,
        steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the argmax action and the spike counts."""
        spike_counts = self.forward_actor(obs, steps=steps)
        actions = spike_counts.argmax(dim=-1)
        return actions, spike_counts

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset neuron state, traces, and eligibility for both networks.

        Call at episode boundaries. Weights are preserved.
        """
        self.actor.reset()
        self.critic.reset()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def actor_layers(self) -> list[TracedLinear]:
        return self.actor.traced_layers()

    def critic_layers(self) -> list[TracedLinear]:
        return self.critic.traced_layers()

    def diagnostics(self) -> dict[str, Any]:
        """Return spike statistics and per-layer eligibility summaries.

        Keys
        ----
        - ``actor_spike_mean``, ``actor_spike_max``: summary of the last
          actor spike count tensor. If ``actor_spike_mean`` is ~0, the
          actor is not firing and no learning will occur.
        - ``critic_value_mean``: mean of the last critic value estimate.
        - ``actor_layer_eligibility``, ``critic_layer_eligibility``:
          per-layer diagnostics from ``TracedLinear``.
        """
        info: dict[str, Any] = {}

        if self._last_actor_spike_counts is not None:
            info["actor_spike_mean"] = float(
                self._last_actor_spike_counts.mean().item()
            )
            info["actor_spike_max"] = float(
                self._last_actor_spike_counts.max().item()
            )
        else:
            info["actor_spike_mean"] = None
            info["actor_spike_max"] = None

        if self._last_critic_value is not None:
            info["critic_value_mean"] = float(
                self._last_critic_value.mean().item()
            )
        else:
            info["critic_value_mean"] = None

        info["actor_layer_eligibility"] = [
            layer.diagnostics() for layer in self.actor_layers()
        ]
        info["critic_layer_eligibility"] = [
            layer.diagnostics() for layer in self.critic_layers()
        ]
        return info