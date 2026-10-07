"""
Training loop for the reward-modulated spiking actor-critic agent.

This module wires together everything we've built:

- ``GridWorldEnv``              — the environment.
- ``ActorCritic``               — the spiking actor and value critic.
- ``ThreeFactorLearner``        — the R-STDP orchestrator.
- ``TracedLinear``              — the traced spiking layers.

Design decisions
----------------
1. Third-factor choice for the actor is configurable:

   - ``"td_error"`` (default): the actor is modulated by the temporal-
     difference error ``r + gamma V(s') - V(s)``. This is the standard
     actor-critic formulation, and the critic provides a learned
     baseline through ``V(s)``.

   - ``"reward_baseline"``: the actor is modulated by
     ``reward - running_mean`` (classical R-STDP). This disconnects the
     actor from the critic's value estimates.

   Both are defensible; ``"td_error"`` is the default because we have a
   critic and it aligns with modern spiking actor-critic practice.

2. The critic is trained by backprop on the TD-error MSE loss. Its
   hidden layers maintain eligibility traces so a future switch to
   TD-LTP is architecturally possible, but they are not used here.

3. Two critic forward passes are made per environment step:
   one to compute ``V(s)`` and one to compute ``V(s')``. This advances
   the SNN's internal state at twice the environment rate, which is
   standard for online TD learning with stateful SNNs.

4. Episode boundaries reset neuron state and eligibility but NOT
   weights and NOT the reward baseline.

5. Every episode logs diagnostics (spike counts, TD error, update
   magnitudes, critic loss). Silent failure modes are caught early.

References
----------
- SpikingJelly A2C example (loop structure).
- Potjans, Diesmann, Morrison (2011), "A spiking neural network model
  of an actor-critic learning agent", Neural Computation 23(2):269-328.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spiking_rl.learning.r_stdp import ThreeFactorLearner
from spiking_rl.models.actor_critic import ActorCritic
from spiking_rl.models.initialization import snn_target_weight_norm


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class AgentConfig:
    """Hyperparameters for the spiking actor-critic agent.

    Every default here is documented with its rationale so that future
    changes can be justified against the original choice.
    """

    # --- RL setup ---
    gamma: float = 0.99
    """Discount factor. Standard value for episodic tasks."""

    # --- Network architecture ---
    actor_hidden: tuple[int, ...] = (64, 64)
    critic_hidden: tuple[int, ...] = (64, 64)
    tau: float = 2.0
    tau_trace: float = 20.0
    tau_e: float = 50.0
    obs_scale: float = 20.0
    v_threshold: float = 1.0
    output_v_threshold: float = 1.0
    init_snn: bool = True
    decision_steps: int = 5
    eval_decision_steps: int | None = None
    action_temperature: float = 1.0

    # --- Learning ---
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    third_factor: str = "td_error"  # or "reward_baseline"
    baseline_alpha: float = 0.01
    clip_weights: float = 2.0
    clip_grad_norm: float = 1.0
    
    # --- Homeostatic mechanisms ---
    use_adaptive_threshold: bool = True
    target_rate: float = 0.1
    eta_threshold: float = 1e-4
    normalize_actor_weights: bool = True
    weight_norm_target: float = 1.0
    
    # --- Sparsity mechanism ---
    inhibition_gain: float = 0.0

    # --- Bookkeeping ---
    log_every: int = 10


# ======================================================================
# Agent
# ======================================================================


class SpikingActorCriticAgent:
    """Online actor-critic agent with spiking actor and value critic.

    The agent owns:

    - The environment (passed in; the agent does not construct it).
    - The ``ActorCritic`` network.
    - A ``ThreeFactorLearner`` bound to the actor's traced layers.
    - An Adam optimizer bound to the critic's parameters.

    Parameters
    ----------
    env : gymnasium.Env
        A Gymnasium-compatible environment.
    config : AgentConfig or None
        Configuration. If None, uses defaults.
    """

    def __init__(
        self,
        env,
        config: AgentConfig | None = None,
    ) -> None:
        self.env = env
        self.config = config or AgentConfig()

        # Infer observation and action dimensions from the env.
        obs_dim = int(np.prod(env.observation_space.shape))
        num_actions = int(env.action_space.n)

        # Build the network.
        self.ac = ActorCritic(
            obs_dim=obs_dim,
            num_actions=num_actions,
            actor_hidden=self.config.actor_hidden,
            critic_hidden=self.config.critic_hidden,
            tau=self.config.tau,
            tau_trace=self.config.tau_trace,
            tau_e=self.config.tau_e,
            obs_scale=self.config.obs_scale,
            decision_steps=self.config.decision_steps,
            action_temperature=self.config.action_temperature,
            v_threshold=self.config.v_threshold,
            output_v_threshold=self.config.output_v_threshold,
            init_snn=self.config.init_snn,
            use_adaptive_threshold=self.config.use_adaptive_threshold,
            target_rate=self.config.target_rate,
            eta_threshold=self.config.eta_threshold,
            inhibition_gain=self.config.inhibition_gain,
        )

        # Actor learner uses the actor's traced layers.
        self.actor_learner = ThreeFactorLearner(
            layers=self.ac.actor_layers(),
            lr=self.config.actor_lr,
            use_baseline=(self.config.third_factor == "reward_baseline"),
            baseline_alpha=self.config.baseline_alpha,
            clip_weights=self.config.clip_weights,
        )

        # Critic optimizer over ALL critic parameters (including output head).
        self.critic_optimizer = torch.optim.Adam(
            self.ac.critic.parameters(),
            lr=self.config.critic_lr,
        )

        # Diagnostic counters.
        self._total_episodes = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert a numpy observation to a batched float32 tensor."""
        return torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

    def _balance_actor_weights(self) -> None:
        """Apply output balancing to all actor layers.

        Rescales each layer's output neuron weight vectors to a target
        L2 norm after each reward-modulated update. This prevents the
        rewarded STDP rule from systematically strengthening a small
        number of output neurons until they dominate the layer. The output layer 
        is excluded because normalizing its weights to
        a fixed L2 norm suppresses the drive needed to reach threshold
        when the preceding hidden layer has many neurons. Hidden layers
        still benefit from balancing to prevent representational collapse. The target 
        L2 norm matches what the SNN-specific initialization
        would produce, so balancing constrains weight *drift* without
        fighting the initialization.
        """
        target = snn_target_weight_norm(v_threshold=1.0)  # ≈ 2.51
        for layer in self.ac.actor_layers()[:-1]:
            layer.normalize_weights(target_norm=target)

    @torch.no_grad()
    def greedy_action_diversity(self, n_samples: int = 64) -> int:
        """Return the number of distinct greedy actions chosen over random observations.

        A value of 1 means the policy is state-invariant (bad).
        A value close to num_actions means the policy uses all actions (good).
        """
        self.ac.reset()
        
        # Use eval resolution for diversity measurement too.
        steps = self.config.eval_decision_steps or self.config.decision_steps

        observations = []
        env = self.env
        obs, _ = env.reset(seed=12345)
        for _ in range(n_samples):
            observations.append(obs)
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                obs, _ = env.reset(seed=12345 + len(observations))

        obs_batch = torch.as_tensor(np.array(observations), dtype=torch.float32)
        with torch.no_grad():
            spike_counts, _ = self.ac.actor(obs_batch, steps=steps)
            greedy_actions = spike_counts.argmax(dim=-1)
        return int(len(greedy_actions.unique()))
    
    @torch.no_grad()
    def measure_sparsity(
        self,
        n_episodes: int = 50,
        seed_start: int = 10_000,
        greedy: bool = False,
    ) -> dict[str, float]:
        """Measure spikes per step during actual policy rollouts.

        Returns a dict with:

        - ``spikes_per_step``: total spikes summed across output neurons,
          averaged over decision steps.
        - ``spikes_per_neuron_per_step``: the same, divided by the number
          of output neurons. This is the normalised sparsity measure.

        This replaces the earlier approach of feeding a fixed batch of
        identical observations, which was biased by the state at (0, 0).
        """
        num_output_neurons = self.ac.actor.num_actions

        total_spikes = 0.0
        total_steps = 0

        for i in range(n_episodes):
            obs, _ = self.env.reset(seed=seed_start + i)
            obs_t = self._to_tensor(obs)
            self.ac.reset()

            for _ in range(10_000):
                if greedy:
                    action, spike_counts = self.ac.greedy_action(obs_t)
                else:
                    action, _, spike_counts = self.ac.sample_action(obs_t)

                total_spikes += float(spike_counts.sum().item())
                total_steps += 1

                next_obs, _, term, trunc, _ = self.env.step(int(action.item()))
                obs_t = self._to_tensor(next_obs)
                if term or trunc:
                    break

        eps = 1e-8
        per_step = total_spikes / max(1, total_steps)
        return {
            "spikes_per_step": per_step,
            "spikes_per_neuron_per_step": per_step / num_output_neurons,
            "total_steps": float(total_steps),
            "n_episodes": float(n_episodes),
        }
    # ------------------------------------------------------------------
    # Episode loops
    # ------------------------------------------------------------------

    def train_episode(self, seed: int | None = None) -> dict[str, Any]:
        """Run one training episode.

        Returns a dict of episode metrics.
        """
        obs, _ = self.env.reset(seed=seed)
        obs_t = self._to_tensor(obs)

        # Reset neuron state and eligibility (NOT weights, NOT baseline).
        self.ac.reset()

        episode_reward = 0.0
        episode_steps = 0

        actor_spikes = []
        values = []
        td_errors = []
        actor_updates = []
        critic_losses = []

        for step in range(10_000):  # hard cap; env's max_steps should trigger earlier
            # --- Forward passes -----------------------------------------
            # Actor: sample action. This also updates actor eligibility.
            action, probs, spike_counts = self.ac.sample_action(obs_t)
            # Critic: value of current state. With grad for backprop.
            value = self.ac.forward_critic(obs_t)

            # --- Environment step ---------------------------------------
            next_obs, reward, terminated, truncated, _ = self.env.step(
                int(action.item())
            )
            done = bool(terminated or truncated)
            episode_reward += reward
            episode_steps += 1

            # --- Compute next value (no grad) ---------------------------
            next_obs_t = self._to_tensor(next_obs)
            with torch.no_grad():
                next_value = self.ac.forward_critic(next_obs_t)

            # --- TD target and error ------------------------------------
            td_target = reward + self.config.gamma * next_value * (
                1.0 - float(terminated)
            )
            td_error_value = float(
                (td_target - value.detach()).mean().item()
            )

            # --- Actor update (R-STDP or baseline) ----------------------
            if self.config.third_factor == "td_error":
                actor_update_norm = self.actor_learner.apply_modulation_centered(
                    td_error_value
                )
            elif self.config.third_factor == "reward_baseline":
                actor_update_norm = self.actor_learner.step(reward)
            else:
                raise ValueError(
                    f"Unknown third_factor: {self.config.third_factor}"
                )
            self._balance_actor_weights()

            # --- Critic update (backprop) -------------------------------
            critic_loss = F.mse_loss(value, td_target.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.ac.critic.parameters(),
                    max_norm=self.config.clip_grad_norm,
                )
            self.critic_optimizer.step()

            # --- Log metrics --------------------------------------------
            actor_spikes.append(float(spike_counts.sum().item()))
            values.append(float(value.detach().mean().item()))
            td_errors.append(td_error_value)
            actor_updates.append(actor_update_norm)
            critic_losses.append(float(critic_loss.item()))

            # --- Advance ------------------------------------------------
            obs_t = next_obs_t
            if done:
                break

        self._total_episodes += 1
        

        return {
            "episode": self._total_episodes,
            "total_reward": float(episode_reward),
            "steps": episode_steps,
            "success": bool(terminated and not truncated),
            "truncated": bool(truncated),
            "mean_actor_spikes": float(np.mean(actor_spikes)) if actor_spikes else 0.0,
            "mean_value": float(np.mean(values)) if values else 0.0,
            "mean_td_error": float(np.mean(td_errors)) if td_errors else 0.0,
            "mean_actor_update": float(np.mean(actor_updates)) if actor_updates else 0.0,
            "mean_critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "baseline": float(self.actor_learner.baseline.value),
            "greedy_action_diversity": self.greedy_action_diversity(),
        }

    @torch.no_grad()
    def evaluate_episode(
        self,
        seed: int | None = None,
        greedy: bool = True,
    ) -> dict[str, Any]:
        """Run one evaluation episode without any learning.

        Uses `config.eval_decision_steps` if set, otherwise falls back to
        `config.decision_steps`. Adaptive thresholds are frozen during
        evaluation so that a different number of internal steps doesn't
        perturb the learned homeostatic state.
        """
        obs, _ = self.env.reset(seed=seed)
        obs_t = self._to_tensor(obs)

        self.ac.reset()

        # Choose eval-time decision steps.
        steps = self.config.eval_decision_steps or self.config.decision_steps

        # Freeze adaptive thresholds during evaluation.
        frozen = []
        for module in self.ac.modules():
            if hasattr(module, "eta_threshold"):
                frozen.append((module, module.eta_threshold))
                module.eta_threshold = 0.0

        try:
            episode_reward = 0.0
            episode_steps = 0
            terminated = False
            truncated = False

            for step in range(10_000):
                if greedy:
                    action, _ = self.ac.greedy_action(obs_t, steps=steps)
                else:
                    action, _, _ = self.ac.sample_action(obs_t, steps=steps)

                next_obs, reward, terminated, truncated, _ = self.env.step(
                    int(action.item())
                )
                done = bool(terminated or truncated)
                episode_reward += reward
                episode_steps += 1

                obs_t = self._to_tensor(next_obs)
                if done:
                    break
        finally:
            # Restore thresholds regardless of outcome.
            for module, eta in frozen:
                module.eta_threshold = eta

        return {
            "total_reward": float(episode_reward),
            "steps": episode_steps,
            "success": bool(terminated and not truncated),
            "truncated": bool(truncated),
        }

    # ------------------------------------------------------------------
    # Training and evaluation loops
    # ------------------------------------------------------------------

    def train(
        self,
        n_episodes: int,
        seed_start: int = 0,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Train for ``n_episodes`` episodes. Returns the history."""
        history = []
        for i in range(n_episodes):
            metrics = self.train_episode(seed=seed_start + i)
            history.append(metrics)
            if verbose and (i + 1) % self.config.log_every == 0:
                recent = history[-self.config.log_every:]
                mean_reward = float(np.mean([m["total_reward"] for m in recent]))
                mean_spikes = float(np.mean([m["mean_actor_spikes"] for m in recent]))
                mean_td = float(np.mean([m["mean_td_error"] for m in recent]))
                mean_upd = float(np.mean([m["mean_actor_update"] for m in recent]))
                mean_div = float(np.mean([m["greedy_action_diversity"] for m in recent]))
                layer_rates = self.layer_firing_rates()
                rate_str = " ".join(
                    f"r{i}={v:.3f}" for i, v in enumerate(layer_rates.values())
                )
                print(
                    f"Episode {i + 1:5d}/{n_episodes} | "
                    f"reward={mean_reward:+.3f} | "
                    f"spikes={mean_spikes:8.3f} | "
                    f"td_err={mean_td:+.4f} | "
                    f"act_upd={mean_upd:.4f} | "
                    f"div={mean_div:.1f} | "
                    f"{rate_str} | "
                    f"baseline={self.actor_learner.baseline.value:+.3f}"
                )
        return history

    def evaluate(
        self,
        n_episodes: int,
        seed_start: int = 10_000,
        greedy: bool = True,
    ) -> dict[str, float]:
        """Evaluate for ``n_episodes`` episodes with no learning."""
        results = []
        for i in range(n_episodes):
            results.append(
                self.evaluate_episode(seed=seed_start + i, greedy=greedy)
            )
        return {
            "mean_reward": float(np.mean([r["total_reward"] for r in results])),
            "success_rate": float(np.mean([r["success"] for r in results])),
            "mean_steps": float(np.mean([r["steps"] for r in results])),
            "n_episodes": n_episodes,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        """Return a diagnostic dict for the current state of the agent."""
        # Per-layer threshold and weight-norm summaries.
        actor_layer_info = []
        for layer in self.ac.actor_layers():
            info = {
                "v_threshold": float(getattr(layer.neuron, "v_threshold", None) or 0.0),
                "weight_norm_mean": float(
                    layer.linear.weight.norm(dim=1).mean().item()
                ),
            }
            actor_layer_info.append(info)
        return {
            "total_episodes": self._total_episodes,
            "actor_learner": self.actor_learner.diagnostics(),
            "network": self.ac.diagnostics(),
            "actor_layers": actor_layer_info,
        }
        
    @torch.no_grad()
    def layer_firing_rates(self) -> dict[str, float]:
        """Return per-layer mean firing rates for the actor (in spikes/step)."""
        rates = {}
        for i, layer in enumerate(self.ac.actor_layers()):
            rate_ema = getattr(layer.neuron, "rate_ema", None)
            if rate_ema is not None:
                rates[f"layer_{i}"] = float(rate_ema.mean().item())
            else:
                rates[f"layer_{i}"] = 0.0
        return rates