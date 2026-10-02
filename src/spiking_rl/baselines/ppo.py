"""
Proximal Policy Optimization (PPO) baseline for the gridworld.

Standard PPO with:
- A shared-trunk actor-critic MLP.
- Generalized Advantage Estimation (GAE).
- Clipped surrogate objective.

This is a direct port of the PPO algorithm (Schulman et al., 2017) to the
gridworld. Architectural size is kept small to be comparable to the
SNN actor-critic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticMLP(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        trunk = []
        prev = obs_dim
        for h in hidden:
            trunk.append(nn.Linear(prev, h))
            trunk.append(nn.Tanh())
            prev = h
        self.trunk = nn.Sequential(*trunk)
        self.actor = nn.Linear(prev, num_actions)
        self.critic = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)


@dataclass
class PPOConfig:
    hidden: tuple[int, ...] = (64, 64)
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    rollout_episodes: int = 10
    n_epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    log_every: int = 10


class PPOAgent:
    """Proximal Policy Optimization agent."""

    def __init__(self, env, config: PPOConfig | None = None) -> None:
        self.env = env
        self.config = config or PPOConfig()

        obs_dim = int(np.prod(env.observation_space.shape))
        self.num_actions = int(env.action_space.n)
        self.obs_dim = obs_dim

        self.net = ActorCriticMLP(obs_dim, self.num_actions, self.config.hidden)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)

        self._total_episodes = 0

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32)

    # ------------------------------------------------------------------

    def _collect_rollout(self, seed_start: int, max_episodes: int | None = None) -> dict:
        """Collect a rollout of `rollout_episodes` episodes (or fewer if max_episodes is smaller)."""
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        n_eps = self.config.rollout_episodes
        if max_episodes is not None:
            n_eps = min(n_eps, max_episodes)

        for i in range(n_eps):
            obs, _ = self.env.reset(seed=seed_start + i)
            done = False
            while not done:
                obs_t = self._to_tensor(obs).unsqueeze(0)
                with torch.no_grad():
                    logits, value = self.net(obs_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    logp = dist.log_prob(action)

                next_obs, reward, terminated, truncated, _ = self.env.step(int(action.item()))
                done = bool(terminated or truncated)

                obs_buf.append(obs)
                act_buf.append(int(action.item()))
                logp_buf.append(float(logp.item()))
                rew_buf.append(float(reward))
                val_buf.append(float(value.item()))
                done_buf.append(done)

                obs = next_obs

        # Compute GAE.
        advantages = np.zeros_like(rew_buf, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(len(rew_buf))):
            if done_buf[t]:
                next_val = 0.0
                last_adv = 0.0
            else:
                next_val = val_buf[t + 1] if t + 1 < len(val_buf) else 0.0
            delta = rew_buf[t] + self.config.gamma * next_val - val_buf[t]
            last_adv = delta + self.config.gamma * self.config.gae_lambda * last_adv
            advantages[t] = last_adv

        returns = advantages + np.array(val_buf, dtype=np.float32)

        return {
            "obs": np.array(obs_buf, dtype=np.float32),
            "actions": np.array(act_buf, dtype=np.int64),
            "logp": np.array(logp_buf, dtype=np.float32),
            "advantages": advantages,
            "returns": returns,
            "rewards": np.array(rew_buf, dtype=np.float32),
            "dones": done_buf,   
        }

    def _update(self, rollout: dict) -> dict:
        obs = torch.as_tensor(rollout["obs"], dtype=torch.float32)
        actions = torch.as_tensor(rollout["actions"], dtype=torch.long)
        old_logp = torch.as_tensor(rollout["logp"], dtype=torch.float32)
        advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32)
        returns = torch.as_tensor(rollout["returns"], dtype=torch.float32)

        # Normalize advantages.
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(obs)
        losses = []
        for _ in range(self.config.n_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.config.batch_size):
                batch_idx = idx[start : start + self.config.batch_size]
                logits, values = self.net(obs[batch_idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(actions[batch_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp[batch_idx])
                surr1 = ratio * adv_norm[batch_idx]
                surr2 = torch.clamp(
                    ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps
                ) * adv_norm[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns[batch_idx])
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                losses.append(float(loss.item()))

        return {"loss": float(np.mean(losses)) if losses else 0.0}

    # ------------------------------------------------------------------

    def train(self, n_episodes: int, seed_start: int = 0, verbose: bool = True) -> list:
        """Train for n_episodes, using rollouts of `rollout_episodes` each."""
        history = []
        ep = 0
        while ep < n_episodes:
            rollout = self._collect_rollout(seed_start + ep, n_episodes - ep)
            update_info = self._update(rollout)

            # Split the rollout into per-episode metrics using real
            # done flags (returned by _collect_rollout).
            ep_rewards = []
            ep_steps = []
            ep_success = []
            cur_r = 0.0
            cur_s = 0
            for r, d in zip(rollout["rewards"], rollout["dones"]):
                cur_r += r
                cur_s += 1
                if d:
                    ep_rewards.append(cur_r)
                    ep_steps.append(cur_s)
                    # Goal reward > 0; step penalty is negative.
                    ep_success.append(bool(r > 0.0))
                    cur_r = 0.0
                    cur_s = 0

            for i in range(len(ep_rewards)):
                ep += 1
                history.append({
                    "episode": ep,
                    "total_reward": float(ep_rewards[i]),
                    "steps": int(ep_steps[i]),
                    "success": ep_success[i],
                    "loss": update_info["loss"],
                })
                if ep >= n_episodes:
                    break

            if verbose and ep % self.config.log_every < self.config.rollout_episodes:
                recent = history[-self.config.log_every :]
                print(
                    f"Episode {ep:5d}/{n_episodes} | "
                    f"reward={np.mean([m['total_reward'] for m in recent]):+.3f} | "
                    f"steps={np.mean([m['steps'] for m in recent]):6.1f}"
                )

        self._total_episodes += ep
        return history
    
    @torch.no_grad()
    def evaluate_episode(self, seed: int | None = None, greedy: bool = True) -> dict:
        obs, _ = self.env.reset(seed=seed)
        episode_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        for _ in range(10_000):
            obs_t = self._to_tensor(obs).unsqueeze(0)
            logits, _ = self.net(obs_t)
            if greedy:
                action = int(logits.argmax(dim=-1).item())
            else:
                probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                action = int(np.random.choice(self.num_actions, p=probs))

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            obs = next_obs
            episode_reward += reward
            steps += 1
            if terminated or truncated:
                break

        return {
            "total_reward": float(episode_reward),
            "steps": steps,
            "success": bool(terminated and not truncated),
            "truncated": bool(truncated),
        }

    def evaluate(
        self, n_episodes: int, seed_start: int = 10_000, greedy: bool = True
    ) -> dict:
        results = [
            self.evaluate_episode(seed=seed_start + i, greedy=greedy)
            for i in range(n_episodes)
        ]
        return {
            "mean_reward": float(np.mean([r["total_reward"] for r in results])),
            "success_rate": float(np.mean([r["success"] for r in results])),
            "mean_steps": float(np.mean([r["steps"] for r in results])),
            "n_episodes": n_episodes,
        }