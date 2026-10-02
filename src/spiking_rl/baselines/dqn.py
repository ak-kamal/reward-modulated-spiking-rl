"""
Deep Q-Network (DQN) baseline for the gridworld.

Standard DQN with:
- A small MLP (64-64 hidden units) mapping observations to Q-values.
- Experience replay buffer.
- Target network with periodic hard copy.
- Epsilon-greedy exploration.

This is a direct port of the original DQN algorithm (Mnih et al., 2015) to
the gridworld. Architectural size is kept small to be comparable to the
SNN actor-critic.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    hidden: tuple[int, ...] = (64, 64)
    lr: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 10_000
    batch_size: int = 32
    target_update_steps: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 200
    train_start: int = 200
    log_every: int = 50

    def __post_init__(self) -> None:
        if self.train_start < self.batch_size:
            raise ValueError(
                f"train_start ({self.train_start}) must be >= batch_size "
                f"({self.batch_size})"
            )

class DQNAgent:
    """Deep Q-Network agent."""

    def __init__(self, env, config: DQNConfig | None = None) -> None:
        self.env = env
        self.config = config or DQNConfig()

        obs_dim = int(np.prod(env.observation_space.shape))
        self.num_actions = int(env.action_space.n)
        self.obs_dim = obs_dim

        self.q_net = QNetwork(obs_dim, self.num_actions, self.config.hidden)
        self.target_net = QNetwork(obs_dim, self.num_actions, self.config.hidden)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.config.lr
        )
        self.buffer: deque = deque(maxlen=self.config.buffer_size)

        self._step_count = 0
        self._total_episodes = 0

    def _current_epsilon(self) -> float:
        c = self.config
        progress = min(1.0, self._total_episodes / c.epsilon_decay_episodes)
        return c.epsilon_start + progress * (c.epsilon_end - c.epsilon_start)

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32)

    # ------------------------------------------------------------------

    def train_episode(self, seed: int | None = None) -> dict:
        obs, _ = self.env.reset(seed=seed)
        episode_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        epsilon = self._current_epsilon()
        last_loss = 0.0

        for _ in range(10_000):
            if np.random.random() < epsilon:
                action = int(np.random.randint(self.num_actions))
            else:
                with torch.no_grad():
                    q_values = self.q_net(self._to_tensor(obs).unsqueeze(0))
                    action = int(q_values.argmax(dim=-1).item())

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.buffer.append((obs, action, reward, next_obs, terminated))
            self._step_count += 1

            # Training step.
            if (
                len(self.buffer) >= self.config.train_start
                and self._step_count % 1 == 0
            ):
                batch = random.sample(self.buffer, self.config.batch_size)
                obs_b, act_b, rew_b, next_obs_b, done_b = zip(*batch)

                obs_t = torch.as_tensor(np.array(obs_b), dtype=torch.float32)
                act_t = torch.as_tensor(act_b, dtype=torch.long).unsqueeze(-1)
                rew_t = torch.as_tensor(rew_b, dtype=torch.float32).unsqueeze(-1)
                next_t = torch.as_tensor(np.array(next_obs_b), dtype=torch.float32)
                done_t = torch.as_tensor(done_b, dtype=torch.float32).unsqueeze(-1)

                q_vals = self.q_net(obs_t).gather(1, act_t)
                with torch.no_grad():
                    next_q = self.target_net(next_t).max(dim=-1, keepdim=True)[0]
                    target = rew_t + self.config.gamma * next_q * (1 - done_t)

                loss = F.mse_loss(q_vals, target)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
                self.optimizer.step()
                last_loss = float(loss.item())

            # Target network update.
            if self._step_count % self.config.target_update_steps == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            obs = next_obs
            episode_reward += reward
            steps += 1
            if terminated or truncated:
                break

        self._total_episodes += 1

        return {
            "episode": self._total_episodes,
            "total_reward": float(episode_reward),
            "steps": steps,
            "success": bool(terminated and not truncated),
            "truncated": bool(truncated),
            "loss": last_loss,
            "epsilon": epsilon,
        }

    @torch.no_grad()
    def evaluate_episode(self, seed: int | None = None, greedy: bool = True) -> dict:
        obs, _ = self.env.reset(seed=seed)
        episode_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        for _ in range(10_000):
            if greedy:
                q_values = self.q_net(self._to_tensor(obs).unsqueeze(0))
                action = int(q_values.argmax(dim=-1).item())
            else:
                q_values = self.q_net(self._to_tensor(obs).unsqueeze(0))
                probs = torch.softmax(q_values, dim=-1).squeeze(0).numpy()
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

    # ------------------------------------------------------------------

    def train(self, n_episodes: int, seed_start: int = 0, verbose: bool = True) -> list:
        history = []
        for i in range(n_episodes):
            metrics = self.train_episode(seed=seed_start + i)
            history.append(metrics)
            if verbose and (i + 1) % self.config.log_every == 0:
                recent = history[-self.config.log_every:]
                mean_reward = float(np.mean([m["total_reward"] for m in recent]))
                mean_steps = float(np.mean([m["steps"] for m in recent]))
                mean_loss = float(np.mean([m["loss"] for m in recent]))
                print(
                    f"Episode {i + 1:5d}/{n_episodes} | "
                    f"reward={mean_reward:+.3f} | steps={mean_steps:6.1f} | "
                    f"loss={mean_loss:.4f} | eps={self._current_epsilon():.3f}"
                )
        return history

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