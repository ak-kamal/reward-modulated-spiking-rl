"""
Tabular Q-learning baseline for the gridworld.

The observation is [agent_row, agent_col, goal_row, goal_col] normalized to
[0, 1]. Since the goal is fixed and both goal coordinates are always 1.0,
the meaningful state is the agent's (row, col) on the grid. We discretize
by rounding the normalized positions back to integer indices.

This is the simplest possible baseline: a Q-table of shape
(size * size, num_actions) updated with standard Q-learning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TabularQConfig:
    """Hyperparameters for tabular Q-learning."""

    alpha: float = 0.1            # learning rate
    gamma: float = 0.99           # discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 200
    log_every: int = 50


class TabularQAgent:
    """Tabular Q-learning agent for the gridworld."""

    def __init__(self, env, config: TabularQConfig | None = None) -> None:
        self.env = env
        self.config = config or TabularQConfig()

        self.size = int(env.size)
        self.n_states = self.size * self.size
        self.n_actions = int(env.action_space.n)

        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self._total_episodes = 0

    # ------------------------------------------------------------------

    def _state_index(self, obs: np.ndarray) -> int:
        """Convert a normalized observation to a discrete state index."""
        # Round to nearest integer grid position, clip to bounds.
        row = int(round(float(obs[0]) * (self.size - 1)))
        col = int(round(float(obs[1]) * (self.size - 1)))
        row = max(0, min(self.size - 1, row))
        col = max(0, min(self.size - 1, col))
        return row * self.size + col

    def _current_epsilon(self) -> float:
        c = self.config
        progress = min(1.0, self._total_episodes / c.epsilon_decay_episodes)
        return c.epsilon_start + progress * (c.epsilon_end - c.epsilon_start)

    # ------------------------------------------------------------------

    def train_episode(self, seed: int | None = None) -> dict:
        """Run one training episode."""
        obs, _ = self.env.reset(seed=seed)
        state = self._state_index(obs)

        episode_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        epsilon = self._current_epsilon()

        for _ in range(10_000):
            # Epsilon-greedy action.
            if np.random.random() < epsilon:
                action = int(np.random.randint(self.n_actions))
            else:
                action = int(np.argmax(self.Q[state]))

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_state = self._state_index(next_obs)

            # Q-learning update.
            td_target = reward
            if not terminated:
                td_target += self.config.gamma * float(np.max(self.Q[next_state]))
            td_error = td_target - self.Q[state, action]
            self.Q[state, action] += self.config.alpha * td_error

            state = next_state
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
        }

    def evaluate_episode(
        self, seed: int | None = None, greedy: bool = True
    ) -> dict:
        """Run one evaluation episode with no exploration."""
        obs, _ = self.env.reset(seed=seed)
        state = self._state_index(obs)

        episode_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        for _ in range(10_000):
            if greedy:
                action = int(np.argmax(self.Q[state]))
            else:
                probs = np.exp(self.Q[state] - self.Q[state].max())
                probs = probs / probs.sum()
                action = int(np.random.choice(self.n_actions, p=probs))

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            state = self._state_index(next_obs)
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
                print(
                    f"Episode {i + 1:5d}/{n_episodes} | "
                    f"reward={mean_reward:+.3f} | steps={mean_steps:6.1f} | "
                    f"eps={self._current_epsilon():.3f}"
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