"""
Gridworld environment for reward-modulated spiking RL experiments.

This environment follows the Gymnasium API standard (reset/step/close) and
implements three configurable noise mechanisms that are central to the
research question:

    1. Transition noise: the executed action may differ from the intended
       action (slippery gridworld).
    2. Observation noise: Gaussian noise added to the agent's state observation.
    3. Reward noise: Gaussian noise added to the reward signal.

The transition noise mechanism follows the classic Berkeley CS188 gridworld
convention (Sutton & Barto "slippery gridworld"): when the agent intends to
move north or south, it slips west or east with probability `transition_noise`;
when it intends to move east or west, it slips north or south.

Observation and reward noise are implemented as Gaussian perturbations.

References
----------
- Gymnasium environment creation tutorial:
  https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
- Berkeley CS188 gridworld (transition noise):
  https://inst.eecs.berkeley.edu/~cs188/sp24/projects/proj3/
- Spiking actor-critic gridworld (task validation):
  Potjans et al., "A Spiking Neural Network Model of an Actor-Critic
  Learning Agent", Neural Computation (2011).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    """A configurable gridworld with transition, observation, and reward noise.

    Parameters
    ----------
    size : int
        The grid is size x size. Default is 5 (a 5x5 grid).
    start : tuple[int, int] or None
        Starting position (row, col). If None, start is (0, 0).
    goal : tuple[int, int] or None
        Goal position (row, col). If None, goal is (size-1, size-1).
    obstacles : list[tuple[int, int]] or None
        Positions that cannot be entered. If None, no obstacles.
    penalty_states : list[tuple[int, int]] or None
        Positions that terminate the episode with a negative reward.
    goal_reward : float
        Reward received upon reaching the goal. Default is 1.0.
    penalty_reward : float
        Reward received upon entering a penalty state. Default is -1.0.
    step_reward : float
        Reward received on every step (living reward). Default is 0.0.
    transition_noise : float
        Probability in [0, 1] that the agent slips perpendicular to its
        intended direction. 0.0 = deterministic, 1.0 = fully random
        perpendicular movement. Default is 0.0.
    observation_noise : float
        Standard deviation of Gaussian noise added to the observation.
        Default is 0.0 (no observation noise).
    reward_noise : float
        Standard deviation of Gaussian noise added to the reward signal.
        Default is 0.0 (no reward noise).
    max_steps : int
        Maximum number of steps per episode before truncation.
        Default is 100.
    render_mode : str or None
        One of "human", "rgb_array", "ansi", or None.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
    }

    # Action indices and their (row_delta, col_delta) mappings.
    # 0 = up, 1 = right, 2 = down, 3 = left
    _ACTION_TO_DIRECTION = {
        0: np.array([-1, 0]),  # up
        1: np.array([0, 1]),   # right
        2: np.array([1, 0]),   # down
        3: np.array([0, -1]),  # left
    }

    def __init__(
        self,
        size: int = 5,
        start: tuple[int, int] | None = None,
        goal: tuple[int, int] | None = None,
        obstacles: list[tuple[int, int]] | None = None,
        penalty_states: list[tuple[int, int]] | None = None,
        goal_reward: float = 1.0,
        penalty_reward: float = -1.0,
        step_reward: float = 0.0,
        transition_noise: float = 0.0,
        observation_noise: float = 0.0,
        reward_noise: float = 0.0,
        max_steps: int = 100,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        # Validate noise parameters.
        if not 0.0 <= transition_noise <= 1.0:
            raise ValueError(
                f"transition_noise must be in [0, 1], got {transition_noise}"
            )
        if observation_noise < 0.0:
            raise ValueError(
                f"observation_noise must be >= 0, got {observation_noise}"
            )
        if reward_noise < 0.0:
            raise ValueError(
                f"reward_noise must be >= 0, got {reward_noise}"
            )

        self.size = size
        self.start = np.array(start if start is not None else (0, 0), dtype=np.int64)
        self.goal = np.array(
            goal if goal is not None else (size - 1, size - 1), dtype=np.int64
        )
        self.obstacles = set(map(tuple, obstacles or []))
        self.penalty_states = set(map(tuple, penalty_states or []))

        self.goal_reward = goal_reward
        self.penalty_reward = penalty_reward
        self.step_reward = step_reward

        self.transition_noise = transition_noise
        self.observation_noise = observation_noise
        self.reward_noise = reward_noise
        self.max_steps = max_steps

        self.render_mode = render_mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # Observation: flattened vector [agent_row, agent_col, goal_row, goal_col]
        # normalized to [0, 1]. This is a compact representation that works
        # cleanly with a spiking neural network input layer.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Actions: 0 = up, 1 = right, 2 = down, 3 = left
        self.action_space = spaces.Discrete(4)

        # Internal state, initialized in reset().
        self._agent_pos: np.ndarray | None = None
        self._step_count: int = 0

        # Rendering resources (lazily initialized).
        self._window = None
        self._clock = None
        self._executed_action: int | None = None
        
    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the starting state.

        Returns
        -------
        observation : np.ndarray
            The initial observation vector of shape (4,).
        info : dict
            Auxiliary information (contains the true agent position).
        """
        super().reset(seed=seed)

        self._agent_pos = self.start.copy()
        self._step_count = 0
        self._executed_action = None
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one action in the environment.

        Parameters
        ----------
        action : int
            One of {0, 1, 2, 3} corresponding to {up, right, down, left}.

        Returns
        -------
        observation : np.ndarray
            The observation after the transition.
        reward : float
            The (possibly noisy) reward signal.
        terminated : bool
            True if the episode ended by reaching the goal or a penalty state.
        truncated : bool
            True if the episode ended because max_steps was exceeded.
        info : dict
            Auxiliary information.
        """
        if self._agent_pos is None:
            raise RuntimeError("step() called before reset().")

        self._step_count += 1

        # --- Transition noise -------------------------------------------
        # The agent may slip perpendicular to its intended direction.
        executed_action = self._apply_transition_noise(action)

        # Compute the proposed new position.
        direction = self._ACTION_TO_DIRECTION[executed_action]
        proposed = self._agent_pos + direction

        # Check boundaries and obstacles.
        if self._is_valid_position(proposed):
            self._agent_pos = proposed
        # If invalid, the agent stays in place (bumps into wall/obstacle).

        # --- Compute base reward -----------------------------------------
        terminated = False
        base_reward = self.step_reward

        if np.array_equal(self._agent_pos, self.goal):
            base_reward = self.goal_reward
            terminated = True
        elif tuple(self._agent_pos) in self.penalty_states:
            base_reward = self.penalty_reward
            terminated = True

        # --- Reward noise ------------------------------------------------
        reward = base_reward
        if self.reward_noise > 0.0:
            reward += float(
                self.np_random.normal(0.0, self.reward_noise)
            )

        # --- Truncation --------------------------------------------------
        truncated = self._step_count >= self.max_steps

        self._executed_action = executed_action
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation and info
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector with optional Gaussian noise."""
        obs = np.array(
            [
                self._agent_pos[0] / (self.size - 1),
                self._agent_pos[1] / (self.size - 1),
                self.goal[0] / (self.size - 1),
                self.goal[1] / (self.size - 1),
            ],
            dtype=np.float32,
        )

        if self.observation_noise > 0.0:
            obs = obs + self.np_random.normal(
                0.0, self.observation_noise, size=obs.shape
            ).astype(np.float32)
            # Clip to the observation space bounds.
            obs = np.clip(obs, 0.0, 1.0).astype(np.float32)

        return obs

    def _get_info(self) -> dict[str, Any]:
        return {
            "agent_pos": self._agent_pos.copy(),
            "step_count": self._step_count,
            "distance_to_goal": float(
                np.sum(np.abs(self._agent_pos - self.goal))
            ),
            "executed_action": self._executed_action,
        }

    # ------------------------------------------------------------------
    # Transition noise
    # ------------------------------------------------------------------

    def _apply_transition_noise(self, action: int) -> int:
        """Return the executed action after applying transition noise.

        With probability `transition_noise`, the agent slips perpendicular
        to its intended direction. For up/down actions, it slips to left
        or right; for left/right actions, it slips to up or down.

        When transition_noise == 1.0, the agent always slips (the intended
        direction is never executed).
        """
        if self.transition_noise <= 0.0:
            return action

        if self.np_random.random() >= self.transition_noise:
            return action

        # Slip perpendicular to the intended direction.
        perpendicular = {
            0: (3, 1),  # up -> left or right
            2: (3, 1),  # down -> left or right
            1: (0, 2),  # right -> up or down
            3: (0, 2),  # left -> up or down
        }[action]

        return int(self.np_random.choice(perpendicular))

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check whether a position is within bounds and not an obstacle."""
        if not (0 <= pos[0] < self.size and 0 <= pos[1] < self.size):
            return False
        if tuple(pos) in self.obstacles:
            return False
        return True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray | str | None:
        """Render the environment.

        Supports "human" (pygame window), "rgb_array" (numpy array), and
        "ansi" (text string).
        """
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.render_mode == "human":
            self._render_human()
            return None
        return None

    def _render_ansi(self) -> str:
        """Text-based rendering using ASCII characters."""
        lines = []
        for row in range(self.size):
            line = ""
            for col in range(self.size):
                pos = (row, col)
                if tuple(pos) == tuple(self._agent_pos):
                    line += " A "
                elif tuple(pos) == tuple(self.goal):
                    line += " G "
                elif pos in self.obstacles:
                    line += " # "
                elif tuple(pos) in self.penalty_states:
                    line += " X "
                else:
                    line += " . "
            lines.append(line)
        return "\n".join(lines)

    def _render_rgb_array(self) -> np.ndarray:
        """Render as an RGB numpy array without requiring pygame."""
        cell_size = 64
        grid = np.ones((self.size * cell_size, self.size * cell_size, 3), dtype=np.uint8) * 255

        for row in range(self.size):
            for col in range(self.size):
                y0, y1 = row * cell_size, (row + 1) * cell_size
                x0, x1 = col * cell_size, (col + 1) * cell_size

                if (row, col) in self.obstacles:
                    grid[y0:y1, x0:x1] = [40, 40, 40]
                elif (row, col) in self.penalty_states:
                    grid[y0:y1, x0:x1] = [255, 200, 200]
                elif (row, col) == tuple(self.goal):
                    grid[y0:y1, x0:x1] = [200, 255, 200]

        # Draw the agent as a blue square in the center of its cell.
        ar, ac = self._agent_pos
        ay, ax = ar * cell_size, ac * cell_size
        margin = cell_size // 4
        grid[ay + margin : ay + cell_size - margin, ax + margin : ax + cell_size - margin] = [0, 0, 255]

        # Draw grid lines.
        for i in range(self.size + 1):
            grid[i * cell_size : i * cell_size + 1, :] = [200, 200, 200]
            grid[:, i * cell_size : i * cell_size + 1] = [200, 200, 200]

        return grid

    def _render_human(self) -> None:
        """Render using pygame (requires pygame to be installed)."""
        try:
            import pygame
        except ImportError as e:
            raise ImportError(
                "pygame is required for human rendering. "
                "Install it with: uv add --dev pygame"
            ) from e

        cell_size = 64
        window_size = self.size * cell_size

        if self._window is None:
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption("GridWorld")

        if self._clock is None:
            self._clock = pygame.time.Clock()

        canvas = pygame.Surface((window_size, window_size))
        canvas.fill((255, 255, 255))

        for row in range(self.size):
            for col in range(self.size):
                rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                if (row, col) in self.obstacles:
                    pygame.draw.rect(canvas, (40, 40, 40), rect)
                elif (row, col) in self.penalty_states:
                    pygame.draw.rect(canvas, (255, 200, 200), rect)
                elif (row, col) == tuple(self.goal):
                    pygame.draw.rect(canvas, (200, 255, 200), rect)

        # Agent.
        ar, ac = self._agent_pos
        center = (int((ac + 0.5) * cell_size), int((ar + 0.5) * cell_size))
        pygame.draw.circle(canvas, (0, 0, 255), center, cell_size // 3)

        # Grid lines.
        for i in range(self.size + 1):
            pygame.draw.line(canvas, (200, 200, 200), (0, i * cell_size), (window_size, i * cell_size), width=1)
            pygame.draw.line(canvas, (200, 200, 200), (i * cell_size, 0), (i * cell_size, window_size), width=1)

        self._window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self._clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        """Close rendering resources."""
        if self._window is not None:
            try:
                import pygame

                pygame.display.quit()
                pygame.quit()
            except ImportError:
                pass
            self._window = None
            self._clock = None