"""
Unit tests for GridWorldEnv.

These tests verify the environment's contract before any learning agent
depends on it. A buggy environment would silently corrupt every downstream
experiment, so the goal here is to catch structural bugs early.

Test categories:
    1. API contract (reset/step signatures, observation shape, info keys)
    2. Movement and boundary behavior (walls, obstacles)
    3. Termination conditions (goal, penalty, truncation)
    4. Noise mechanisms (transition, observation, reward)
    5. Reproducibility (seeded runs produce identical trajectories)
"""

from __future__ import annotations

import numpy as np
import pytest

from spiking_rl.environment.gridworld import GridWorldEnv


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def env() -> GridWorldEnv:
    """A basic 5x5 deterministic gridworld with goal at bottom-right."""
    return GridWorldEnv(size=5, render_mode=None)


@pytest.fixture
def env_with_obstacles() -> GridWorldEnv:
    """A 5x5 gridworld with obstacles and penalty states."""
    return GridWorldEnv(
        size=5,
        obstacles=[(1, 1), (2, 2)],
        penalty_states=[(0, 4), (4, 0)],
        render_mode=None,
    )


# ----------------------------------------------------------------------
# 1. API contract
# ----------------------------------------------------------------------


def test_reset_returns_correct_observation_shape(env: GridWorldEnv) -> None:
    obs, info = env.reset(seed=0)
    assert obs.shape == (4,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_reset_observation_within_bounds(env: GridWorldEnv) -> None:
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_reset_places_agent_at_start(env: GridWorldEnv) -> None:
    env.reset(seed=0)
    assert np.array_equal(env._agent_pos, env.start)


def test_step_returns_five_tuple(env: GridWorldEnv) -> None:
    env.reset(seed=0)
    result = env.step(1)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "agent_pos" in info
    assert "step_count" in info
    assert "distance_to_goal" in info


def test_step_before_reset_raises() -> None:
    env = GridWorldEnv(size=5)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


# ----------------------------------------------------------------------
# 2. Movement and boundary behavior
# ----------------------------------------------------------------------


def test_move_right(env: GridWorldEnv) -> None:
    env.reset(seed=0)
    env.step(1)  # right
    assert np.array_equal(env._agent_pos, np.array([0, 1]))


def test_move_down(env: GridWorldEnv) -> None:
    env.reset(seed=0)
    env.step(2)  # down
    assert np.array_equal(env._agent_pos, np.array([1, 0]))


def test_wall_collision_keeps_agent_in_place(env: GridWorldEnv) -> None:
    env.reset(seed=0)
    env.step(0)  # up from (0,0) hits top wall
    assert np.array_equal(env._agent_pos, np.array([0, 0]))


def test_left_wall_collision(env: GridWorldEnv) -> None:
    env.reset(seed=0)
    env.step(3)  # left from (0,0) hits left wall
    assert np.array_equal(env._agent_pos, np.array([0, 0]))


def test_obstacle_blocks_movement(env_with_obstacles: GridWorldEnv) -> None:
    env_with_obstacles.reset(seed=0)
    # Move right to (0,1), then down would hit obstacle at (1,1)
    env_with_obstacles.step(1)  # right -> (0,1)
    env_with_obstacles.step(2)  # down -> blocked by (1,1)
    assert np.array_equal(env_with_obstacles._agent_pos, np.array([0, 1]))


# ----------------------------------------------------------------------
# 3. Termination conditions
# ----------------------------------------------------------------------


def test_reaching_goal_terminates_with_positive_reward() -> None:
    env = GridWorldEnv(size=3, goal=(0, 1), goal_reward=1.0)
    env.reset(seed=0)
    _, reward, terminated, truncated, _ = env.step(1)  # right into goal
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(1.0)


def test_entering_penalty_state_terminates_with_negative_reward() -> None:
    env = GridWorldEnv(size=3, penalty_states=[(0, 1)], penalty_reward=-1.0)
    env.reset(seed=0)
    _, reward, terminated, truncated, _ = env.step(1)  # right into penalty
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(-1.0)


def test_truncation_at_max_steps() -> None:
    env = GridWorldEnv(size=5, max_steps=3)
    env.reset(seed=0)
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(0)  # bump wall, no termination
        assert terminated is False
        assert truncated is False
    # Third step should truncate.
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated is False
    assert truncated is True


def test_step_count_increments(env: GridWorldEnv) -> None:
    env.reset(seed=0)
    env.step(0)
    env.step(0)
    assert env._step_count == 2


# ----------------------------------------------------------------------
# 4. Noise mechanisms
# ----------------------------------------------------------------------


def test_zero_transition_noise_is_deterministic() -> None:
    env = GridWorldEnv(size=5, transition_noise=0.0)
    for seed in range(20):
        env.reset(seed=seed)
        env.step(1)  # right
        assert np.array_equal(env._agent_pos, np.array([0, 1]))


def test_full_transition_noise_never_executes_intended_direction() -> None:
    env = GridWorldEnv(size=5, transition_noise=1.0)
    intended = 0  # up; perpendicular options are left (3) and right (1)
    executed_set = set()
    for seed in range(50):
        env.reset(seed=seed)
        env.step(intended)
        executed_set.add(env._get_info()["executed_action"])
    # With noise=1.0, the intended action should never appear.
    assert intended not in executed_set
    # Only perpendicular options should appear.
    assert executed_set.issubset({1, 3})


def test_transition_noise_frequency_matches_probability() -> None:
    """Statistical check: slips should occur ~30% of the time at p=0.3."""
    env = GridWorldEnv(size=5, transition_noise=0.3)
    n_trials = 5000
    slips = 0
    for seed in range(n_trials):
        env.reset(seed=seed)
        env.step(1)  # intended right; perpendicular is up (0) or down (2)
        if env._get_info()["executed_action"] != 1:
            slips += 1
    observed_rate = slips / n_trials
    assert 0.27 < observed_rate < 0.33, f"Observed slip rate: {observed_rate}"


def test_observation_noise_stays_within_bounds() -> None:
    env = GridWorldEnv(size=5, observation_noise=0.5)
    for seed in range(100):
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        obs, _, _, _, _ = env.step(1)
        assert env.observation_space.contains(obs)


def test_reward_noise_perturbs_goal_reward() -> None:
    env = GridWorldEnv(size=3, goal=(0, 1), goal_reward=1.0, reward_noise=0.5)
    rewards = []
    for seed in range(200):
        env.reset(seed=seed)
        _, reward, terminated, _, _ = env.step(1)
        assert terminated
        rewards.append(reward)
    rewards_arr = np.array(rewards)
    # Mean should be near 1.0, but with variance.
    assert rewards_arr.mean() == pytest.approx(1.0, abs=0.1)
    assert rewards_arr.std() > 0.2
    # Not all rewards are exactly 1.0.
    assert not np.allclose(rewards_arr, 1.0)


def test_zero_reward_noise_is_deterministic() -> None:
    env = GridWorldEnv(size=3, goal=(0, 1), goal_reward=1.0, reward_noise=0.0)
    for seed in range(20):
        env.reset(seed=seed)
        _, reward, _, _, _ = env.step(1)
        assert reward == pytest.approx(1.0)


# ----------------------------------------------------------------------
# 5. Reproducibility
# ----------------------------------------------------------------------


def test_seeded_runs_produce_identical_trajectories() -> None:
    env = GridWorldEnv(size=5, transition_noise=0.4, observation_noise=0.1)

    def run(seed: int) -> tuple[list[float], list[np.ndarray]]:
        obs, _ = env.reset(seed=seed)
        rewards = []
        observations = [obs]
        for _ in range(20):
            obs, reward, terminated, truncated, _ = env.step(1)
            rewards.append(reward)
            observations.append(obs)
            if terminated or truncated:
                break
        return rewards, observations

    rewards_a, obs_a = run(seed=42)
    rewards_b, obs_b = run(seed=42)

    assert rewards_a == rewards_b
    for oa, ob in zip(obs_a, obs_b):
        assert np.array_equal(oa, ob)


def test_different_seeds_produce_different_trajectories() -> None:
    env = GridWorldEnv(size=5, transition_noise=0.9)
    env.reset(seed=0)
    path_a = []
    for _ in range(10):
        env.step(1)
        path_a.append(env._agent_pos.copy())

    env.reset(seed=1)
    path_b = []
    for _ in range(10):
        env.step(1)
        path_b.append(env._agent_pos.copy())

    # Paths should differ under high noise with different seeds.
    assert not all(
        np.array_equal(a, b) for a, b in zip(path_a, path_b)
    )


# ----------------------------------------------------------------------
# 6. Rendering (smoke tests)
# ----------------------------------------------------------------------


def test_ansi_render_returns_string(env: GridWorldEnv) -> None:
    env.render_mode = "ansi"
    env.reset(seed=0)
    out = env.render()
    assert isinstance(out, str)
    assert "A" in out  # agent symbol
    assert "G" in out  # goal symbol


def test_rgb_array_render_shape(env: GridWorldEnv) -> None:
    env.render_mode = "rgb_array"
    env.reset(seed=0)
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3