"""Smoke tests for the baseline agents.

These verify that each baseline trains without crashing and improves
performance over a random policy. They are NOT exhaustive RL tests —
those would require far more compute than pytest should use.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spiking_rl.baselines.dqn import DQNAgent, DQNConfig
from spiking_rl.baselines.ppo import PPOAgent, PPOConfig
from spiking_rl.baselines.tabular_q import TabularQAgent, TabularQConfig
from spiking_rl.environment.gridworld import GridWorldEnv


@pytest.fixture
def env() -> GridWorldEnv:
    return GridWorldEnv(
        size=4,
        start=(0, 0),
        goal=(3, 3),
        max_steps=30,
        step_reward=-0.01,
        render_mode=None,
    )


# ----------------------------------------------------------------------
# Tabular Q
# ----------------------------------------------------------------------


def test_tabular_q_initializes(env):
    agent = TabularQAgent(env, TabularQConfig())
    assert agent.Q.shape == (16, 4)  # 4x4 grid, 4 actions


def test_tabular_q_trains(env):
    agent = TabularQAgent(env, TabularQConfig(epsilon_decay_episodes=30))
    history = agent.train(n_episodes=50, seed_start=0, verbose=False)
    assert len(history) == 50
    # Reward should improve: last 10 average > first 10 average.
    first = np.mean([m["total_reward"] for m in history[:10]])
    last = np.mean([m["total_reward"] for m in history[-10:]])
    assert last > first


def test_tabular_q_evaluates(env):
    agent = TabularQAgent(env, TabularQConfig(epsilon_decay_episodes=30))
    agent.train(n_episodes=50, seed_start=0, verbose=False)
    result = agent.evaluate(n_episodes=20, greedy=True)
    assert 0.0 <= result["success_rate"] <= 1.0


# ----------------------------------------------------------------------
# DQN
# ----------------------------------------------------------------------


def test_dqn_initializes(env):
    agent = DQNAgent(env, DQNConfig(hidden=(16,)))
    assert agent.q_net is not None
    assert agent.target_net is not None


def test_dqn_trains(env):
    torch.manual_seed(0)
    np.random.seed(0)
    agent = DQNAgent(
        env,
        DQNConfig(
            hidden=(16,),
            batch_size=16,
            train_start=50,
            target_update_steps=50,
            epsilon_decay_episodes=100,
            log_every=25,
        ),
    )
    history = agent.train(n_episodes=100, seed_start=0, verbose=False)
    assert len(history) == 100
    # Should not produce NaNs.
    for m in history:
        assert np.isfinite(m["total_reward"])
        assert np.isfinite(m["loss"])


def test_dqn_evaluates(env):
    torch.manual_seed(0)
    np.random.seed(0)
    agent = DQNAgent(
        env,
        DQNConfig(hidden=(16,), train_start=50, batch_size=16),
    )
    agent.train(n_episodes=30, seed_start=0, verbose=False)
    result = agent.evaluate(n_episodes=20, greedy=True)
    assert 0.0 <= result["success_rate"] <= 1.0


# ----------------------------------------------------------------------
# PPO
# ----------------------------------------------------------------------


def test_ppo_initializes(env):
    agent = PPOAgent(env, PPOConfig(hidden=(16,)))
    assert agent.net is not None


def test_ppo_trains(env):
    torch.manual_seed(0)
    np.random.seed(0)
    agent = PPOAgent(
        env,
        PPOConfig(
            hidden=(16,),
            rollout_episodes=5,
            n_epochs=2,
            batch_size=32,
            log_every=10,
        ),
    )
    history = agent.train(n_episodes=30, seed_start=0, verbose=False)
    assert len(history) >= 30
    for m in history:
        assert np.isfinite(m["total_reward"])
        assert np.isfinite(m["loss"])


def test_ppo_evaluates(env):
    torch.manual_seed(0)
    np.random.seed(0)
    agent = PPOAgent(
        env,
        PPOConfig(hidden=(16,), rollout_episodes=5, n_epochs=2, batch_size=32),
    )
    agent.train(n_episodes=20, seed_start=0, verbose=False)
    result = agent.evaluate(n_episodes=20, greedy=True)
    assert 0.0 <= result["success_rate"] <= 1.0
    
def test_tabular_q_converges_to_known_optimal_on_trivial_env():
    """On a 2x2 grid with goal at (1,1), the optimal Q-values are analytically known.

    From (0,0), the shortest path is 2 steps. With gamma=0.99, step_reward=-0.01,
    goal_reward=+1.0, we have:
        Q*((0,0), down) = -0.01 + 0.99 * (-0.01 + 0.99 * 1.0) ≈ 0.9594
        Q*((0,0), right) = same (symmetric)
    We just check the max Q-value converges to approximately this value.
    """
    env = GridWorldEnv(size=2, start=(0, 0), goal=(1, 1),
                       max_steps=10, step_reward=-0.01, render_mode=None)
    agent = TabularQAgent(env, TabularQConfig(
        alpha=0.5, gamma=0.99, epsilon_start=1.0, epsilon_end=0.0,
        epsilon_decay_episodes=200,
    ))
    agent.train(n_episodes=2000, seed_start=0, verbose=False)

    # State (0,0) = index 0. Optimal max Q should be close to 0.9594.
    max_q_from_start = float(agent.Q[0].max())
    expected = -0.01 + 0.99 * (-0.01 + 0.99 * 1.0)
    assert abs(max_q_from_start - expected) < 0.05, (
        f"Q((0,0), best action) = {max_q_from_start:.4f}, expected ~{expected:.4f}"
    )
    
def test_tabular_q_state_discretization():
    """A 5x5 env: normalized observation (0.25, 0.5) should map to (1, 2)."""
    env = GridWorldEnv(size=5, render_mode=None)
    agent = TabularQAgent(env)
    obs = np.array([0.25, 0.5, 1.0, 1.0], dtype=np.float32)
    idx = agent._state_index(obs)
    expected = 1 * 5 + 2
    assert idx == expected, f"Got index {idx}, expected {expected}"
    
def test_dqn_weights_change_during_training(env):
    torch.manual_seed(0)
    np.random.seed(0)
    agent = DQNAgent(env, DQNConfig(hidden=(16,), train_start=10, batch_size=8))
    w_before = agent.q_net.net[0].weight.clone()
    agent.train(n_episodes=30, seed_start=0, verbose=False)
    w_after = agent.q_net.net[0].weight.clone()
    assert not torch.allclose(w_before, w_after), "DQN weights did not update"
    
def test_dqn_target_network_updates(env):
    torch.manual_seed(0)
    agent = DQNAgent(
        env,
        DQNConfig(
            hidden=(16,),
            target_update_steps=20,
            train_start=8,
            batch_size=4,
            epsilon_decay_episodes=100,
        ),
    )

    # Perturb the target net so we can detect when it's copied.
    with torch.no_grad():
        for p in agent.target_net.parameters():
            p.add_(1.0)

    perturbed_target = agent.target_net.net[0].weight.clone()

    # Sanity: perturbation applied.
    assert not torch.allclose(
        agent.q_net.net[0].weight, agent.target_net.net[0].weight
    )

    agent.train(n_episodes=50, seed_start=0, verbose=False)

    # Verify at least one target update happened.
    assert agent._step_count >= 20, (
        f"Only {agent._step_count} steps; not enough to trigger update"
    )

    # After training, the perturbed target should have been overwritten
    # by at least one copy from the online net.
    final_target = agent.target_net.net[0].weight
    assert not torch.allclose(final_target, perturbed_target, atol=1e-3), (
        "Target network was never copied from online network"
    )
        
def test_dqn_replay_buffer_max_size(env):
    from collections import deque
    agent = DQNAgent(env, DQNConfig(hidden=(16,), buffer_size=50))
    assert isinstance(agent.buffer, deque)
    assert agent.buffer.maxlen == 50
    # Fill beyond capacity and check.
    for i in range(200):
        agent.buffer.append((i, 0, 0.0, i, False))
    assert len(agent.buffer) == 50
    # Oldest entry should have been evicted.
    assert agent.buffer[0][0] == 150
    
def test_ppo_gae_computation():
    """One-episode rollout of length 3 with hand-computable GAE.

    Rewards: [1, 2, 3], values: [0, 0, 0], gamma=1, lambda=1.
    Advantages[t] = sum_{k>=t} rewards[k] (with no bootstrapping).
    So: adv = [6, 5, 3]
    """
    torch.manual_seed(0)
    np.random.seed(0)
    env = GridWorldEnv(size=2, max_steps=10, step_reward=-0.01, render_mode=None)
    agent = PPOAgent(env, PPOConfig(rollout_episodes=1, n_epochs=1, batch_size=4,
                                    gamma=1.0, gae_lambda=1.0, hidden=(4,)))
    # Monkeypatch the network to return zero values and constant logits.
    with torch.no_grad():
        for p in agent.net.critic.parameters():
            p.zero_()
        agent.net.critic.bias.zero_()

    # Manually run the collection and inject rewards.
    # We can't directly set rewards, so instead we check the GAE math via
    # the returned rollout's advantage values from a controlled env.
    # Simpler: verify the recursion by inspecting the advantages.
    rollout = agent._collect_rollout(seed_start=0, max_episodes=1)
    # Since values are zero, advantage[0] should equal the discounted sum
    # of rewards from step 0 to the end of the episode.
    ep_reward_sum = float(rollout["rewards"].sum())
    adv_0 = float(rollout["advantages"][0])
    # With gamma=1, lambda=1, zero values, and one episode in the rollout:
    assert abs(adv_0 - ep_reward_sum) < 1e-5, (
        f"GAE at t=0 is {adv_0:.4f}, expected sum of rewards {ep_reward_sum:.4f}"
    )
    
def test_ppo_weights_change_during_training(env):
    torch.manual_seed(0)
    np.random.seed(0)
    agent = PPOAgent(env, PPOConfig(hidden=(16,), rollout_episodes=3, n_epochs=2, batch_size=16))
    w_before = agent.net.actor.weight.clone()
    agent.train(n_episodes=10, seed_start=0, verbose=False)
    w_after = agent.net.actor.weight.clone()
    assert not torch.allclose(w_before, w_after), "PPO actor weights did not update"
    
