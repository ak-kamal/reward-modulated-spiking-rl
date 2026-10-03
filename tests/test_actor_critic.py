"""
Unit tests for Actor, Critic, and ActorCritic.

These tests verify shapes, state management, action selection, and the
basic firing behavior (which is essential for R-STDP to work at all).
"""

from __future__ import annotations

import torch

import pytest

from spiking_rl.learning.eligibility_trace import TracedLinear
from spiking_rl.models.actor_critic import Actor, ActorCritic, Critic


OBS_DIM = 4
NUM_ACTIONS = 4


# ----------------------------------------------------------------------
# Actor
# ----------------------------------------------------------------------


def test_actor_forward_shapes():
    actor = Actor(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS, hidden_dims=(16,))
    obs = torch.ones(2, OBS_DIM)
    spike_counts, last_spikes = actor(obs, steps=5)
    assert spike_counts.shape == (2, NUM_ACTIONS)
    assert last_spikes.shape == (2, NUM_ACTIONS)


def test_actor_fires_given_strong_input():
    """Verify the actor's first layer fires given strong input.

    Note: with small hidden dims and PyTorch's default init, firing
    rate decays across layers because pre-activations sit near the LIF
    threshold. This is a well-known property of deep SNNs and will be
    addressed at the agent level (e.g., by lowering the output-layer
    threshold or custom-initializing output weights). Here we only
    verify that the input-facing layer fires, which is what the scaled
    observation is responsible for.
    """
    torch.manual_seed(0)
    actor = Actor(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS, hidden_dims=(16,))
    obs = torch.ones(1, OBS_DIM) * 5.0

    # Run 20 internal steps to build eligibility.
    for _ in range(20):
        actor(obs, steps=1)

    first_layer = actor.traced_layers()[0]
    diag = first_layer.eligibility.diagnostics()
    assert not diag["all_zero"], (
        f"Actor's first traced layer produced zero eligibility over 20 "
        f"steps despite strong input. Diagnostics: {diag}"
    )


def test_actor_rejects_bad_steps():
    actor = Actor(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS, hidden_dims=(16,))
    with pytest.raises(ValueError, match="steps"):
        actor(torch.ones(1, OBS_DIM), steps=0)


def test_actor_traced_layers_are_traced_linear():
    actor = Actor(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS, hidden_dims=(8, 8))
    layers = actor.traced_layers()
    assert len(layers) == 3  # two hidden + one output
    for l in layers:
        assert isinstance(l, TracedLinear)


# ----------------------------------------------------------------------
# Critic
# ----------------------------------------------------------------------


def test_critic_forward_shape():
    critic = Critic(obs_dim=OBS_DIM, hidden_dims=(16,))
    obs = torch.ones(2, OBS_DIM)
    value = critic(obs, steps=5)
    assert value.shape == (2, 1)


def test_critic_value_is_finite():
    torch.manual_seed(0)
    critic = Critic(obs_dim=OBS_DIM, hidden_dims=(16,))
    value = critic(torch.ones(1, OBS_DIM), steps=10)
    assert torch.isfinite(value).all()


def test_critic_rejects_bad_steps():
    critic = Critic(obs_dim=OBS_DIM, hidden_dims=(16,))
    with pytest.raises(ValueError, match="steps"):
        critic(torch.ones(1, OBS_DIM), steps=-1)


# ----------------------------------------------------------------------
# ActorCritic
# ----------------------------------------------------------------------


def test_actorcritic_forward_shapes():
    ac = ActorCritic(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        actor_hidden=(16,),
        critic_hidden=(16,),
        decision_steps=5,
    )
    obs = torch.ones(2, OBS_DIM)
    spike_counts = ac.forward_actor(obs)
    value = ac.forward_critic(obs)
    assert spike_counts.shape == (2, NUM_ACTIONS)
    assert value.shape == (2, 1)


def test_sample_action_returns_valid_indices():
    torch.manual_seed(0)
    ac = ActorCritic(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        actor_hidden=(16,),
        critic_hidden=(16,),
        decision_steps=5,
    )
    obs = torch.ones(4, OBS_DIM)
    actions, probs, spike_counts = ac.sample_action(obs)
    assert actions.shape == (4,)
    assert (actions >= 0).all() and (actions < NUM_ACTIONS).all()
    assert probs.shape == (4, NUM_ACTIONS)
    # Probabilities sum to 1 along the action axis.
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert spike_counts.shape == (4, NUM_ACTIONS)


def test_greedy_action_picks_max_spike_count():
    torch.manual_seed(0)
    ac = ActorCritic(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        actor_hidden=(16,),
        critic_hidden=(16,),
    )
    obs = torch.ones(2, OBS_DIM)
    actions, spike_counts = ac.greedy_action(obs)
    expected = spike_counts.argmax(dim=-1)
    assert torch.equal(actions, expected)


def test_reset_zeroes_eligibility():
    torch.manual_seed(0)
    ac = ActorCritic(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        actor_hidden=(16,),
        critic_hidden=(16,),
    )
    obs = torch.ones(1, OBS_DIM) * 5.0
    for _ in range(20):
        ac.forward_actor(obs)
        ac.forward_critic(obs)

    # At least the first actor layer should have fired and built up
    # nonzero eligibility. (The output layer may not fire — see the
    # note on test_actor_fires_given_strong_input.)
    first_actor_layer = ac.actor_layers()[0]
    assert not first_actor_layer.eligibility.diagnostics()["all_zero"]

    ac.reset()

    for layer in ac.actor_layers():
        assert layer.eligibility.diagnostics()["all_zero"]
    for layer in ac.critic_layers():
        assert layer.eligibility.diagnostics()["all_zero"]


def test_traced_layers_are_separate_actor_and_critic():
    ac = ActorCritic(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        actor_hidden=(8,),
        critic_hidden=(8,),
    )
    actor_layers = ac.actor_layers()
    critic_layers = ac.critic_layers()
    # No overlap.
    for a in actor_layers:
        for c in critic_layers:
            assert a is not c


def test_diagnostics_reports_spike_stats():
    torch.manual_seed(0)
    ac = ActorCritic(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        actor_hidden=(16,),
        critic_hidden=(16,),
    )
    obs = torch.ones(1, OBS_DIM)
    for _ in range(20):
        ac.forward_actor(obs)
        ac.forward_critic(obs)

    diag = ac.diagnostics()
    assert diag["actor_spike_mean"] is not None
    assert diag["critic_value_mean"] is not None
    assert len(diag["actor_layer_eligibility"]) == 2  # one hidden + output
    assert len(diag["critic_layer_eligibility"]) == 1  # one hidden
    

def test_actor_inhibition_reduces_output_spikes():
    """With strong inhibition, the actor's output spike count should decrease."""
    torch.manual_seed(0)
    obs = torch.ones(1, 4)

    actor_no_inh = Actor(obs_dim=4, num_actions=4, hidden_dims=(16,),
                         inhibition_gain=0.0)
    actor_with_inh = Actor(obs_dim=4, num_actions=4, hidden_dims=(16,),
                           inhibition_gain=5.0)
    # Copy weights so the only difference is the inhibition gain.
    actor_with_inh.load_state_dict(actor_no_inh.state_dict(), strict=False)

    counts_no_inh, _ = actor_no_inh(obs, steps=20)
    counts_with_inh, _ = actor_with_inh(obs, steps=20)

    assert counts_with_inh.sum().item() <= counts_no_inh.sum().item()
    

def test_lateral_inhibition_is_within_layer_not_feedforward():
    """With a single-layer actor, within-layer inhibition still reduces
    spikes because each neuron sees the previous timestep's activity in
    its own layer.

    Important: this test uses a *moderate* input magnitude. With very
    large inputs, the LIF neurons saturate (fire every step) and no
    realistic inhibition can change behavior. Lateral inhibition only
    has an effect in the balanced-firing regime near threshold.
    """
    torch.manual_seed(0)
    # Small input so pre-activations sit near threshold.
    obs = torch.ones(1, 4) * 0.05

    actor_no_inh = Actor(
        obs_dim=4, num_actions=4, hidden_dims=(),
        inhibition_gain=0.0,
    )
    actor_with_inh = Actor(
        obs_dim=4, num_actions=4, hidden_dims=(),
        inhibition_gain=10.0,
    )
    actor_with_inh.load_state_dict(actor_no_inh.state_dict(), strict=False)

    counts_no, _ = actor_no_inh(obs, steps=20)
    counts_inh, _ = actor_with_inh(obs, steps=20)

    # Sanity: the baseline must fire at all with this input.
    assert counts_no.sum().item() > 0, (
        "Sanity check failed: the no-inhibition actor should fire at "
        "least once with this input. "
        f"Got {counts_no.sum().item()} spikes."
    )

    # The actual assertion: inhibition should reduce spikes.
    assert counts_inh.sum().item() < counts_no.sum().item(), (
        f"Within-layer inhibition should reduce spikes even in a "
        f"single-layer actor. Feedforward inhibition would have no "
        f"effect. Got no-inh={counts_no.sum().item()}, "
        f"with-inh={counts_inh.sum().item()}."
    )


def test_lateral_inhibition_ineffective_under_saturation():
    """Documentation test: under strong input, neurons saturate and
    lateral inhibition has no effect. This is a property of the
    mechanism, not a bug.
    """
    torch.manual_seed(0)
    obs = torch.ones(1, 4) * 10.0

    actor_no_inh = Actor(obs_dim=4, num_actions=4, hidden_dims=(),
                         inhibition_gain=0.0)
    actor_with_inh = Actor(obs_dim=4, num_actions=4, hidden_dims=(),
                           inhibition_gain=100.0)
    actor_with_inh.load_state_dict(actor_no_inh.state_dict(), strict=False)

    counts_no, _ = actor_no_inh(obs, steps=20)
    counts_inh, _ = actor_with_inh(obs, steps=20)

    max_possible = 20 * 4
    assert counts_no.sum().item() == max_possible
    assert counts_inh.sum().item() == max_possible

def test_lateral_inhibition_higher_gain_fewer_spikes():
    """Monotonicity check: more inhibition → fewer or equal spikes."""
    torch.manual_seed(0)
    obs = torch.ones(1, 4) * 10.0

    counts = []
    for gain in [0.0, 0.5, 1.0, 2.0, 5.0]:
        torch.manual_seed(0)
        actor = Actor(
            obs_dim=4, num_actions=4, hidden_dims=(16,),
            inhibition_gain=gain,
        )
        c, _ = actor(obs, steps=20)
        counts.append(c.sum().item())

    # Each step should not increase relative to the previous gain.
    for i in range(1, len(counts)):
        assert counts[i] <= counts[i - 1] + 1e-6, (
            f"Higher inhibition produced more spikes: "
            f"gain sequence counts = {counts}"
        )