# Reward-Modulated Spiking Neural Networks for Adaptive Decision Making Under Noise

A focused investigation of whether a spiking neural network agent can learn adaptive
decision-making through a reward-modulated three-factor learning rule, and what
sparsity–performance–robustness trade-offs emerge.

## Status

🚧 **In development.** The environment, neuron models, eligibility trace
mechanism, three-factor learning rule, and actor-critic architecture are
implemented and tested (109 unit tests passing). The training loop converges
reliably on the task. Baselines and noise-robustness experiments are next.

## Research Question

Can a spiking neural network agent learn adaptive decision-making in a noisy
gridworld environment using a biologically grounded three-factor learning rule,
and how does performance trade off against spike sparsity and robustness to
noise?

The specific angle we pursue is the interaction between reward-modulated
plasticity, spike sparsity, and noise robustness — a combination that is
underexplored in reward-modulated spiking RL.

## Motivation

Spiking neural networks (SNNs) process information through sparse, event-driven
spikes and promise energy-efficient implementations on neuromorphic hardware.
But training them for reinforcement learning is hard: spikes are
non-differentiable, rewards are often delayed, and credit must be assigned
across both space (which synapses) and time (which events).

The three-factor learning rule — pre-synaptic activity × post-synaptic activity
× a global neuromodulatory signal — is a biologically grounded solution to the
temporal credit assignment problem (Izhikevich, 2007; Frémaux & Gerstner, 2016).
Its factorization into eligibility traces and a reward signal is what makes
learning from delayed rewards tractable.

## Architecture

### Actor — spiking, trained by R-STDP

A stack of `TracedLinear` layers ending in a spiking output layer. The action
distribution is derived from the spike counts of the output neurons over `T`
internal timesteps. Every actor layer maintains an eligibility trace. Weight
updates follow the three-factor rule:

    Δw = lr × modulation × E

where `modulation` is the centered temporal-difference error.

### Critic — spiking hidden layers, non-spiking output, backprop

A stack of spiking hidden layers ending in `Linear → NonSpikingLIFNode` that
outputs a continuous value estimate. Trained by backpropagation on the TD-error
MSE loss.

The actor and critic share no parameters. Full separation avoids the "which
learning rule owns this synapse" problem that shared encoders create.

### Environment

A custom gridworld implemented against the Gymnasium API with three configurable
noise mechanisms:

- **Transition noise:** with probability `p`, the agent slips perpendicular to
  its intended direction.
- **Observation noise:** Gaussian perturbation of the state vector.
- **Reward noise:** Gaussian perturbation of the reward signal.

The environment is deterministic under seed control and verified via unit tests
for movement, boundaries, termination, noise statistics, and reproducibility.

## Empirical Findings

The training pipeline required four fixes, each grounded in published work and
verified empirically. We document them because they reflect real constraints of
reward-modulated spiking RL that are not always obvious from the literature.

### Finding 1: Dense reward shaping is required

A sparse-reward gridworld (0 everywhere, +1 at goal) fails to train. The critic
correctly learns that all states have value ≈ 0, the TD error collapses to zero,
and the actor receives no learning signal. This is the sparse-reward bootstrap
problem in spiking RL.

**Fix:** A small per-step penalty (`step_reward = -0.01`) provides a continuous
learning signal. The optimal policy becomes "reach the goal in the minimum
number of steps."

### Finding 2: Centered modulation prevents weight saturation

Using raw TD error as the modulation signal causes systematic weight growth,
because TD error is positive during most of the learning phase. Weights drift
until they hit the clipping bound, and the output layer becomes state-invariant.

**Fix:** Subtract a running mean from the TD error before applying it as the
modulation signal. This is the standard variance-reduction baseline from the
R-STDP literature, generalized to the TD-error case.

### Finding 3: Homeostatic mechanisms prevent network collapse

Naive centering of the modulation signal causes a different failure mode: with
no positive bias, weights drift downward until every neuron falls below
threshold. The network goes silent and both actor and critic stop learning.

**Fix:** Two homeostatic mechanisms, both grounded in the R-STDP literature
(Skorheim et al., 2014; Sanda et al., 2017; Geng & Li, 2023):

- **Adaptive threshold (homeostatic intrinsic plasticity):** Each neuron's
  firing threshold slowly rises when firing above a target rate and falls when
  below. This prevents permanent silencing.
- **Output balancing:** Each output neuron's incoming weight vector is rescaled
  to a target L2 norm after each reward-modulated update. The target matches the
  SNN-specific initialization scale, so balancing constrains drift without
  fighting the initialization.

Together these mechanisms stabilize the network in a healthy firing regime
across the entire training run.

### Finding 4: Greedy readout fails on symmetric tasks

Our gridworld has a symmetric optimal policy: on an empty grid, "right" and
"down" are equally good from most states. R-STDP correctly learns a distribution
with approximately equal probability over the tied actions. But the integer
spike counts used for readout are **exactly tied** in these states, and
`argmax` breaks ties by index. The greedy policy therefore collapses to a fixed
action and fails completely (reward −1.0, 0% success), even though the sampled
policy reaches the goal in ~10 steps with 100% success.

**Decision:** We report sampled-policy evaluation as the primary metric. This is
a legitimate practice in RL — many published policies are stochastic — and it
accurately reflects what the network has learned. The greedy-readout limitation
is documented as a property of integer spike-count output combined with a
symmetric task, not a failure of the learning rule.

A future direction is to replace the integer spike-count readout with a
continuous readout (e.g., membrane potential of a non-spiking output layer), but
this would require changing the actor's learning rule, since R-STDP requires
spikes for its eligibility traces.

## Key Design Decisions

### Three-factor rule with eligibility traces

The eligibility trace encodes recent pre×post-synaptic coincidences and decays
exponentially. This factorization is the basis of both the classical three-factor
rule (Izhikevich, 2007; Frémaux & Gerstner, 2016) and the modern e-prop
algorithm (Bellec et al., 2020).

### Actor-critic with spiking actor, value critic

Mirrors SpikingJelly's A2C example and the BSVogler actor-critic framework.
We use TD error as the modulation signal (default) with the option to switch to
a running reward baseline.

### SNN-specific weight initialization

Standard ANN initializations cause dying signals in deep SNNs because
pre-activations are binarized by the firing threshold. We adopt the
initialization from Micheli et al. (2025), which conserves membrane-potential
variance across layers via:

    Var[W] = 1 / (n_in × P(u > θ))

This was the primary fix for the dying-signal problem in early runs.

### Firing threshold validated via operational manifold

The firing threshold `v_threshold = 1.0` follows the normalized LIF convention
(Gerstner & Kistler, 2002). We validate the choice by sweeping thresholds and
mapping the operational manifold (Mazurek et al., 2026) — the region where
activity is neither silent nor saturated. The sweep (in
`results/figures/threshold_sweep_*.png`) confirms 1.0 sits in the balanced
region.

### Homeostatic mechanisms

Adaptive threshold and output balancing are included based on the Skorheim
et al. (2014) and Sanda et al. (2017) findings that reward-modulated STDP
requires homeostatic regulation to prevent both silent and saturated regimes.

### Sparsity mechanism (planned)

Lateral inhibition is the planned sparsity mechanism for the trade-off
experiments. Q1 literature (2024 *Biomimetics*; 2013 *PLoS Comp Biol*) supports
lateral inhibition as a primary driver of sparse cortical codes. The trade-off
analysis follows the spirit of Bacho & Chu (2023), which established that
maximum sparsity is not optimal for supervised SNNs — we extend this question to
reward-modulated spiking RL.

## Results Summary

On the training task (5×5 gridworld, goal at (4,4), dense reward):

| Policy | Mean reward | Success rate | Mean steps |
|--------|-------------|--------------|------------|
| Random | −0.16 | 56% | 73 |
| BFS-optimal | +0.93 | 100% | 8 |
| **Trained (sampled)** | **+0.91** | **100%** | **10** |

The trained spiking agent essentially matches the BFS-optimal policy in success
rate while using 10 steps instead of 8.

## Repository Structure

src/spiking_rl/
├── environment/ # Gridworld + noise mechanisms
├── models/ # LIF neurons, SNN weight init, actor-critic
├── learning/ # Eligibility trace, three-factor learner, agent
├── baselines/ # Tabular Q, DQN, PPO (planned)
└── utils/ # Logging, plotting

notebooks/ # Exploration and analysis
tests/ # Unit tests (109 passing)
scripts/ # Diagnostics (threshold sweep)
results/ # Figures and metrics


## Experiments (Planned)

1. **Baselines.** Implement tabular Q, DQN, and PPO on the same environment.
2. **Sparsity–performance trade-off.** Sweep the lateral-inhibition gain and
   measure spikes per episode against success rate (Pareto curve).
3. **Noise robustness.** Evaluate all agents under transition, observation, and
   reward noise at multiple levels.
4. **Eligibility-trace ablation.** Compare the full three-factor rule against
   immediate reward-modulated STDP without traces.

## What This Project Does Not Claim

- The agent is not a basal-ganglia model. It is biologically *inspired* by
  dopaminergic reward modulation and three-factor plasticity, but does not model
  biological detail.
- Energy-efficiency claims are limited to spike counts and synaptic operation
  counts, not measured hardware energy.
- The gridworld is a controlled toy environment, appropriate for the focused
  research question but not a benchmark for real-world tasks.

## References

- Bacho, F. & Chu, D. (2023). Exploring trade-offs in spiking neural networks.
  *Neural Computation*, 35(10), 1627–1656.
- Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R.,
  & Maass, W. (2020). A solution to the learning dilemma for recurrent networks
  of spiking neurons. *Nature Communications*, 11, 3625.
- Frémaux, N. & Gerstner, W. (2016). Neuromodulated STDP and theory of
  three-factor learning rules. *Frontiers in Neural Circuits*, 9, 85.
- Geng, H. & Li, P. (2023). HoSNN: Adversarially-Robust Homeostatic Spiking
  Neural Networks with Adaptive Firing Thresholds. arXiv:2308.10373.
- Gerstner, W. & Kistler, W. M. (2002). *Spiking Neuron Models: Single Neurons,
  Populations, Plasticity*. Cambridge University Press.
- Hertag, L. & Sprekeler, H. (2020). Learning prediction error neurons through
  homeostatic plasticity. *Nature Communications*, 11, 3358.
- Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of
  STDP and dopamine signaling. *Cerebral Cortex*, 17(10), 2443–2452.
- Kozdon, K. & Bentley, P. (2019). Normalisation of weights and firing rates in
  spiking neural networks with spike-timing-dependent plasticity.
  arXiv:1910.00122.
- Mazurek, Caputa, Maj, Wielgosz (2026). Operational manifolds in spiking neural
  networks. *Frontiers in Neuroscience*.
- Micheli, Booij, van Gemert, Tomen (2025). Deep activity propagation via weight
  initialization in spiking neural networks. *IEEE*.
- Potjans, W., Diesmann, M., & Morrison, A. (2011). A spiking neural network
  model of an actor-critic learning agent. *Neural Computation*, 23(2),
  269–328.
- Sanda, P., Skorheim, S., & Bazhenov, M. (2017). Multi-layer network utilizing
  rewarded spike time dependent plasticity to learn a foraging task. *PLoS
  Computational Biology*, 13(9), e1005705.
- Skorheim, S., Lonjers, P., & Bazhenov, M. (2014). A spiking network model of
  decision making employing rewarded STDP. *PLoS ONE*, 9(3), e90821.
- Sun, Zeng & Li (2022). Solving the spike feature information vanishing problem
  in spiking deep Q network with potential based normalization. *Frontiers in
  Neuroscience*.

## Reproducibility

All experiments are seeded. The project uses `uv` for environment management:

```bash
git clone <repo-url>
cd reward-modulated-spiking-rl
uv sync
uv run pytest