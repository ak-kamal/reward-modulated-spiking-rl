# Reward-Modulated Spiking Neural Networks for Adaptive Decision Making Under Noise

A focused investigation of whether a spiking neural network agent can learn adaptive
decision-making through a reward-modulated three-factor learning rule, and what
sparsity–performance–robustness trade-offs emerge.

## Status

🚧 **In development.** Core pipeline implemented, trained, and evaluated.
Baselines (Tabular Q, DQN, PPO) and the noise-robustness study are complete.
Sparsity experiments and ablation studies are next.

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
verified empirically. The noise-robustness study then produced one additional
finding. We document them because they reflect real constraints of
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

### Finding 4: Greedy readout collapses on symmetric tasks

Our gridworld has a symmetric optimal policy: on an empty grid, "right" and
"down" are equally good from most states. R-STDP correctly learns a distribution
with approximately equal probability over the tied actions. But the integer
spike counts used for readout are **exactly tied** in these states, and
`argmax` breaks ties by index. The greedy policy collapses to a fixed action
and fails completely (reward −1.0, 0% success), while the sampled policy
reaches the goal with 100% success.

This is not SNN-specific: **PPO exhibits the same failure** (0% greedy success
despite 100% sampled success). Both methods learn a stochastic-optimal policy
on a symmetric task, and argmax collapses the tied actions. Tabular Q and DQN
avoid this because their value estimates diverge slightly on the tied actions
during training, giving argmax a meaningful preference.

**Decision:** We report sampled-policy evaluation for both SNN and PPO, as this
reflects what the networks actually learned. Greedy collapse is documented as
a property of integer/discrete readout combined with a symmetric task.

### Finding 5: Partial robustness advantage under observation noise

We evaluated all trained agents (SNN, Tabular Q, DQN, PPO) under three
independent noise sweeps without retraining. Success rate was saturated at 1.0
across all noise levels, so we used **step count** as the discriminating metric.

**Transition noise:** All agents affected similarly (~120% increase in steps
from 0.0 to 0.5 noise). No robustness advantage for the SNN.

**Observation noise:** The SNN's step count stays flat (10.4 → 10.2) across
σ ∈ [0.0, 0.20], while Tabular Q, DQN, and PPO all show meaningful increases
(+0.78 to +1.04 steps). The SNN is **measurably more robust** to sensor noise.

**Reward noise:** Not applicable to policy behavior, since we do not retrain.
All observed deltas are within evaluation noise.

**Interpretation:** The SNN's 5-step spike-count readout acts as a low-pass
filter: brief input perturbations do not change the accumulated spike count
enough to change the sampled action. The baselines react to each noisy
observation. This mechanism helps against short-timescale input perturbations
but not against systematic action-execution failures (transition noise), which
is why the advantage is specific to observation noise.

This is a more nuanced result than a simple "stochastic is better" claim, and
it is consistent with the low-pass-filter interpretation of spike-count
integration.

## Key Design Decisions

### Three-factor rule with eligibility traces

The eligibility trace encodes recent pre×post-synaptic coincidences and decays
exponentially. This factorization is the basis of both the classical
three-factor rule (Izhikevich, 2007; Frémaux & Gerstner, 2016) and the modern
e-prop algorithm (Bellec et al., 2020).

### Actor-critic with spiking actor, value critic

Mirrors SpikingJelly's A2C example and the BSVogler actor-critic framework.
We use TD error as the modulation signal (default) with the option to switch
to a running reward baseline.

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
maximum sparsity is not optimal for supervised SNNs — we extend this question
to reward-modulated spiking RL.

## Results Summary

### Task performance

5×5 gridworld, goal at (4,4), dense reward, 100 evaluation episodes.

| Policy | Mean reward | Success rate | Mean steps |
|--------|-------------|--------------|------------|
| Random | −0.17 | 56% | 74 |
| BFS-optimal | +0.93 | 100% | 8 |
| Tabular Q (greedy) | +0.93 | 100% | 8 |
| DQN (greedy) | +0.93 | 100% | 8 |
| PPO (sampled) | +0.92 | 100% | 9 |
| **SNN (sampled)** | **+0.91** | **100%** | **10** |

The SNN matches the success rate of the conventional baselines while using
~9,300 parameters and training the actor entirely with local three-factor rules
(no backprop through the actor).

### Noise robustness (step count change, 0.0 → max noise)

| Agent | Transition (0 → 0.5) | Observation (0 → 0.2) | Reward (0 → 2.0) |
|-------|----------------------|----------------------|-------------------|
| Tabular Q | +124% | +13% | 0% |
| DQN | +124% | +11% | 0% |
| PPO | +124% | +9% | 0% |
| SNN | +117% | **−2% (flat)** | 0% |

The SNN is measurably more robust to observation noise, equally affected by
transition noise, and unaffected by reward noise (as expected).

## Repository Structure

src/spiking_rl/
├── environment/ # Gridworld + noise mechanisms
├── models/ # LIF neurons, SNN weight init, actor-critic
├── learning/ # Eligibility trace, three-factor learner, agent
├── baselines/ # Tabular Q, DQN, PPO
└── utils/ # Logging, plotting

notebooks/ # Exploration and analysis
tests/ # Unit tests
scripts/ # Diagnostics (threshold sweep)
results/ # Figures and metrics


## Experiments (Planned)

1. **Sparsity–performance trade-off.** Sweep the lateral-inhibition gain and
   measure spikes per episode against success rate (Pareto curve).
2. **Eligibility-trace ablation.** Compare the full three-factor rule against
   immediate reward-modulated STDP without traces.
3. **Architecture ablation.** Compare 1 vs 2 hidden layers, following the
   Zanatta et al. (2024) finding that shallower SNN-RL topologies perform better.

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
- Mnih, V. et al. (2015). Human-level control through deep reinforcement
  learning. *Nature*, 518, 529–533.
- Potjans, W., Diesmann, M., & Morrison, A. (2011). A spiking neural network
  model of an actor-critic learning agent. *Neural Computation*, 23(2), 269–328.
- Sanda, P., Skorheim, S., & Bazhenov, M. (2017). Multi-layer network utilizing
  rewarded spike time dependent plasticity to learn a foraging task. *PLoS
  Computational Biology*, 13(9), e1005705.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
  Proximal policy optimization algorithms. arXiv:1707.06347.
- Skorheim, S., Lonjers, P., & Bazhenov, M. (2014). A spiking network model of
  decision making employing rewarded STDP. *PLoS ONE*, 9(3), e90821.
- Sun, Zeng & Li (2022). Solving the spike feature information vanishing problem
  in spiking deep Q network with potential based normalization. *Frontiers in
  Neuroscience*.
- Zanatta, L. et al. (2024). Comparing SNNs and ANNs for deep reinforcement
  learning. *Scientific Reports*.

## Reproducibility

All experiments are seeded. The project uses `uv` for environment management:

```bash
git clone <repo-url>
cd reward-modulated-spiking-rl
uv sync
uv run pytest