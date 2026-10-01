# Reward-Modulated Spiking Neural Networks for Adaptive Decision Making Under Noise

A focused investigation of whether a spiking neural network agent can learn adaptive
decision-making through a reward-modulated three-factor learning rule, and what
sparsity–performance–robustness trade-offs emerge.

## Status

🚧 **In development.** Environment, neuron models, eligibility trace mechanism,
three-factor learning rule, and actor-critic architecture are implemented and
tested. The training loop works; homeostatic mechanisms are being added to
stabilize learning.

## Research Question

Can a spiking neural network agent learn adaptive decision-making in a noisy
gridworld environment using a biologically grounded three-factor learning rule,
and how does performance trade off against spike sparsity and robustness to
noise?

## Motivation

Spiking neural networks (SNNs) are attractive for reinforcement learning
because they process information through sparse, event-driven spikes. But
training them is hard: spikes are non-differentiable, rewards are often
delayed, and credit must be assigned across both space and time.

The three-factor learning rule — pre-synaptic activity × post-synaptic
activity × a global neuromodulatory signal — is a biologically grounded
solution to the temporal credit assignment problem (Izhikevich, 2007;
Frémaux & Gerstner, 2016). Its factorization into eligibility traces and a
reward signal is what makes learning from delayed rewards tractable.

## Architecture

### Actor (spiking, R-STDP)

A stack of `TracedLinear` layers ending in a spiking output layer whose
spike counts over `T` internal timesteps determine the action distribution.
Weight updates follow the three-factor rule:

    Δw = lr × (reward − baseline) × E

### Critic (spiking hidden, non-spiking output, backprop)

A stack of spiking hidden layers ending in `Linear → NonSpikingLIFNode`
that outputs a continuous scalar value. Trained by backpropagation on the
TD-error MSE loss.

### Environment

A custom gridworld implemented against the Gymnasium API with three
configurable noise mechanisms: transition noise (perpendicular slips),
observation noise (Gaussian perturbation), and reward noise (Gaussian
perturbation). The environment is deterministic under seed control and
has been verified via unit tests.

## Empirical Findings

### Finding 1: Reward shaping is required for R-STDP in sparse-reward tasks

The first training run used a sparse-reward gridworld (0 everywhere, +1 at
goal). The agent failed to learn: TD error collapsed to zero by episode 50,
actor weight updates stopped, and the policy degraded to random. This is
consistent with the known difficulty of temporal credit assignment in
spiking RL with sparse rewards.

**Fix:** A dense step penalty (`step_reward=-0.01`) was added. This gave
the agent a continuous learning signal at every step. Training then
succeeded: the agent reached 98% success rate with mean reward +0.67.

### Finding 2: Uncentered modulation saturates weights and produces state-invariant policies

The second training run used the dense reward but still used raw TD error
as the modulation signal. Because the TD error was systematically positive
during learning, every eligible synapse was strengthened in the same
direction. Weights grew monotonically until they saturated at the clipping
bound (`clip_weights=5.0`), and the output layer fired identically
regardless of the input observation. The greedy policy was degenerate
(always action 1), and success was achieved only via random-walk behavior
under softmax sampling.

**Fix:** A running-mean subtraction was applied to the TD error before it
was used as the modulation signal (`apply_modulation_centered`). This
centered the learning signal around zero so that synapses were both
strengthened and weakened.

### Finding 3: Centering alone causes silent networks without homeostatic support

The third training run added centering but also tightened the weight clip
to 2.0. The network went completely silent: output spike counts dropped to
zero by episode 50, and both the actor and critic stopped learning. This
is the classic "homeostatic imbalance" of rewarded STDP described in the
literature: centering removed the systematic positive bias, but with no
mechanism to maintain excitability, the weights drifted downward until
every neuron fell below threshold.

**Fix (in progress):** Two homeostatic mechanisms are being added:

1. **Adaptive threshold (homeostatic intrinsic plasticity):** Each
   neuron's firing threshold slowly rises when it fires above a target
   rate and falls when it fires below. This prevents permanent silencing.

2. **Output balancing:** Each output neuron's incoming weight vector is
   rescaled to a target L2 norm after each reward-modulated update. This
   prevents any single neuron from dominating the layer.

These mechanisms are grounded in the following literature:

- Skorheim, Lonjers & Bazhenov (2014), *PLoS ONE*: "Rewarded STDP is
  homeostatically unbalanced." The authors found that "if the [output
  balancing] mechanism was not implemented, performance greatly
  suffered."[reference:4]

- Sanda, Skorheim & Bazhenov (2017), *PLoS Comput Biol*: A multi-layer
  R-STDP network that required "heterosynaptic plasticity, gain control,
  output balancing, activity normalization of rewarded STDP and hard
  limits on synaptic strength."[reference:5]

- Kozdon & Bentley (2019), arXiv: "Weight drift is a known problem which
  impedes learning by leading to either weight silencing or saturation."
  "Neuron type-specific normalisation is a promising approach for
  preventing weight drift."[reference:6]

- Geng & Li (2023), HoSNN: A "threshold-adapting leaky integrate-and-fire
  (TA-LIF) neuron model" with a "self-stabilizing dynamic thresholding
  mechanism."[reference:7]

- Hertag & Sprekeler (2020), *Nature Communications*: Homeostatic
  plasticity must operate on a slower timescale than Hebbian plasticity
  to avoid oscillations.

## Key Design Decisions and Citations

### Three-factor learning rule with eligibility traces

The eligibility trace is the memory of recent pre×post-synaptic
coincidences (Izhikevich, 2007; Frémaux & Gerstner, 2016; Bellec et al.,
2020).

### Actor-critic with spiking actor, value critic

Mirrors SpikingJelly's A2C example and the BSVogler actor-critic
framework. Full separation avoids the "which rule owns this synapse"
problem that shared encoders create.

### SNN-specific weight initialization

We adopt the initialization from Micheli et al. (2025), which conserves
membrane-potential variance across layers by scaling weights according
to `Var[W] = 1 / (n_in × P(u > θ))`. This is the primary fix for the
dying-signal problem in our network.

### Firing threshold selection via operational manifold

The firing threshold `v_threshold = 1.0` follows the standard normalized
LIF convention (Gerstner & Kistler, 2002). We validate our choice by
mapping the operational manifold (Mazurek et al., 2026) — the region in
threshold space where spiking activity is neither silent nor saturated.

### Homeostatic mechanisms (added after empirical findings)

Adaptive threshold and output balancing are added based on the Skorheim
et al. (2014) and Sanda et al. (2017) findings that rewarded STDP
requires homeostatic regulation to prevent both silent and saturated
networks.

## References

- Bacho, F. & Chu, D. (2023). Exploring trade-offs in spiking neural
  networks. *Neural Computation*, 35(10), 1627–1656.
- Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein,
  R., & Maass, W. (2020). A solution to the learning dilemma for recurrent
  networks of spiking neurons. *Nature Communications*, 11, 3625.
- Frémaux, N. & Gerstner, W. (2016). Neuromodulated STDP and theory of
  three-factor learning rules. *Frontiers in Neural Circuits*, 9, 85.
- Geng, H. & Li, P. (2023). HoSNN: Adversarially-Robust Homeostatic Spiking
  Neural Networks with Adaptive Firing Thresholds. arXiv:2308.10373.
- Gerstner, W. & Kistler, W. M. (2002). *Spiking Neuron Models: Single
  Neurons, Populations, Plasticity*. Cambridge University Press.
- Hertag, L. & Sprekeler, H. (2020). Learning prediction error neurons
  through homeostatic plasticity. *Nature Communications*, 11, 3358.
- Izhikevich, E. M. (2007). Solving the distal reward problem through
  linkage of STDP and dopamine signaling. *Cerebral Cortex*, 17(10),
  2443–2452.
- Kozdon, K. & Bentley, P. (2019). Normalisation of Weights and Firing
  Rates in Spiking Neural Networks with Spike-Timing-Dependent Plasticity.
  arXiv:1910.00122.
- Mazurek, Caputa, Maj, Wielgosz (2026). Operational manifolds in spiking
  neural networks. *Frontiers in Neuroscience*.
- Micheli, Booij, van Gemert, Tomen (2025). Deep activity propagation via
  weight initialization in spiking neural networks. *IEEE*.
- Potjans, W., Diesmann, M., & Morrison, A. (2011). A spiking neural
  network model of an actor-critic learning agent. *Neural Computation*,
  23(2), 269–328.
- Sanda, P., Skorheim, S., & Bazhenov, M. (2017). Multi-layer network
  utilizing rewarded spike time dependent plasticity to learn a foraging
  task. *PLoS Computational Biology*, 13(9), e1005705.
- Skorheim, S., Lonjers, P., & Bazhenov, M. (2014). A spiking network
  model of decision making employing rewarded STDP. *PLoS ONE*, 9(3),
  e90821.
- Sun, Zeng & Li (2022). Solving the spike feature information vanishing
  problem in spiking deep Q network with potential based normalization.
  *Frontiers in Neuroscience*.

## Reproducibility

All experiments are seeded. The project uses `uv` for environment
management:

```bash
git clone <repo-url>
cd reward-modulated-spiking-rl
uv sync
uv run pytest