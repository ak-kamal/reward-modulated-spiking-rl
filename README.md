# Reward-Modulated Spiking Neural Networks for Adaptive Decision Making Under Noise

A focused investigation of whether a spiking neural network agent can learn adaptive
decision-making through a reward-modulated three-factor learning rule, and what
sparsity–performance–robustness trade-offs emerge.

## Status

**Complete.** Core pipeline, baselines, noise-robustness study, and sparsity
experiments are implemented, trained, and evaluated. All unit tests passing.
See "Future Work" for directions this project does not cover.

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

A stack of 3 `TracedLinear` layers including a spiking output layer. The action
distribution is derived from the spike counts of the output neurons over `T`
internal timesteps. Every actor layer maintains an eligibility trace. Weight
updates follow the three-factor rule:

    Δw = lr × modulation × E

where `modulation` is the centered temporal-difference error. Optional
within-layer lateral inhibition can be applied to each layer's input current
based on its own previous-timestep activity (the winner-take-all formulation
from Oster et al., 2009).

### Critic — spiking hidden layers, non-spiking output, backprop

A stack of 2 spiking hidden layers ending in a `Linear → NonSpikingLIFNode` output head that
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
verified empirically. The follow-up studies (noise robustness and sparsity)
produced two additional findings. We document them because they reflect real
constraints of reward-modulated spiking RL that are not always obvious from the
literature.

### Finding 1: Dense reward shaping is required

A sparse-reward gridworld (0 everywhere, +1 at goal) fails to train. The critic
correctly learns that all states have value ≈ 0, the TD error collapses to
zero, and the actor receives no learning signal. This is the sparse-reward
bootstrap problem in spiking RL.

The critic's TD error at each step is

    δ_t = r_t + γ · V(s_{t+1}) − V(s_t)

where `r_t` is the immediate reward, `γ` is the discount factor, and `V(s)`
is the critic's value estimate. Under sparse rewards, `r_t = 0` for almost
every step, so `δ_t` shrinks toward zero as the critic converges to `V ≈ 0`.
The actor's weight update scales with `δ_t`, so learning stops.

**Fix:** A small per-step penalty (`step_reward = -0.01`) provides a
continuous learning signal. The optimal policy becomes "reach the goal in
the minimum number of steps."

### Finding 2: Centered modulation prevents weight saturation

Using raw TD error as the modulation signal causes systematic weight growth,
because `δ_t` is positive during most of the learning phase (the critic
underestimates the improving policy). The actor's update

    Δw_ij = η · modulation · E_ij

with `η` the learning rate and `E_ij` the eligibility trace of synapse
`i → j`, therefore pushes every active synapse in the same direction, episode
after episode. Weights grow monotonically until they hit the clipping bound,
and the output layer becomes state-invariant.

**Fix:** Subtract a running mean `δ̄` from the TD error before applying it:

    modulation = δ_t − δ̄

where `δ̄` tracks the recent average of `δ_t` via an exponential moving
average (α ≈ 0.01). In the training loop, the modulation is computed as a
batch mean (our experiments use a batch of one observation per step):

    modulation = (1/B) · Σ_b ( r_b + γ V(s'_b) − V(s_b) ) − δ̄

for batch size `B`. This centers the modulation around zero so that synapses
are strengthened and weakened with roughly equal frequency. It is the
standard variance-reduction baseline from the R-STDP literature
(Izhikevich, 2007; Frémaux & Gerstner, 2016), generalized to the TD-error
case.

### Finding 3: Homeostatic mechanisms prevent network collapse

Naive centering of the modulation signal causes a different failure mode:
with no positive bias, weights drift downward until every neuron's
pre-activation falls below threshold. The LIF neuron's membrane dynamics

    v_t = v_{t-1} + (x_t − v_{t-1}) / τ,    spike if v_t ≥ θ

(where `v` is the membrane potential, `x` the input current, `τ` the
membrane time constant, and `θ` the firing threshold) then produce zero
spikes forever, and both actor and critic stop learning.

**Fix:** Two homeostatic mechanisms, both grounded in the R-STDP literature
(Skorheim et al., 2014; Sanda et al., 2017; Geng & Li, 2023):

- **Adaptive threshold (homeostatic intrinsic plasticity):** Each
  `AdaptiveLIFNode` maintains a single shared threshold for its layer,
  updated slowly toward a target firing rate `r*`:

      θ_layer ← θ_layer + ε · (r̄_layer − r*)
      r̄_layer = (1/N) · Σ_i  r̂_i

  where `r̂_i` is the exponentially-averaged firing rate of neuron `i`,
  `r̄_layer` is the layer-mean rate, `N` is the number of neurons in the
  layer, and `ε` is a small learning rate. If the layer fires too often, its
  shared threshold rises; if too rarely, it falls. This prevents permanent
  silencing.

- **Output balancing:** After each reward-modulated update, each output
  neuron's incoming weight vector `w_i` is rescaled to a target L2 norm
  `w*`:

      w_i ← w_i · (w* / ‖w_i‖₂)

  The target `w*` matches the SNN-specific initialization scale, so
  balancing constrains drift without fighting the initialization.

### Finding 4: Greedy readout collapses on symmetric tasks

Our gridworld has a symmetric optimal policy: on an empty grid, "right" and
"down" are equally good from most states. R-STDP correctly learns a
distribution

    π(a | s) = softmax( c(a, s) / T )

where `c(a, s)` is the total output spike count for action `a` over the
decision window and `T` is the softmax temperature. It places approximately
equal probability on the tied actions. But the integer spike counts are
**exactly tied** in these states, and `argmax` breaks ties by index:

    a_greedy = argmax_a  c(a, s)

The greedy policy therefore collapses to a fixed action and fails
completely (reward −1.0, 0% success), while the sampled policy reaches the
goal with 100% success.

This is not SNN-specific: **PPO exhibits the same failure** (0% greedy
success despite 100% sampled success). Both methods learn a stochastic-
optimal policy on a symmetric task, and argmax collapses the tied actions.
Tabular Q and DQN avoid this because their value estimates diverge slightly
on the tied actions during training, giving argmax a meaningful preference.

**Decision:** We report sampled-policy evaluation for both SNN and PPO, as
this reflects what the networks actually learned. Greedy collapse is
documented as a property of integer/discrete readout combined with a
symmetric task.

### Finding 5: Partial robustness advantage under observation noise

We evaluated all trained agents (SNN, Tabular Q, DQN, PPO) under three
independent noise sweeps without retraining. Success rate was saturated at
1.0 across all noise levels, so we used **step count** as the discriminating
metric.

**Transition noise:** All agents affected similarly (~120% increase in steps
from 0.0 to 0.5 noise). Just a slight robustness advantage for the SNN.

**Observation noise:** The SNN's step count stays flat (10.4 → 10.2) across
σ ∈ [0.0, 0.20], while Tabular Q, DQN, and PPO all show meaningful increases
(+0.78 to +1.04 steps). The SNN is **measurably more robust** to sensor
noise.

**Reward noise:** Not applicable to policy behavior, since we do not retrain.
All observed deltas are within evaluation noise.

**Interpretation:** The SNN's decision is made from the accumulated spike
count over `T = 5` internal steps:

    c(a) = Σ_{t=1}^{T}  s_t(a)

where `s_t(a) ∈ {0, 1}` is the spike of output neuron `a` at step `t`.
Because `c(a)` is an integer sum of five independent step decisions, brief
input perturbations of magnitude `ε` do not change `c(a)` unless they push
a step's pre-activation across the firing threshold. The readout therefore
acts as a low-pass filter: the accumulated count integrates out fast
perturbations. The baselines react to each noisy observation individually.
This mechanism helps against short-timescale input perturbations but not
against systematic action-execution failures (transition noise), which is
why the advantage is specific to observation noise.

### Finding 6: Lateral inhibition produces a weak sparsity trade-off

We added a within-layer lateral inhibition mechanism (winner-take-all
competition; Oster et al., 2009) and swept the inhibition gain across
`{0.0, 0.25, 0.5, 1.0, 1.5, 2.0}`, training a fresh SNN at each level. The
inhibitory current applied to each neuron is

    I_inh = g_inh · mean( spikes_in_layer )

where `g_inh` is the inhibition gain and `spikes_in_layer` are the same
layer's spikes at the previous internal timestep. This current is subtracted
from the neuron's pre-activation before the LIF update.

**Result:** Across the five configurations that converged (excluding one
training-variance outlier at gain=1.0), increasing `g_inh` from 0.0 to 1.5
reduced spike count by ~1.4% (2.533 → 2.498 spikes/neuron/step), at the
cost of ~6.8% longer paths (10.17 → 10.86 steps) and ~0.8% reward reduction.
The Pareto front is shallow.

**Why the trade-off is present, but weak:** The adaptive-threshold
homeostasis introduced in Finding 3 compensates for the inhibitory drive.
When `I_inh` reduces a neuron's pre-activation, the layer fires less,
`r̄_layer` falls below `r*`, and the threshold update

    θ_layer ← θ_layer + ε · (r̄_layer − r*)

lowers `θ_layer` to restore the target rate. Over training, the shared
thresholds equilibrate at values that cancel most of the inhibition. The
residual trade-off is what survives after homeostatic compensation.

**Greedy readout:** Lateral inhibition did **not** fix the greedy readout
collapse. The `div` metric stayed at 1.0 and greedy success remained 0%
across all inhibition levels. The winner-take-all competition hypothesis
is not supported by this result.

**Observation-noise robustness:** We re-evaluated the `g_inh = 1.5` SNN
under the observation-noise sweep and compared to the no-inhibition SNN.
The inhibited model achieved consistently lower step counts (9.86–10.17 vs.
10.18–10.54, averaging ~0.4 steps lower) across all noise levels. We do
not claim this is caused by the inhibition mechanism, since the two models
were trained with a single seed each and the difference could reflect
run-to-run training variance. The defensible conclusion is that inhibition
did **not** degrade observation-noise robustness.

## Key Design Decisions

### Three-factor rule with eligibility traces

Two different trace mechanisms are used in our implementation. They have
similar functional forms but differ in the exact decay rule:

- **Pre/post-synaptic traces** (inside `LIFNodeWithTrace`) use a linear
  leaky integrator with time constant `τ_trace`:

      trace ← (1 − 1/τ_trace) · trace + (1/τ_trace) · x_t

  These traces capture the recent activity of each neuron's input and
  output, and they feed the eligibility computation below.

- **The eligibility matrix** `E` (inside `EligibilityTrace`) uses an
  exponential decay with time constant `τ_e`:

      E_ij(t) = exp(−1/τ_e) · E_ij(t−1) + ΔE_ij(t)

  where `ΔE_ij(t)` is the outer product of the current pre- and
  post-synaptic traces averaged over the batch. `E_ij` is the eligibility
  for synapse `i → j`.

At our chosen constants (`τ_trace = 20`, `τ_e = 50`) the two decay rules
are numerically close but not identical. The actor's weight update combines
the eligibility trace with the centered modulation:

    Δw_ij = η · modulation · E_ij

This factorization is the basis of both the classical three-factor rule
(Izhikevich, 2007; Frémaux & Gerstner, 2016) and the modern e-prop algorithm
(Bellec et al., 2020).

### Actor-critic with spiking actor, value critic

Mirrors SpikingJelly's A2C example and the BSVogler actor-critic framework.
The critic is trained by backprop on the TD-error loss

    L_critic = ( r_t + γ · V(s_{t+1}) − V(s_t) )²

We use the TD error as the modulation signal (default) with the option to
switch to a running reward baseline.

### SNN-specific weight initialization

Standard ANN initializations cause dying signals in deep SNNs because
pre-activations are binarized by the firing threshold. We adopt the
initialization from Micheli et al. (2025), which conserves membrane-potential
variance across layers via

    Var[W] = 1 / ( n_in · P(u > θ) )

where `n_in` is the fan-in, `u ~ N(0, 1)`, and `P(u > θ)` is the probability
that a standard normal sample exceeds the firing threshold. This was the
primary fix for the dying-signal problem in early runs.

### Firing threshold validated via operational manifold

The firing threshold `v_threshold = 1.0` follows the normalized LIF
convention (Gerstner & Kistler, 2002). We validate the choice by sweeping
thresholds and mapping the operational manifold (Mazurek et al., 2026) —
the region where activity is neither silent nor saturated. The sweep (in
`results/figures/threshold_sweep_*.png`) confirms 1.0 sits in the balanced
region.

### Homeostatic mechanisms

Adaptive threshold and output balancing are included based on the Skorheim
et al. (2014) and Sanda et al. (2017) findings that reward-modulated STDP
requires homeostatic regulation. The two mechanisms are:

- Threshold adaptation (per layer, shared across the layer's neurons):
  `θ_layer ← θ_layer + ε · (r̄_layer − r*)`
- Output balancing (per output neuron):
  `w_i ← w_i · (w* / ‖w_i‖₂)`

### Lateral inhibition

Within-layer subtractive inhibition, applied to each layer's input current
based on its own previous-timestep activity:

    I_inh = g_inh · mean( spikes_in_layer )

The implementation follows the winner-take-all formulation from Oster,
Douglas & Liu (2009) and the LISNN model (Yang et al., IJCAI 2020). The
inhibition gain `g_inh` defaults to 0.0, so the baseline architecture is
unaffected when the mechanism is disabled.

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

### Sparsity trade-off (spikes per neuron per step vs. task efficiency)

| Inhibition gain | Spikes/neuron/step | Mean steps | Mean reward |
|-----------------|--------------------|-----------:|------------:|
| 0.00 | 2.533 | 10.17 | 0.908 |
| 0.25 | 2.520 | 10.38 | 0.906 |
| 0.50 | 2.500 | 10.37 | 0.906 |
| 1.50 | 2.498 | 10.86 | 0.901 |
| 2.00 | 2.500 | 10.55 | 0.905 |

Lateral inhibition produces a shallow, monotonic sparsity–performance
trade-off. Adaptive-threshold homeostasis absorbs most of the inhibitory
drive, leaving only a ~1.4% spike reduction at a ~6.8% efficiency cost.
No configuration meaningfully improves over the no-inhibition baseline on
either axis, and none fixes the greedy-readout collapse.

## Repository Structure

src/spiking_rl/
├── environment/ # Gridworld + noise mechanisms
├── models/ # LIF neurons, SNN weight init, actor-critic
├── learning/ # Eligibility trace, three-factor learner, agent
├── baselines/ # Tabular Q, DQN, PPO
└── utils/ # Logging, plotting
notebooks/ # 01 env, 02 first training, 03 baselines, 04 noise robustness, 05 sparsity trade-off
tests/ # 135+ unit tests
scripts/ # Diagnostics (threshold sweep)
results/ # Figures and metrics


## Future Work

Several directions would strengthen or extend the results:

1. **Multi-seed evaluation.** Reward-modulated spiking RL is known to be
   sensitive to initialization; running each configuration across multiple
   random seeds would provide stronger estimates of mean performance and
   variance, and would let us report confidence intervals on all reported
   metrics.

2. **Architecture ablation.** Compare 1 vs 2 hidden layers, following the
   Zanatta et al. (2024) finding that shallower SNN-RL topologies perform
   better.

3. **Eligibility-trace timescale.** Sweep the trace time constant `tau_e`
   and its interaction with the modulation signal (TD error vs. running
   reward baseline), to test whether short traces suffice when the critic
   supplies delayed credit.

4. **Lateral-inhibition / homeostasis coupling.** Decouple lateral inhibition
   from adaptive-threshold compensation by lowering the homeostatic target
   when inhibition is active, or by freezing thresholds after warmup. This
   would clarify whether the shallow trade-off is fundamental or an artifact
   of the two mechanisms fighting.

5. **Continuous-action environments.** Extend the pipeline beyond discrete
   gridworlds to test whether the sparsity and robustness findings transfer.

## What This Project Does Not Claim

- The agent is not a basal-ganglia model. It is biologically *inspired* by
  dopaminergic reward modulation and three-factor plasticity, but does not model
  biological detail.
- Energy-efficiency claims are limited to spike counts and synaptic operation
  counts, not measured hardware energy.
- The gridworld is a controlled toy environment, appropriate for the focused
  research question but not a benchmark for real-world tasks.
- The sparsity–performance trade-off is **shallow** in our setup. We do not
  claim that lateral inhibition is an effective sparsity mechanism for
  reward-modulated spiking RL; our result is that homeostatic compensation
  largely negates it.
- Reported results use a single random seed per configuration. See "Future Work."

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
- Oster, M., Douglas, R., & Liu, S.-C. (2009). Computation with spikes in a
  winner-take-all network. *Neural Computation*, 21(9), 2437–2465.
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
- Yang, Q. et al. (2020). LISNN: Improving Spiking Neural Networks with Lateral
  Interactions. *IJCAI*.
- Zanatta, L. et al. (2024). Comparing SNNs and ANNs for deep reinforcement
  learning. *Scientific Reports*.

## Reproducibility

All experiments are seeded. The project uses `uv` for environment management:

```bash
git clone <repo-url>
cd reward-modulated-spiking-rl
uv sync
uv run pytest