"""
Operational manifold diagnostic for the actor-critic SNN.

This script sweeps neuron firing thresholds and measures per-layer firing
rates, following the operational manifold methodology of:

    Mazurek, Caputa, Maj, Wielgosz (2026),
    "Operational manifolds in spiking neural networks",
    Frontiers in Neuroscience, vol. 20, art. 1755119.

The operational manifold is the region in neuron hyperparameter space
where spiking activity is balanced (neither silent nor saturated) while
task performance is maintained. In this diagnostic we measure the
firing-rate condition of the manifold, which is the necessary half we
can evaluate before training.

Two sweeps are performed:

1. 1D sweep over the actor's output-layer firing threshold
   (output_v_threshold), with the hidden threshold fixed at 1.0.
2. 2D sweep over (hidden v_threshold, output_v_threshold) to map the
   balanced region as a heatmap.

For each configuration we measure:

- Per-layer mean firing rate (spikes per neuron per timestep).
- Fraction of neurons in each layer that ever fire.
- Actor output-layer total spike count over the decision window.
- Entropy of the softmax action distribution.

Outputs:
    results/metrics/threshold_sweep_1d.csv
    results/metrics/threshold_sweep_2d.csv
    results/figures/threshold_sweep_1d.png
    results/figures/threshold_sweep_2d.png

Usage:
    uv run python scripts/diagnose_threshold.py
    uv run python scripts/diagnose_threshold.py --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from spiking_rl.models.actor_critic import ActorCritic


# ======================================================================
# Configuration
# ======================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIGURES_DIR = REPO_ROOT / "results" / "figures"

# Network configuration for the diagnostic (small, fast).
OBS_DIM = 4
NUM_ACTIONS = 4
HIDDEN_DIMS = (64, 64)
DECISION_STEPS = 20
BATCH_SIZE = 128
OBS_SCALE = 20.0
SEED = 42

# Default thresholds (the "current" configuration to mark on plots).
DEFAULT_HIDDEN_THRESHOLD = 1.0
DEFAULT_OUTPUT_THRESHOLD = 1.0


# ======================================================================
# Diagnostics helpers
# ======================================================================


def run_actor_diagnostics(
    actor,
    obs: torch.Tensor,
    steps: int,
) -> dict:
    """Run the actor for `steps` steps and collect per-layer spike stats.

    Returns a dict with:
        layer_spike_counts: list of tensors, one per layer, shape
            (batch, out_features), total spikes over the run.
        layer_firing_rates: list of floats, mean fraction of neurons
            firing per timestep.
        layer_frac_active: list of floats, fraction of neurons that
            fired at least once.
    """
    x = obs * actor.obs_scale
    layer_spike_counts = [
        torch.zeros(obs.shape[0], layer.out_features)
        for layer in actor.layers
    ]

    for _ in range(steps):
        h = x
        for i, layer in enumerate(actor.layers):
            h = layer(h)
            layer_spike_counts[i] = layer_spike_counts[i] + h

    firing_rates = []
    frac_active = []
    for counts in layer_spike_counts:
        # Mean spikes per neuron per timestep.
        firing_rates.append(float((counts / steps).mean().item()))
        # Fraction of neurons that fired at least once.
        frac_active.append(float((counts > 0).float().mean().item()))

    return {
        "layer_spike_counts": layer_spike_counts,
        "layer_firing_rates": firing_rates,
        "layer_frac_active": frac_active,
    }


def run_critic_diagnostics(
    critic,
    obs: torch.Tensor,
    steps: int,
) -> dict:
    """Run the critic for `steps` steps and collect hidden-layer spike stats."""
    x = obs * critic.obs_scale
    hidden_spike_counts = [
        torch.zeros(obs.shape[0], layer.out_features)
        for layer in critic.hidden
    ]

    for _ in range(steps):
        h = x
        for i, layer in enumerate(critic.hidden):
            h = layer(h)
            hidden_spike_counts[i] = hidden_spike_counts[i] + h
        # Skip the output head; it's non-spiking.

    firing_rates = [
        float((counts / steps).mean().item()) for counts in hidden_spike_counts
    ]
    frac_active = [
        float((counts > 0).float().mean().item()) for counts in hidden_spike_counts
    ]

    return {
        "layer_spike_counts": hidden_spike_counts,
        "layer_firing_rates": firing_rates,
        "layer_frac_active": frac_active,
    }


def action_entropy(spike_counts: torch.Tensor, temperature: float = 1.0) -> float:
    """Entropy of the softmax action distribution over spike counts."""
    probs = torch.softmax(spike_counts / temperature, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()
    return float(entropy)


# ======================================================================
# Sweep functions
# ======================================================================


def sweep_output_threshold(
    output_thresholds: list[float],
    hidden_threshold: float,
    obs: torch.Tensor,
    seed: int = SEED,
) -> pd.DataFrame:
    """1D sweep over the actor's output-layer firing threshold."""
    rows = []

    for out_th in output_thresholds:
        torch.manual_seed(seed)  # deterministic init
        ac = ActorCritic(
            obs_dim=OBS_DIM,
            num_actions=NUM_ACTIONS,
            actor_hidden=HIDDEN_DIMS,
            critic_hidden=HIDDEN_DIMS,
            obs_scale=OBS_SCALE,
            decision_steps=DECISION_STEPS,
            v_threshold=hidden_threshold,
            output_v_threshold=out_th,
            init_snn=True,
        )
        ac.reset()

        actor_diag = run_actor_diagnostics(ac.actor, obs, DECISION_STEPS)
        critic_diag = run_critic_diagnostics(ac.critic, obs, DECISION_STEPS)

        # Actor output layer is the last one.
        output_counts = actor_diag["layer_spike_counts"][-1]
        output_mean_spike_count = float(output_counts.sum(dim=-1).mean().item())
        entropy = action_entropy(output_counts)

        # Record per-layer actor stats.
        for i, (fr, fa) in enumerate(
            zip(actor_diag["layer_firing_rates"], actor_diag["layer_frac_active"])
        ):
            rows.append({
                "sweep": "1d_output",
                "hidden_v_threshold": hidden_threshold,
                "output_v_threshold": out_th,
                "network": "actor",
                "layer_index": i,
                "layer_name": f"actor_layer_{i}",
                "firing_rate": fr,
                "frac_active": fa,
                "output_mean_spike_count": output_mean_spike_count,
                "action_entropy": entropy,
            })

        # Record critic hidden-layer stats.
        for i, (fr, fa) in enumerate(
            zip(critic_diag["layer_firing_rates"], critic_diag["layer_frac_active"])
        ):
            rows.append({
                "sweep": "1d_output",
                "hidden_v_threshold": hidden_threshold,
                "output_v_threshold": out_th,
                "network": "critic",
                "layer_index": i,
                "layer_name": f"critic_hidden_{i}",
                "firing_rate": fr,
                "frac_active": fa,
                "output_mean_spike_count": output_mean_spike_count,
                "action_entropy": entropy,
            })

    return pd.DataFrame(rows)


def sweep_2d(
    hidden_thresholds: list[float],
    output_thresholds: list[float],
    obs: torch.Tensor,
    seed: int = SEED,
) -> pd.DataFrame:
    """2D sweep over (hidden v_threshold, output_v_threshold)."""
    rows = []

    for hid_th in hidden_thresholds:
        for out_th in output_thresholds:
            torch.manual_seed(seed)
            ac = ActorCritic(
                obs_dim=OBS_DIM,
                num_actions=NUM_ACTIONS,
                actor_hidden=HIDDEN_DIMS,
                critic_hidden=HIDDEN_DIMS,
                obs_scale=OBS_SCALE,
                decision_steps=DECISION_STEPS,
                v_threshold=hid_th,
                output_v_threshold=out_th,
                init_snn=True,
            )
            ac.reset()

            actor_diag = run_actor_diagnostics(ac.actor, obs, DECISION_STEPS)

            # Network-level actor firing rate (mean across all actor layers).
            net_fr = float(np.mean(actor_diag["layer_firing_rates"]))
            out_fr = actor_diag["layer_firing_rates"][-1]
            out_fa = actor_diag["layer_frac_active"][-1]

            output_counts = actor_diag["layer_spike_counts"][-1]
            entropy = action_entropy(output_counts)

            rows.append({
                "hidden_v_threshold": hid_th,
                "output_v_threshold": out_th,
                "actor_network_firing_rate": net_fr,
                "actor_output_firing_rate": out_fr,
                "actor_output_frac_active": out_fa,
                "action_entropy": entropy,
            })

    return pd.DataFrame(rows)


# ======================================================================
# Plotting
# ======================================================================


def plot_1d_sweep(df: pd.DataFrame, out_path: Path) -> None:
    """Plot firing rates and action statistics vs output threshold."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # --- Panel 1: per-layer firing rate (actor) ---
    ax = axes[0, 0]
    actor_df = df[df["network"] == "actor"]
    for layer_name in actor_df["layer_name"].unique():
        sub = actor_df[actor_df["layer_name"] == layer_name]
        ax.plot(
            sub["output_v_threshold"], sub["firing_rate"],
            "o-", label=layer_name.replace("actor_", ""), linewidth=2,
        )
    ax.axvline(
        DEFAULT_OUTPUT_THRESHOLD, color="red", linestyle="--",
        label=f"current default ({DEFAULT_OUTPUT_THRESHOLD})",
    )
    ax.set_xlabel("output v_threshold")
    ax.set_ylabel("firing rate (spikes/neuron/timestep)")
    ax.set_title("Actor: per-layer firing rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 2: output-layer spike count ---
    ax = axes[0, 1]
    out_df = actor_df[actor_df["layer_name"] == "actor_layer_2"]
    ax.plot(
        out_df["output_v_threshold"], out_df["output_mean_spike_count"],
        "o-", color="steelblue", linewidth=2,
    )
    ax.axvline(DEFAULT_OUTPUT_THRESHOLD, color="red", linestyle="--")
    ax.set_xlabel("output v_threshold")
    ax.set_ylabel("mean total spikes per action neuron")
    ax.set_title("Actor: output-layer spike count")
    ax.grid(alpha=0.3)

    # --- Panel 3: action entropy ---
    ax = axes[1, 0]
    ax.plot(
        out_df["output_v_threshold"], out_df["action_entropy"],
        "o-", color="seagreen", linewidth=2,
    )
    ax.axhline(
        np.log(NUM_ACTIONS), color="gray", linestyle=":",
        label=f"max entropy = log({NUM_ACTIONS}) = {np.log(NUM_ACTIONS):.2f}",
    )
    ax.axvline(DEFAULT_OUTPUT_THRESHOLD, color="red", linestyle="--")
    ax.set_xlabel("output v_threshold")
    ax.set_ylabel("action entropy (nats)")
    ax.set_title("Action distribution entropy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 4: fraction of neurons ever firing ---
    ax = axes[1, 1]
    for layer_name in actor_df["layer_name"].unique():
        sub = actor_df[actor_df["layer_name"] == layer_name]
        ax.plot(
            sub["output_v_threshold"], sub["frac_active"],
            "o-", label=layer_name.replace("actor_", ""), linewidth=2,
        )
    ax.axvline(DEFAULT_OUTPUT_THRESHOLD, color="red", linestyle="--")
    ax.set_xlabel("output v_threshold")
    ax.set_ylabel("fraction of neurons that ever fire")
    ax.set_title("Actor: fraction of active neurons")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_2d_sweep(df: pd.DataFrame, out_path: Path) -> None:
    """Plot 2D heatmaps of network and output firing rates."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Reshape into grid for pcolormesh.
    hidden_vals = sorted(df["hidden_v_threshold"].unique())
    output_vals = sorted(df["output_v_threshold"].unique())

    def grid(column):
        pivot = df.pivot(
            index="hidden_v_threshold",
            columns="output_v_threshold",
            values=column,
        )
        return pivot.reindex(index=hidden_vals, columns=output_vals).values

    for ax, column, title, cmap in [
        (axes[0], "actor_output_firing_rate",
         "Actor output-layer firing rate", "viridis"),
        (axes[1], "actor_network_firing_rate",
         "Actor network firing rate", "viridis"),
        (axes[2], "action_entropy",
         "Action entropy", "magma"),
    ]:
        data = grid(column)
        pcm = ax.pcolormesh(
            output_vals, hidden_vals, data,
            shading="auto", cmap=cmap,
        )
        ax.set_xlabel("output v_threshold")
        ax.set_ylabel("hidden v_threshold")
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax)
        # Mark the current default.
        ax.plot(
            DEFAULT_OUTPUT_THRESHOLD, DEFAULT_HIDDEN_THRESHOLD,
            "r*", markersize=15, markeredgecolor="white",
            label="current default",
        )
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only the 1D sweep (skip the 2D sweep).",
    )
    args = parser.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Fixed batch of random observations for all configurations.
    torch.manual_seed(SEED)
    obs = torch.rand(BATCH_SIZE, OBS_DIM)

    # --- 1D sweep over output threshold ---
    output_thresholds = np.linspace(0.05, 2.0, 25).tolist()
    print(f"Running 1D sweep over {len(output_thresholds)} output thresholds...")
    df_1d = sweep_output_threshold(
        output_thresholds,
        hidden_threshold=DEFAULT_HIDDEN_THRESHOLD,
        obs=obs,
    )
    df_1d.to_csv(METRICS_DIR / "threshold_sweep_1d.csv", index=False)
    print(f"Saved: {METRICS_DIR / 'threshold_sweep_1d.csv'}")
    plot_1d_sweep(df_1d, FIGURES_DIR / "threshold_sweep_1d.png")

    # --- 2D sweep ---
    if not args.quick:
        hidden_thresholds = np.linspace(0.3, 2.0, 12).tolist()
        output_thresholds_2d = np.linspace(0.05, 1.5, 12).tolist()
        print(
            f"Running 2D sweep over "
            f"{len(hidden_thresholds)}x{len(output_thresholds_2d)} configs..."
        )
        df_2d = sweep_2d(
            hidden_thresholds,
            output_thresholds_2d,
            obs=obs,
        )
        df_2d.to_csv(METRICS_DIR / "threshold_sweep_2d.csv", index=False)
        print(f"Saved: {METRICS_DIR / 'threshold_sweep_2d.csv'}")
        plot_2d_sweep(df_2d, FIGURES_DIR / "threshold_sweep_2d.png")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("Summary: output-layer firing rate at each threshold")
    print("=" * 60)
    actor_out = df_1d[
        (df_1d["network"] == "actor")
        & (df_1d["layer_name"] == "actor_layer_2")
    ]
    for _, row in actor_out.iterrows():
        print(
            f"  out_th={row['output_v_threshold']:.3f} | "
            f"fire_rate={row['firing_rate']:.4f} | "
            f"frac_active={row['frac_active']:.3f} | "
            f"spike_count={row['output_mean_spike_count']:.2f} | "
            f"entropy={row['action_entropy']:.3f}"
        )


if __name__ == "__main__":
    main()