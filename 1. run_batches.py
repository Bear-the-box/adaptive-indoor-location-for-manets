#!/usr/bin/env python3
import os
import sys
import subprocess

PY = sys.executable
SIM = os.path.join(os.path.dirname(__file__), "simulate_positioning_adaptive_cli.py")

# ================================
# Scenario grids (edit as you like)
# ================================
densities = {
    "HD": dict(num_nodes=40, area_size=20.0),  # High density
    "LD": dict(num_nodes=15, area_size=30.0),  # Low density
}
mobs = {"L": 0.15, "H": 0.80}                  # mobility_sigma
noises = {"L": 0.01, "H": 0.06}                # noise_std
energies = {
    "H": dict(energy_budget_total=400, energy_per_anchor_per_window=1.0),
    "L": dict(energy_budget_total=80,  energy_per_anchor_per_window=1.5),
}
seeds = [1, 2]  # bump for more repetitions

# Fixed-K sweep to map accuracy–cost Pareto
fixed_K_list = [3, 6, 9, 12]

# ================================
# Common defaults for all runs
# ================================
base_common = dict(
    alpha=0.1,
    allow_reflection=True,
    K_min=3, K_max=20,
    num_windows=40,
    # Adaptive controller (used only by method=adaptive)
    K_init=2,
    step=1,
    cooldown_after_change=2,
    # Retained bands (logged only; objective-based K is used)
    target_low=1.2, target_high=2.2,
    patience_up=3, patience_down=2,
    # Cost
    lambda_cost=0.3, cost_per_anchor=0.6,
    # Guards (turn on if you see spikes)
    spike_guard=False,
    per_node_jump_guard=False,
    max_node_jump=2.0,
    # Output (no frames/gif in batches to save time/space)
    save_frames=False, gif=False, fps=4
)

def run_one(run_name, **kwargs):
    cmd = [PY, SIM, "--run_name", run_name]
    for k, v in kwargs.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd)

def main():
    for D, Dcfg in densities.items():
        for M, ms in mobs.items():
            for N, ns in noises.items():
                for E, Ecfg in energies.items():
                    for s in seeds:
                        common = dict(base_common, **Dcfg, mobility_sigma=ms, noise_std=ns, seed=s, **Ecfg)

                        # === Adaptive (three relative-map modes) ===
                        # A1) Adaptive + classical MDS
                        rn = f"{D}-M{M}-N{N}-E{E}-ADAPT-CLASSICAL-S{s}"
                        run_one(rn, method="adaptive", adaptive_rel_mode="classical", **common)

                        # A2) Adaptive + graph_mds (geodesic distances)
                        rn = f"{D}-M{M}-N{N}-E{E}-ADAPT-GMDS-S{s}"
                        run_one(rn, method="adaptive", adaptive_rel_mode="GMDS",
                                contact_threshold=0.5, **common)

                        # A3) Adaptive + SMACOF
                        rn = f"{D}-M{M}-N{N}-E{E}-ADAPT-SMACOF-S{s}"
                        run_one(rn, method="adaptive", adaptive_rel_mode="SMACOF",
                                smacof_iters=100, **common)

                        # === Baselines ===
                        # B1) Fixed-K (no adaptation) sweep
                        for Kf in fixed_K_list:
                            rn = f"{D}-M{M}-N{N}-E{E}-FIXK{Kf}-S{s}"
                            run_one(rn, method="fixed", K_fixed=Kf, **common)

                        # B2) Graph-MDS baseline (K fixed = 6 by default)
                        rn = f"{D}-M{M}-N{N}-E{E}-GMDS-S{s}"
                        run_one(rn, method="GMDS", K_fixed=7, contact_threshold=0.5, **common)

                        # B3) SMACOF baseline (K fixed = 6)
                        rn = f"{D}-M{M}-N{N}-E{E}-SMACOF-S{s}"
                        run_one(rn, method="SMACOF", K_fixed=7, smacof_iters=100, **common)

if __name__ == "__main__":
    main()
