#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contact-based indoor localization in MANETs via MDS + Procrustes,
with multiple strategies:

Methods:
- adaptive  : self-adaptive anchors (K changes over time via objective + hysteresis + energy cap)
- fixed     : fixed K anchors (K never changes; run stops if energy budget cannot sustain K)
- graph_mds : geodesic (shortest-path) distances on a thresholded contact graph + MDS (fixed K)
- smacof    : SMACOF iterative MDS (stress majorization) (fixed K)

Extras:
- Spike guards: global RMSE spike rejection, per-node jump clamp
- Per-window frames and optional GIF
- CSV logs and a summary.csv with averages & indicators

Outputs go to: runs/<run_name>/...
"""

import argparse
import csv
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Optional GIF creation
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


# =========================
# Linear algebra & geometry
# =========================

def classical_mds(dist_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    n = dist_matrix.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (dist_matrix ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[idx][:n_components], 0)
    eigvecs = eigvecs[:, idx][:, :n_components]
    return eigvecs * np.sqrt(eigvals)


def procrustes_general(source: np.ndarray,
                       target: np.ndarray,
                       allow_reflection: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    mu_s = source.mean(axis=0); mu_t = target.mean(axis=0)
    S = source - mu_s; T = target - mu_t
    H = S.T @ T
    U, Sigma, Vt = np.linalg.svd(H)
    R = U @ Vt
    if not allow_reflection and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    s = Sigma.sum() / (S ** 2).sum()
    t = mu_t - s * (R @ mu_s)
    return float(s), R, t


def apply_transform(coords: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * coords @ R.T) + t


# =========================
# Utilities & alternatives
# =========================

def rmse(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def rmse_non_anchors(est_abs: np.ndarray, coords: np.ndarray, anchor_idx: np.ndarray) -> float:
    mask = np.ones(coords.shape[0], dtype=bool)
    mask[anchor_idx] = False
    if not mask.any():
        return 0.0
    return rmse(est_abs[mask], coords[mask])


def farthest_first_indices(points: np.ndarray, K: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = points.shape[0]
    if K >= N:
        return np.arange(N, dtype=int)
    start = rng.integers(0, N)
    selected = [int(start)]
    dists = np.linalg.norm(points - points[start], axis=1)
    for _ in range(1, K):
        idx = int(np.argmax(dists))
        selected.append(idx)
        dists = np.minimum(dists, np.linalg.norm(points - points[idx], axis=1))
    return np.array(selected, dtype=int)


def spread_indices_from_prev(prev_est_abs: Optional[np.ndarray],
                             rel_coords: np.ndarray,
                             K: int,
                             seed: Optional[int] = None) -> np.ndarray:
    """k-means++-style seeding on previous absolute map (robust), fallback to farthest-first."""
    rng = np.random.default_rng(seed)
    if prev_est_abs is None:
        return farthest_first_indices(rel_coords, K, seed=seed)
    X = prev_est_abs
    N = X.shape[0]
    if K >= N:
        return np.arange(N, dtype=int)
    idx = [int(rng.integers(0, N))]
    d2 = np.sum((X - X[idx[0]])**2, axis=1)
    for _ in range(1, K):
        probs = d2 / (d2.sum() + 1e-12)
        j = int(rng.choice(N, p=probs))
        idx.append(j)
        d2 = np.minimum(d2, np.sum((X - X[j])**2, axis=1))
    return np.array(idx, dtype=int)


def train_val_split(anchor_idx: np.ndarray,
                    val_ratio: float = 0.3,
                    seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    K = len(anchor_idx)
    if K <= 2:
        if K == 1:
            return anchor_idx, anchor_idx
        perm = anchor_idx.copy()
        rng.shuffle(perm)
        return perm[:-1], perm[-1:]
    perm = anchor_idx.copy()
    rng.shuffle(perm)
    val_count = max(1, int(round(K * val_ratio)))
    return perm[:-val_count], perm[-val_count:]


def eval_val_curve(coords: np.ndarray,
                   rel_coords: np.ndarray,
                   allow_reflection: bool,
                   K_min: int,
                   K_max: int,
                   seed: Optional[int] = None) -> List[Tuple[int, float]]:
    history: List[Tuple[int, float]] = []
    order = farthest_first_indices(rel_coords, K_max, seed=seed)
    for K in range(K_min, K_max + 1):
        anchor_idx = order[:K]
        train_idx, val_idx = train_val_split(anchor_idx, val_ratio=0.3, seed=seed)
        s, R, t = procrustes_general(rel_coords[train_idx], coords[train_idx], allow_reflection=allow_reflection)
        est_abs = apply_transform(rel_coords, s, R, t)
        history.append((K, rmse(est_abs[val_idx], coords[val_idx])))
    return history


def _mad(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return np.median(np.abs(x - med))


# ----- Graph geodesic distances (for graph_mds) -----

def all_pairs_shortest_paths(weights: np.ndarray) -> np.ndarray:
    n = weights.shape[0]
    D = weights.copy()
    np.fill_diagonal(D, 0.0)
    for k in range(n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D


def graph_geodesic_distances(probs: np.ndarray, alpha: float, tau: float) -> np.ndarray:
    n = probs.shape[0]
    inf = 1e9
    W = np.full((n, n), inf, dtype=float)
    mask = probs >= tau
    i_idx, j_idx = np.where(mask)
    if i_idx.size > 0:
        L = -np.log(np.clip(probs[mask], 1e-9, 1.0)) / alpha
        for i, j, w in zip(i_idx, j_idx, L):
            W[i, j] = w
    np.fill_diagonal(W, 0.0)
    D = all_pairs_shortest_paths(W)
    if not np.any(D < inf):
        D[:] = 0.0
    else:
        D[D >= inf * 0.5] = np.max(D[D < inf])
    return D


# ----- SMACOF iterative MDS (simple) -----

def smacof(dist: np.ndarray,
           init: Optional[np.ndarray] = None,
           n_components: int = 2,
           iters: int = 100,
           eps: float = 1e-6) -> np.ndarray:
    n = dist.shape[0]
    D = np.maximum(dist, 1e-9)
    if init is None:
        X = classical_mds(D, n_components=n_components)
    else:
        X = init.copy()
    for _ in range(iters):
        delta = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        delta = np.maximum(delta, 1e-12)
        B = -D / delta
        np.fill_diagonal(B, 0.0)
        B[np.diag_indices(n)] = -B.sum(axis=1)
        X_new = (1.0 / n) * B @ X
        if np.linalg.norm(X_new - X) < eps:
            X = X_new
            break
        X = X_new
    X -= X.mean(axis=0, keepdims=True)
    return X[:, :n_components]


# =========================
# Plot/frames helpers
# =========================

def save_window_frame(window: int,
                      coords: np.ndarray,
                      est_abs: np.ndarray,
                      anchor_idx: np.ndarray,
                      area_size: float,
                      rmse_all: float,
                      rmse_non: float,
                      K_curr: int,
                      energy_remaining: float,
                      out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7.2, 7.2))
    ax = plt.gca()

    ax.scatter(coords[:, 0], coords[:, 1], c='C0', s=36, label='Real')
    ax.scatter(est_abs[:, 0], est_abs[:, 1], c='C3', marker='x', s=48, label='Estimated')

    for i in range(coords.shape[0]):
        ax.plot([coords[i, 0], est_abs[i, 0]],
                [coords[i, 1], est_abs[i, 1]],
                color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

    ax.scatter(coords[anchor_idx, 0], coords[anchor_idx, 1],
               facecolors='none', edgecolors='black', s=140, linewidths=1.6, label='Antorchs')

    ax.set_xlim(0, area_size); ax.set_ylim(0, area_size)
    ax.set_aspect('equal', 'box'); ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='upper right')
    ax.set_title(f"Window {window} | K={K_curr} | RMSE(all)={rmse_all:.2f} | RMSE(non)={rmse_non:.2f} | E_rem={energy_remaining:.1f}")

    out_path = os.path.join(out_dir, f"frame_{window:04d}.png")
    plt.tight_layout(); plt.savefig(out_path, dpi=180); plt.close()


def create_gif_from_dir(frames_dir: str, output_gif: str, fps: float = 4.0, loop: int = 0) -> None:
    if imageio is None:
        print("[GIF] imageio not available; skipping GIF creation.")
        return
    import glob, re
    def _natkey(p):
        return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', os.path.basename(p))]
    paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")), key=_natkey)
    if not paths:
        print(f"[GIF] No frames found in {frames_dir}")
        return
    duration = 1.0 / max(fps, 0.01)
    frames = [imageio.imread(p) for p in paths]
    imageio.mimsave(output_gif, frames, duration=duration, loop=0)
    print(f"[GIF] Saved {output_gif} with {len(frames)} frames @ {fps} fps")


# =========================
# Main runner
# =========================

def run_timeseries(
    # method
    method: str, K_fixed: int, contact_threshold: float, smacof_iters: int, adaptive_rel_mode: str,
    # sim
    num_windows: int, num_nodes: int, area_size: float,
    alpha: float, noise_std: float, allow_reflection: bool,
    # adaptation (only for adaptive)
    K_init: int, K_min: int, K_max: int,
    target_low: float, target_high: float,
    patience_up: int, patience_down: int, step: int, cooldown_after_change: int,
    # dynamics & seed
    mobility_sigma: float, seed: Optional[int],
    # cost
    cost_per_anchor: float, lambda_cost: float,
    # energy (hard cap)
    energy_per_anchor_per_window: float, energy_budget_total: float,
    # spike guards
    spike_guard: bool, spike_window: int, spike_factor: float,
    spike_action: str, spike_blend_alpha: float,
    per_node_jump_guard: bool, max_node_jump: float,
    # IO
    run_dir: str, save_frames: bool, make_gif: bool, fps: float
) -> Dict[str, object]:

    rng = np.random.default_rng(seed)
    coords = rng.random((num_nodes, 2)) * area_size

    # Initialize K
    if method == "adaptive":
        K_curr = int(max(K_min, min(K_max, K_init)))
    else:
        K_curr = int(max(1, min(num_nodes, K_fixed)))  # fixed K for non-adaptive

    cooldown = 0
    energy_remaining = float(energy_budget_total)
    cost_cumulative = 0.0

    frames_dir = os.path.join(run_dir, "frames")
    os.makedirs(run_dir, exist_ok=True); os.makedirs(frames_dir, exist_ok=True)

    log_csv = os.path.join(run_dir, "adaptive_log.csv")
    val_csv = os.path.join(run_dir, "adaptive_val_history.csv")
    positions_csv = os.path.join(run_dir, "positions_windows.csv")

    rmse_hist = deque(maxlen=max(3, spike_window))
    prev_est_abs = None
    prev_transform = None

    with open(log_csv, "w", newline="") as f_log, \
         open(val_csv, "w", newline="") as f_val, \
         open(positions_csv, "w", newline="") as f_pos:
        wlog = csv.writer(f_log); wval = csv.writer(f_val); wpos = csv.writer(f_pos)

        wlog.writerow([
            "window","method","adaptive_rel_mode","K","K_fixed","k_cap_energy",
            "val_rmse","rmse_all_nodes","rmse_non_anchors",
            "objective_value",
            "energy_spent","energy_remaining",
            "cost_this_window","cost_cumulative",
            "target_low","target_high","decision",
            "spike_rejected","nodes_clamped"
        ])
        wval.writerow(["window","K","val_rmse"])
        wpos.writerow(["window","node","x_real","y_real","x_est","y_est","is_anchor"])

        for w in range(1, num_windows + 1):
            # mobility
            coords = coords + rng.normal(0, mobility_sigma, size=coords.shape)
            coords = np.clip(coords, 0, area_size)

            # contacts -> distances
            dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
            probs = np.exp(-alpha * dists)
            noisy_probs = np.clip(probs + rng.normal(0, noise_std, size=probs.shape), 1e-4, 1.0)
            est_dists = -np.log(noisy_probs) / alpha

            # relative map (by method)
            if method == "adaptive":
                rel_mode = adaptive_rel_mode
                if rel_mode == "classical":
                    rel_coords = classical_mds(est_dists)
                elif rel_mode == "GMDS":
                    Dg = graph_geodesic_distances(noisy_probs, alpha=alpha, tau=contact_threshold)
                    rel_coords = classical_mds(Dg)
                elif rel_mode == "SMACOF":
                    rel_coords = smacof(est_dists, init=None, n_components=2, iters=smacof_iters)
                else:
                    raise ValueError(rel_mode)
            elif method == "fixed":
                rel_coords = classical_mds(est_dists)
            elif method == "GMDS":
                Dg = graph_geodesic_distances(noisy_probs, alpha=alpha, tau=contact_threshold)
                rel_coords = classical_mds(Dg)
            elif method == "SMACOF":
                rel_coords = smacof(est_dists, init=None, n_components=2, iters=smacof_iters)
            else:
                raise ValueError(f"Unknown method {method}")

            # Energy feasibility for non-adaptive methods (K must stay constant)
            if method != "adaptive":
                if energy_per_anchor_per_window > 0:
                    required = K_curr * energy_per_anchor_per_window
                    if energy_remaining < required:
                        # Can't fund this window at fixed K -> stop the run (do NOT change K)
                        break
                decision = "fixed"
                val_hist = []
                val_dict = {}
            else:
                # --- Adaptive K selection (objective-driven + hysteresis) ---
                if energy_per_anchor_per_window > 0:
                    k_cap_energy = int(np.floor(energy_remaining / energy_per_anchor_per_window))
                else:
                    k_cap_energy = K_max
                k_feasible_max = max(K_min, min(K_max, k_cap_energy))

                decision = "hold"
                val_hist = eval_val_curve(coords, rel_coords, allow_reflection, K_min, K_max,
                                          seed=rng.integers(1, 1_000_000))
                val_dict = dict(val_hist)
                bestK = K_curr
                best_obj = float("inf")
                for Kc, v_rmse in val_hist:
                    if Kc < K_min or Kc > k_feasible_max:
                        continue
                    costK = cost_per_anchor * Kc
                    objK = float(v_rmse + lambda_cost * costK)
                    if objK < best_obj:
                        best_obj, bestK = objK, Kc

                if cooldown > 0:
                    cooldown -= 1
                else:
                    if bestK > K_curr and K_curr < k_feasible_max:
                        K_curr = min(k_feasible_max, K_curr + step)
                        decision = "increase"; cooldown = cooldown_after_change
                    elif bestK < K_curr and K_curr > K_min:
                        K_curr = max(K_min, K_curr - step)
                        decision = "decrease"; cooldown = cooldown_after_change
                    else:
                        decision = "hold"

            # Choose anchors (K is fixed for non-adaptive; may vary for adaptive)
            anchor_idx = spread_indices_from_prev(prev_est_abs, rel_coords, K_curr,
                                                  seed=rng.integers(1, 1_000_000))
            s, R, t = procrustes_general(rel_coords[anchor_idx], coords[anchor_idx],
                                         allow_reflection=allow_reflection)
            est_abs = apply_transform(rel_coords, s, R, t)

            # base metrics
            full_rmse = rmse(est_abs, coords)
            non_anchor_rmse = rmse_non_anchors(est_abs, coords, anchor_idx)

            # Spike guards
            spike_rejected = 0
            nodes_clamped = 0

            if spike_guard and len(rmse_hist) >= max(3, int(0.6*spike_window)):
                med = float(np.median(rmse_hist)); mad = float(_mad(rmse_hist))
                thresh = med + spike_factor * max(mad, 1e-6)
                if full_rmse > thresh:
                    spike_rejected = 1
                    if prev_est_abs is not None:
                        if spike_action == "keep_prev":
                            est_abs = prev_est_abs.copy()
                        elif spike_action == "reuse_prev_transform" and prev_transform is not None:
                            s_prev, R_prev, t_prev = prev_transform
                            est_alt = apply_transform(rel_coords, s_prev, R_prev, t_prev)
                            if rmse(est_alt, coords) <= full_rmse:
                                est_abs = est_alt
                        elif spike_action == "blend":
                            a = float(spike_blend_alpha)
                            est_abs = a*prev_est_abs + (1.0 - a)*est_abs
                        full_rmse = rmse(est_abs, coords)
                        non_anchor_rmse = rmse_non_anchors(est_abs, coords, anchor_idx)

            if per_node_jump_guard and prev_est_abs is not None:
                disp = np.linalg.norm(est_abs - prev_est_abs, axis=1)
                mask = disp > max_node_jump
                if np.any(mask):
                    nodes_clamped = int(np.count_nonzero(mask))
                    est_abs[mask] = prev_est_abs[mask]
                    full_rmse = rmse(est_abs, coords)
                    non_anchor_rmse = rmse_non_anchors(est_abs, coords, anchor_idx)

            # objective & energy/cost for logging
            cost_this_window = float(max(cost_per_anchor, 0.0) * K_curr)
            objective_value = float(full_rmse + lambda_cost * cost_this_window)
            energy_spent = float(K_curr * energy_per_anchor_per_window)
            energy_remaining = max(0.0, energy_remaining - energy_spent)

            # logs
            k_cap_energy = (int(np.floor(energy_remaining / energy_per_anchor_per_window))
                            if energy_per_anchor_per_window > 0 else K_curr)
            val_rmse_log = val_dict.get(K_curr, float("nan")) if method == "adaptive" else ""
            wlog.writerow([
                w, method, (adaptive_rel_mode if method == "adaptive" else ""),
                K_curr, (K_fixed if method != "adaptive" else ""), k_cap_energy,
                val_rmse_log, full_rmse, non_anchor_rmse,
                objective_value,
                energy_spent, energy_remaining,
                cost_this_window, cost_cumulative + cost_this_window,
                target_low, target_high, decision,
                spike_rejected, nodes_clamped
            ])
            if method == "adaptive":
                for K, v in val_hist:
                    wval.writerow([w, K, v])

            is_anchor_mask = np.zeros(coords.shape[0], dtype=bool)
            is_anchor_mask[anchor_idx] = True
            for i in range(coords.shape[0]):
                wpos.writerow([
                    w, i,
                    f"{coords[i,0]:.6f}", f"{coords[i,1]:.6f}",
                    f"{est_abs[i,0]:.6f}", f"{est_abs[i,1]:.6f}",
                    int(is_anchor_mask[i])
                ])

            if save_frames:
                save_window_frame(
                    window=w, coords=coords, est_abs=est_abs, anchor_idx=anchor_idx,
                    area_size=area_size, rmse_all=full_rmse, rmse_non=non_anchor_rmse,
                    K_curr=K_curr, energy_remaining=energy_remaining,
                    out_dir=frames_dir
                )

            # update for next loop
            cost_cumulative += cost_this_window
            rmse_hist.append(full_rmse)
            prev_est_abs = est_abs.copy()
            prev_transform = (s, R, t)

            # For adaptive: if budget is exhausted next step, loop will break via cap
            # For fixed/smacof/graph_mds: we already break *before* the window if budget insufficient

            if method == "adaptive":
                # If no energy remains, stop (can't fund any anchors next window)
                if energy_per_anchor_per_window > 0 and energy_remaining < (K_min * energy_per_anchor_per_window):
                    break

    # Time-series plots from log
    def _col(path, col, cast=float):
        out = []
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    out.append(cast(row[col]))
                except Exception:
                    out.append(row[col])
        return out

    W = _col(log_csv, "window", int)
    if len(W) == 0:
        return {}

    K_arr = _col(log_csv, "K", int)
    RMSE_all = _col(log_csv, "rmse_all_nodes", float)
    E_spent = _col(log_csv, "energy_spent", float)
    E_left = _col(log_csv, "energy_remaining", float)
    C_win = _col(log_csv, "cost_this_window", float)
    C_cum = _col(log_csv, "cost_cumulative", float)
    OBJ = _col(log_csv, "objective_value", float)

    # RMSE vs K
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Window"); ax1.set_ylabel("RMSE (m)", color="tab:blue")
    ax1.plot(W, RMSE_all, marker='o', label="RMSE (all nodes)", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue"); ax1.grid(True, linestyle='--', alpha=0.5)
    ax2 = ax1.twinx(); ax2.set_ylabel("Anchors (K)", color="tab:red")
    ax2.plot(W, K_arr, marker='s', label="Anchors (K)", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    fig.tight_layout(); plt.title("RMSE vs Number of Anchors over Time")
    plt.savefig(os.path.join(run_dir, "rmse_vs_anchors.png"), dpi=300); plt.close()

    # Energy
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(W, E_spent, alpha=0.6, label="Energy spent (per window)")
    ax.set_xlabel("Window"); ax.set_ylabel("Energy spent (units)")
    ax2 = ax.twinx(); ax2.plot(W, E_left, marker='o', label="Energy remaining", linestyle='-')
    ax2.set_ylabel("Energy remaining (units)")
    ax.grid(True, linestyle='--', alpha=0.5); fig.tight_layout()
    plt.title("Energy Budget Over Time")
    plt.savefig(os.path.join(run_dir, "energy_over_time.png"), dpi=300); plt.close()

    # Cost
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(W, C_win, alpha=0.6, label="Cost (per window)")
    ax.set_xlabel("Window"); ax.set_ylabel("Cost (per window)")
    ax2 = ax.twinx(); ax2.plot(W, C_cum, marker='o', label="Cumulative cost", linestyle='-')
    ax2.set_ylabel("Cumulative cost"); ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout(); plt.title("Cost Over Time")
    plt.savefig(os.path.join(run_dir, "cost_over_time.png"), dpi=300); plt.close()

    # RMSE and Objective
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(W, RMSE_all, marker='o', label="RMSE (all nodes)")
    ax.plot(W, OBJ, marker='s', label="Objective = RMSE + λ·cost")
    ax.set_xlabel("Window"); ax.set_ylabel("Value"); ax.set_title("RMSE and Objective over time")
    ax.grid(True); ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "rmse_vs_objective.png"), dpi=300); plt.close()

    # Optional GIF
    if make_gif:
        create_gif_from_dir(frames_dir=frames_dir,
                            output_gif=os.path.join(run_dir, "simulation.gif"),
                            fps=fps, loop=0)

    # ---- Summary CSV ----
    with open(log_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    import statistics as stats

    def _floats(col):
        return [float(r[col]) for r in rows]

    rmse_all = _floats("rmse_all_nodes")
    rmse_non = _floats("rmse_non_anchors")
    K_vals   = [int(r["K"]) for r in rows]
    k_cap    = [int(r["k_cap_energy"]) for r in rows]
    obj_vals = _floats("objective_value")
    e_spent  = _floats("energy_spent")
    e_left   = _floats("energy_remaining")
    spikes   = [int(r["spike_rejected"]) for r in rows]
    clamps   = [int(r["nodes_clamped"]) for r in rows]

    def _avg(x): return float(sum(x)/len(x)) if x else float("nan")
    def _std(x): return float(stats.pstdev(x)) if len(x) > 1 else 0.0

    K_max_obs = max(K_vals) if K_vals else 0
    def _pct_cap_limited():
        count = 0
        for kv, kc in zip(K_vals, k_cap):
            if kc < K_max_obs and kv == kc:
                count += 1
        return 100.0 * count / len(K_vals)

    run_name = os.path.basename(run_dir)
    summary = {
        "run_name": run_name,
        "method": method,
        "adaptive_rel_mode": (adaptive_rel_mode if method == "adaptive" else ""),
        "windows_run": len(rows),
        "avg_rmse_all": _avg(rmse_all),
        "std_rmse_all": _std(rmse_all),
        "p95_rmse_all": float(np.percentile(rmse_all, 95)) if rmse_all else float("nan"),
        "avg_rmse_non": _avg(rmse_non),
        "std_rmse_non": _std(rmse_non),
        "p95_rmse_non": float(np.percentile(rmse_non, 95)) if rmse_non else float("nan"),
        "avg_K": _avg(K_vals),
        "min_K": min(K_vals) if K_vals else 0,
        "max_K": max(K_vals) if K_vals else 0,
        "pct_cap_limited": _pct_cap_limited(),
        "avg_objective": _avg(obj_vals),
        "total_energy_spent": float(sum(e_spent)),
        "final_energy_remaining": e_left[-1] if e_left else float("nan"),
        "spike_rejections": int(sum(spikes)),
        "nodes_clamped_total": int(sum(clamps)),
        "K_fixed": (K_fixed if method != "adaptive" else ""),
        "contact_threshold": (contact_threshold if method == "graph_mds" else ""),
        "smacof_iters": (smacof_iters if method == "smacof" else "")
    }

    with open(os.path.join(run_dir, "summary.csv"), "w", newline="") as fsum:
        w = csv.writer(fsum)
        w.writerow(list(summary.keys()))
        w.writerow([summary[k] for k in summary.keys()])

    return summary


# =========================
# CLI
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="Contact-based indoor localization (CLI).")
    # method
    p.add_argument("--method", type=str, default="adaptive",
                   choices=["adaptive","fixed","GMDS","SMACOF"])
    p.add_argument("--K_fixed", type=int, default=6)
    p.add_argument("--contact_threshold", type=float, default=0.5)
    p.add_argument("--smacof_iters", type=int, default=100)
    p.add_argument("--adaptive_rel_mode", type=str, default="classical",
                   choices=["classical","GMDS","SMACOF"],
                   help="Relative-map method to use when method=adaptive.")

    # run/output
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--save_frames", action="store_true")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--fps", type=float, default=4.0)

    # sim
    p.add_argument("--num_windows", type=int, default=30)
    p.add_argument("--num_nodes", type=int, default=20)
    p.add_argument("--area_size", type=float, default=20.0)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--noise_std", type=float, default=0.02)
    p.add_argument("--allow_reflection", action="store_true")

    # adaptation (used by adaptive)
    p.add_argument("--K_init", type=int, default=2)
    p.add_argument("--K_min", type=int, default=2)
    p.add_argument("--K_max", type=int, default=20)
    p.add_argument("--target_low", type=float, default=5.0)   # retained for logging/compat
    p.add_argument("--target_high", type=float, default=6.0)  # retained for logging/compat
    p.add_argument("--patience_up", type=int, default=1)      # retained for compat
    p.add_argument("--patience_down", type=int, default=2)    # retained for compat
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--cooldown_after_change", type=int, default=1)

    # dynamics & seed
    p.add_argument("--mobility_sigma", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1234)

    # cost
    p.add_argument("--cost_per_anchor", type=float, default=0.5)
    p.add_argument("--lambda_cost", type=float, default=0.2)

    # energy
    p.add_argument("--energy_per_anchor_per_window", type=float, default=1.0)
    p.add_argument("--energy_budget_total", type=float, default=200.0)

    # spike guards
    p.add_argument("--spike_guard", action="store_true")
    p.add_argument("--spike_window", type=int, default=5)
    p.add_argument("--spike_factor", type=float, default=3.0)
    p.add_argument("--spike_action", type=str, default="keep_prev",
                   choices=["keep_prev","reuse_prev_transform","blend"])
    p.add_argument("--spike_blend_alpha", type=float, default=0.2)
    p.add_argument("--per_node_jump_guard", action="store_true")
    p.add_argument("--max_node_jump", type=float, default=3.0)

    return p.parse_args()


def main():
    args = parse_args()
    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    summary = run_timeseries(
        # method
        method=args.method, K_fixed=args.K_fixed,
        contact_threshold=args.contact_threshold, smacof_iters=args.smacof_iters,
        adaptive_rel_mode=args.adaptive_rel_mode,
        # sim
        num_windows=args.num_windows, num_nodes=args.num_nodes, area_size=args.area_size,
        alpha=args.alpha, noise_std=args.noise_std, allow_reflection=args.allow_reflection,
        # adaptation
        K_init=args.K_init, K_min=args.K_min, K_max=args.K_max,
        target_low=args.target_low, target_high=args.target_high,
        patience_up=args.patience_up, patience_down=args.patience_down,
        step=args.step, cooldown_after_change=args.cooldown_after_change,
        # dynamics & seed
        mobility_sigma=args.mobility_sigma, seed=args.seed,
        # cost & energy
        cost_per_anchor=args.cost_per_anchor, lambda_cost=args.lambda_cost,
        energy_per_anchor_per_window=args.energy_per_anchor_per_window,
        energy_budget_total=args.energy_budget_total,
        # guards
        spike_guard=args.spike_guard, spike_window=args.spike_window,
        spike_factor=args.spike_factor, spike_action=args.spike_action,
        spike_blend_alpha=args.spike_blend_alpha,
        per_node_jump_guard=args.per_node_jump_guard, max_node_jump=args.max_node_jump,
        # IO
        run_dir=run_dir, save_frames=args.save_frames, make_gif=args.gif, fps=args.fps
    )

    print(f"[DONE] Outputs saved under: {run_dir}")
    if summary:
        print("[SUMMARY]")
        for k, v in summary.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
