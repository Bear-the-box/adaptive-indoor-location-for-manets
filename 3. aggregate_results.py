#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate all run summaries into one dataset and plot comparative charts,
highlighting proposed strategies (adaptive variants) with custom colors.

Inputs
------
Expects simulator outputs under:
    runs/<run_name>/summary.csv

Outputs
-------
- datasets/summary_all.csv
- plots/rmse_by_method.png
- plots/avg_rmse_all_by_method_LD.png
- plots/avg_rmse_all_by_method_HD.png
- plots/energy_per_window_by_method.png
- plots/energy_per_window_by_method_LD.png       [NEW]
- plots/energy_per_window_by_method_HD.png       [NEW]
- plots/objective_by_method.png                  (optional; --plot_objective)
- plots/pareto_rmse_vs_energy.png
- plots/combined_score_by_method.png
"""

import argparse
import csv
import glob
import math
import os
from collections import defaultdict
from statistics import mean, stdev

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Aggregate simulation summaries and plot comparisons.")
    p.add_argument("--runs_root", type=str, default="runs", help="Folder containing run subfolders.")
    p.add_argument("--out_csv", type=str, default=os.path.join("datasets", "summary_all.csv"),
                   help="Path to save the combined dataset CSV.")
    p.add_argument("--out_dir", type=str, default="plots", help="Folder to save plots.")
    p.add_argument("--min_runs_per_group", type=int, default=1,
                   help="Minimum runs required to include a group in charts.")
    p.add_argument("--plot_objective", action="store_true",
                   help="Also plot Objective (RMSE + λ·cost) by method.")
    # Weights for the combined score
    p.add_argument("--w_precision", type=float, default=0.5,
                   help="Weight for precision (1-RMSE_norm) in combined score.")
    p.add_argument("--w_energy", type=float, default=0.5,
                   help="Weight for energy saving (1-Energy_norm) in combined score.")
    return p.parse_args()


# ----------------------------
# IO helpers
# ----------------------------
def ensure_dir_for_file(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_summary_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
        if not rows:
            return None
        return rows[0]  # single-row summary


# ----------------------------
# Run-name parsing
# ----------------------------
def parse_run_name(run_name: str):
    parts = run_name.split("-")
    meta = {
        "run_name": run_name,
        "density": "",
        "mobility": "",
        "noise": "",
        "energy": "",
        "seed": "",
        "method_tag": ""
    }
    if len(parts) >= 2:
        for p in parts:
            if p in ("HD", "LD"): meta["density"] = p
            if p.startswith("M") and len(p) == 2: meta["mobility"] = p[1:]
            if p.startswith("N") and len(p) == 2: meta["noise"] = p[1:]
            if p.startswith("E") and len(p) == 2: meta["energy"] = p[1:]
            if p.startswith("S"): meta["seed"] = p[1:]
        # method tag candidates (e.g., ADAPT-GMDS, FIXK6, GMDS, SMACOF)
        candidates = []
        for i in range(len(parts)):
            for j in range(i+1, min(i+3, len(parts))+1):
                token = "-".join(parts[i:j])
                if ("ADAPT" in token) or token.startswith("FIXK") or token in ("GMDS", "SMACOF"):
                    candidates.append(token)
        if candidates:
            meta["method_tag"] = sorted(candidates, key=len, reverse=True)[0]
    return meta


def collect_runs(runs_root: str):
    pattern = os.path.join(runs_root, "*", "summary.csv")
    rows = []
    for spath in sorted(glob.glob(pattern)):
        row = read_summary_csv(spath)
        if row is None:
            continue
        run_name = os.path.basename(os.path.dirname(spath))
        meta = parse_run_name(run_name)
        rows.append({"run_name": run_name, **meta, **row})
    return rows


def write_combined_csv(rows, out_csv):
    ensure_dir_for_file(out_csv)
    headers = set()
    for r in rows: headers.update(r.keys())
    preferred = [
        "run_name","method","method_tag","adaptive_rel_mode",
        "density","mobility","noise","energy","seed",
        "windows_run",
        "avg_rmse_all","std_rmse_all","p95_rmse_all",
        "avg_rmse_non","std_rmse_non","p95_rmse_non",
        "avg_objective",
        "avg_K","min_K","max_K","pct_cap_limited",
        "total_energy_spent","final_energy_remaining",
        "spike_rejections","nodes_clamped_total",
        "K_fixed","contact_threshold","smacof_iters"
    ]
    ordered = [h for h in preferred if h in headers] + [h for h in headers if h not in preferred]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in rows: w.writerow(r)
    return ordered


# ----------------------------
# Grouping & labels
# ----------------------------
def method_label(row):
    """Split adaptive variants into separate labels: e.g., adaptive[graph_mds]."""
    m = row.get("method", "") or ""
    if m == "adaptive":
        arm = (row.get("adaptive_rel_mode", "") or "classical")
        return f"adaptive[{arm}]"
    mt = row.get("method_tag", "") or ""
    return mt if mt else m


# --- Color mapping for bars/scatter (highlight proposed strategies) ---
PROPOSED_COLORS = {
    "adaptive[SMACOF]":    "#ff7f00",
    "adaptive[GMDS]": "#ff7f00",
    "adaptive[classical]": "#ff7f00",
}
BASELINE_COLORS = {"GMDS": "#999999", "SMACOF": "#999999"}
COLOR_FIXED = "#999999"
COLOR_DEFAULT = "#999999"

def color_for_label(label: str) -> str:
    if label in PROPOSED_COLORS:
        return PROPOSED_COLORS[label]
    if label in BASELINE_COLORS:
        return BASELINE_COLORS[label]
    if label.startswith("FIXK"):
        return COLOR_FIXED
    return COLOR_DEFAULT


# ----------------------------
# Stats helpers
# ----------------------------
def group_stats(rows, value_extractor):
    """Return list of (label, n, mean, sd). Ignores NaNs."""
    buckets = defaultdict(list)
    for r in rows:
        label = method_label(r)
        val = value_extractor(r)
        if val is None:
            continue
        try:
            v = float(val)
            if math.isnan(v):
                continue
        except Exception:
            continue
        buckets[label].append(v)
    stats = []
    for lab, vals in buckets.items():
        if not vals:
            continue
        mu = mean(vals)
        sd = stdev(vals) if len(vals) > 1 else 0.0
        n = len(vals)
        stats.append((lab, n, mu, sd))
    stats.sort(key=lambda x: x[2])
    return stats


# ----------------------------
# Plot helpers
# ----------------------------
def _auto_figsize(n_bars, base_w=8.0, per_bar=0.6, h=4.8):
    return (max(base_w, per_bar * n_bars + 2.0), h)


def _bar_with_sem(ax, labels, means, sds, ns, title, ylabel, with_values=False, fmt="{:.2f}"):
    sems = [(sd / math.sqrt(n)) if n > 0 else 0.0 for sd, n in zip(sds, ns)]
    x = np.arange(len(labels))
    width = 0.65
    colors = [color_for_label(lab) for lab in labels]
    bars = ax.bar(x, means, yerr=sems, capsize=5, width=width, color=colors, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    if with_values:
        try:
            ax.bar_label(bars, labels=[fmt.format(m) for m in means], padding=3, fontsize=9)
        except Exception:
            for rect, m in zip(bars, means):
                ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.01,
                        fmt.format(m), ha='center', va='bottom', fontsize=9)

    from matplotlib.patches import Patch
    legend_handles = []
    used = set()
    for k, col in PROPOSED_COLORS.items():
        if k in labels and col not in used:
            legend_handles.append(Patch(facecolor=col, label=f"Adaptive strategies"))
            used.add(col)
    base_used = set()
    #for k, col in BASELINE_COLORS.items():
    #    if (k in labels) and (col not in base_used):
    #        legend_handles.append(Patch(facecolor=col, label=f"Baseline strategies"))
    #        base_used.add(col)
    if any(lab.startswith("FIXK") for lab in labels):
        legend_handles.append(Patch(facecolor=COLOR_FIXED, label="Baseline strategies"))
    if any((lab not in PROPOSED_COLORS and lab not in BASELINE_COLORS and not lab.startswith("FIXK")) for lab in labels):
        legend_handles.append(Patch(facecolor=COLOR_DEFAULT, label="Baseline: other"))

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=9, frameon=False, loc="best")


def plot_metric_by_method(rows, value_extractor, title: str, ylabel: str, out_path: str, with_values=False):
    stats = group_stats(rows, value_extractor)
    if not stats:
        print(f"[WARN] No data for {title}; skipping {out_path}")
        return
    labels = [s[0] for s in stats]
    ns     = [s[1] for s in stats]
    means  = [s[2] for s in stats]
    sds    = [s[3] for s in stats]
    ensure_dir_for_file(out_path)
    fig, ax = plt.subplots(figsize=_auto_figsize(len(labels)))
    _bar_with_sem(ax, labels, means, sds, ns, title, ylabel, with_values=with_values)
    fig.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()
    print(f"[PLOT] Saved {out_path}")


def plot_metric_by_method_per_density(rows, value_extractor, metric_name: str, ylabel: str, out_dir: str, min_runs=1):
    ensure_dir(out_dir)
    buckets = defaultdict(list)
    for r in rows:
        dkey = r.get("density", "")
        if not dkey: continue
        buckets[dkey].append(r)

    for dkey, subset in buckets.items():
        stats = group_stats(subset, value_extractor)
        stats = [s for s in stats if s[1] >= min_runs]
        if not stats: continue
        labels = [s[0] for s in stats]
        ns     = [s[1] for s in stats]
        means  = [s[2] for s in stats]
        sds    = [s[3] for s in stats]
        fig, ax = plt.subplots(figsize=_auto_figsize(len(labels)))
        #title = f"{metric_name} by Method (Density={dkey})"
        title = f""
        _bar_with_sem(ax, labels, means, sds, ns, title, ylabel)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{metric_name}_by_method_{dkey}.png")
        plt.savefig(out_path, dpi=300); plt.close()
        print(f"[PLOT] Saved {out_path}")


# ----------------------------
# NEW: Pareto & Combined Score
# ----------------------------
def compute_means_for_methods(rows):
    """Return dict label -> dict with means of rmse, energy_per_window, objective, avg_K, n."""
    buckets = defaultdict(list)
    for r in rows:
        label = method_label(r)
        buckets[label].append(r)

    out = {}
    for lab, items in buckets.items():
        rmses = []
        energies = []
        objs = []
        ks = []
        for r in items:
            try:
                rm = float(r.get("avg_rmse_all", "nan"))
                en = float(r.get("total_energy_spent", "nan"))
                wn = float(r.get("windows_run", "nan"))
                ob = float(r.get("avg_objective", "nan"))
                k  = float(r.get("avg_K", "nan"))
                if math.isfinite(rm): rmses.append(rm)
                if math.isfinite(en) and math.isfinite(wn) and wn > 0: energies.append(en / wn)
                if math.isfinite(ob): objs.append(ob)
                if math.isfinite(k): ks.append(k)
            except Exception:
                pass
        if not rmses or not energies:
            continue
        out[lab] = {
            "n": len(items),
            "rmse": mean(rmses),
            "energy_per_window": mean(energies),
            "objective": mean(objs) if objs else float("nan"),
            "avg_K": mean(ks) if ks else float("nan")
        }
    return out


def plot_pareto_rmse_energy(stats_dict, out_path):
    """Scatter: X=RMSE (lower better), Y=Energy/window (lower better)."""
    if not stats_dict:
        print("[WARN] No data for Pareto plot.")
        return
    labels = list(stats_dict.keys())
    X = np.array([stats_dict[l]["rmse"] for l in labels])
    Y = np.array([stats_dict[l]["energy_per_window"] for l in labels])

    ensure_dir_for_file(out_path)
    fig, ax = plt.subplots(figsize=(8.7, 6))
    colors = [color_for_label(lab) for lab in labels]
    ax.scatter(X, Y, s=90, c=colors)
    for x, y, lab in zip(X, Y, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Average RMSE (m)")
    ax.set_ylabel("Average Jules per Window (J/w)")
    #ax.set_title("Pareto: Accuracy vs Energy")
    ax.set_title("")
    ax.grid(True, linestyle='--', alpha=0.35)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    print(f"[PLOT] Saved {out_path}")


def plot_combined_score(stats_dict, w_precision, w_energy, out_path):
    """
    Score = w_precision * (1 - rmse_norm) + w_energy * (1 - energy_norm)
    where *_norm are min-max normalized across methods.
    """
    if not stats_dict:
        print("[WARN] No data for combined score.")
        return
    labels = list(stats_dict.keys())
    rmses  = np.array([stats_dict[l]["rmse"] for l in labels], dtype=float)
    eners  = np.array([stats_dict[l]["energy_per_window"] for l in labels], dtype=float)

    def mm_norm(x):
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if xmax - xmin < 1e-12:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    rn = mm_norm(rmses)
    en = mm_norm(eners)
    scores = w_precision * (1.0 - rn) + w_energy * (1.0 - en)

    order = np.argsort(scores)[::-1]
    labels = [labels[i] for i in order]
    scores = [float(scores[i]) for i in order]
    colors = [color_for_label(lab) for lab in labels]

    ensure_dir_for_file(out_path)
    fig, ax = plt.subplots(figsize=_auto_figsize(len(labels), per_bar=0.55))
    bars = ax.bar(np.arange(len(labels)), scores, width=0.65, color=colors, edgecolor="none")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel(f"Combined Score (w_p={w_precision:.2f}, w_e={w_energy:.2f})")
    ax.set_title("Combined Score: Accuracy & Energy (min–max normalized)")
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    try:
        ax.bar_label(bars, labels=[f"{s:.3f}" for s in scores], padding=3, fontsize=9)
    except Exception:
        for rect, s in zip(bars, scores):
            ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.01,
                    f"{s:.3f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    print(f"[PLOT] Saved {out_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    rows = collect_runs(args.runs_root)
    # If you need to drop a method globally (e.g., FIXK3), uncomment:
    # rows = [r for r in rows if not method_label(r).startswith("FIXK3")]

    if not rows:
        print(f"[WARN] No summary files found under {args.runs_root}")
        return

    # Dataset
    ensure_dir_for_file(args.out_csv)
    write_combined_csv(rows, args.out_csv)
    print(f"[DATA] Wrote combined dataset: {args.out_csv}  ({len(rows)} rows)")

    # RMSE (overall)
    plot_metric_by_method(
        rows=rows,
        value_extractor=lambda r: float(r.get("avg_rmse_all", "nan")),
        #title="Comparative RMSE by Method (mean ± SEM across runs)",
        title="",
        ylabel="Average RMSE (m)",
        out_path=os.path.join(args.out_dir, "rmse_by_method.png"),
        with_values=True
    )

    # RMSE per density (HD/LD)
    plot_metric_by_method_per_density(
        rows=rows,
        value_extractor=lambda r: float(r.get("avg_rmse_all", "nan")),
        metric_name="avg_rmse_all",
        ylabel="Average RMSE (m)",
        out_dir=args.out_dir,
        min_runs=args.min_runs_per_group
    )

    # Energy per window (overall)
    def energy_per_window(r):
        try:
            e = float(r.get("total_energy_spent", "nan"))
            w = float(r.get("windows_run", "nan"))
            if not (math.isfinite(e) and math.isfinite(w)) or w <= 0:
                return None
            return e / w
        except Exception:
            return None

    plot_metric_by_method(
        rows=rows,
        value_extractor=energy_per_window,
        #title="Average J/w by Method (mean ± SEM across runs)",
        title="",
        ylabel="Jules / window",
        out_path=os.path.join(args.out_dir, "energy_per_window_by_method.png")
    )

    # --- NEW: Energy per window per density (HD/LD) ---
    plot_metric_by_method_per_density(
        rows=rows,
        value_extractor=energy_per_window,
        metric_name="energy_per_window",
        ylabel="Jules / window",
        out_dir=args.out_dir,
        min_runs=args.min_runs_per_group
    )

    # Objective (optional)
    if args.plot_objective:
        plot_metric_by_method(
            rows=rows,
            value_extractor=lambda r: float(r.get("avg_objective", "nan")),
            #title="Comparative Objective by Method (mean ± SEM across runs)",
            title="",
            ylabel="Average Objective (RMSE + λ·cost)",
            out_path=os.path.join(args.out_dir, "objective_by_method.png")
        )

    # Pareto (RMSE vs Energy)
    stats_dict = compute_means_for_methods(rows)
    plot_pareto_rmse_energy(stats_dict, os.path.join(args.out_dir, "pareto_rmse_vs_energy.png"))

    # Combined score
    plot_combined_score(stats_dict, args.w_precision, args.w_energy,
                        os.path.join(args.out_dir, "combined_score_by_method.png"))


if __name__ == "__main__":
    main()
