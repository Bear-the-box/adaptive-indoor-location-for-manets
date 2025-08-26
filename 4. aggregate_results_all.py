#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate all run summaries into one dataset and plot comparative charts,
highlighting proposed strategies (adaptive variants) with custom colors.

Extra: also generate per-window time-series plots of RMSE and K for each method
in Low Density and High Density scenarios, with average lines included.
"""

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

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
        return rows if rows else []

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
        # method tag candidates
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
        raw_rows = read_summary_csv(spath)
        if not raw_rows: 
            continue
        run_name = os.path.basename(os.path.dirname(spath))
        meta = parse_run_name(run_name)
        for r in raw_rows:
            rows.append({"run_name": run_name, **meta, **r})
    return rows

def write_combined_csv(rows, out_csv):
    ensure_dir_for_file(out_csv)
    headers = set()
    for r in rows: headers.update(r.keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(headers))
        w.writeheader()
        for r in rows: w.writerow(r)

# ----------------------------
# Grouping & labels
# ----------------------------
def method_label(row):
    m = row.get("method", "") or ""
    if m == "adaptive":
        arm = (row.get("adaptive_rel_mode", "") or "classical")
        return f"adaptive[{arm}]"
    mt = row.get("method_tag", "") or ""
    return mt if mt else m

# ----------------------------
# New: per-window time-series
# ----------------------------
def plot_rmse_and_K_timeseries(rows, out_dir):
    """
    For each method × density, plot RMSE vs windows (left y-axis) and K vs windows (right y-axis).
    Include average RMSE and average K as dashed lines.
    """
    ensure_dir(out_dir)
    # Group by (method, density)
    grouped = defaultdict(list)
    for r in rows:
        label = method_label(r)
        density = r.get("density", "")
        if not density: continue
        grouped[(label, density)].append(r)

    for (label, density), items in grouped.items():
        # Sort by window index if available
        try:
            xs = [int(r.get("window", i)) for i, r in enumerate(items)]
        except Exception:
            xs = list(range(len(items)))
        rmses = [float(r.get("rmse_window", r.get("avg_rmse_all", "nan"))) for r in items]
        ks    = [float(r.get("K_window", r.get("avg_K", "nan"))) for r in items]

        # Filter non-finite
        xs, rmses, ks = zip(*[(x, rm, k) for x, rm, k in zip(xs, rmses, ks)
                              if math.isfinite(rm) and math.isfinite(k)])

        avg_rmse = np.mean(rmses)
        avg_k = np.mean(ks)

        fig, ax1 = plt.subplots(figsize=(8, 4.8))
        color_rmse = "tab:blue"
        ax1.set_xlabel("Window")
        ax1.set_ylabel("RMSE (m)", color=color_rmse)
        l1, = ax1.plot(xs, rmses, color=color_rmse, marker="o", label="RMSE per window")
        l2 = ax1.axhline(avg_rmse, color=color_rmse, linestyle="--", label=f"Avg RMSE = {avg_rmse:.2f}")
        ax1.tick_params(axis="y", labelcolor=color_rmse)
        ax1.grid(True, linestyle="--", alpha=0.35)

        ax2 = ax1.twinx()
        color_k = "tab:red"
        ax2.set_ylabel("Number of Anchors (K)", color=color_k)
        l3, = ax2.plot(xs, ks, color=color_k, marker="s", label="K per window")
        l4 = ax2.axhline(avg_k, color=color_k, linestyle="--", label=f"Avg K = {avg_k:.1f}")
        ax2.tick_params(axis="y", labelcolor=color_k)

        # Legend (combine both axes handles)
        lines = [l1, l2, l3, l4]
        labels = [ln.get_label() for ln in lines]
        fig.legend(lines, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.15))

        plt.title(f"{label} – {density} scenario")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = f"timeseries_rmse_K_{label}_{density}.png".replace("/", "_")
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=300); plt.close()
        print(f"[PLOT] Saved {out_path}")

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    rows = collect_runs(args.runs_root)
    if not rows:
        print(f"[WARN] No summary files found under {args.runs_root}")
        return

    write_combined_csv(rows, args.out_csv)
    print(f"[DATA] Wrote combined dataset: {args.out_csv} ({len(rows)} rows)")

    # Generate the new per-window time-series plots
    plot_rmse_and_K_timeseries(rows, args.out_dir)

if __name__ == "__main__":
    main()
