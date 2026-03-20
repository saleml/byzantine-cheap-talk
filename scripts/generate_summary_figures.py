#!/usr/bin/env python3
"""
Generate three publication-ready figures for the Byzantine / Soft Byzantine
experiments.  LNCS-appropriate sizing, clean style, no gridlines.

Usage:
  python scripts/generate_summary_figures.py --version v1
  python scripts/generate_summary_figures.py --version v3

Outputs (saved to results/figures_{version}/):
  1. byzantine_learning_curves.pdf   -  Round-by-round honest cooperation rate
  2. payoff_gap_bars.pdf             -  FD vs PC avg payoff/round across conditions
  3. results_summary_table.pdf       -  Full results summary as a table figure
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))

# FD/PC are derived from data at runtime (set in main)
FD_FAMILIES = set()
PC_FAMILIES = set()


# =====================================================================
# helpers (importable without side effects)
# =====================================================================
def load_csv(path):
    """Return list of dicts from a CSV file."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def valid_rows(rows):
    """Filter out rows where action parsing failed (action is empty/None)."""
    return [r for r in rows if r.get("action") and r["action"] not in ("", "None")]


def honest_rows(rows):
    """Filter to honest (non-adversary), valid rows."""
    return [r for r in rows
            if r["is_adversary"] in ("False", "0", False)
            and r.get("action") and r["action"] not in ("", "None")]


def coop_rate(rows):
    """Fraction of rows where action == 'Hunt Stag' (individual-level)."""
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["action"] == "Hunt Stag") / len(rows)


def group_coop_rate(rows):
    """Fraction of (trial, round) groups where ALL agents chose Hunt Stag."""
    if not rows:
        return 0.0
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["trial_id"], r["round"])].append(r)
    if not groups:
        return 0.0
    all_stag = sum(1 for agents in groups.values()
                   if all(a["action"] == "Hunt Stag" for a in agents))
    return all_stag / len(groups)


def avg_payoff_per_round(rows):
    """Average payoff across rows."""
    if not rows:
        return 0.0
    return sum(float(r["payoff"]) for r in rows) / len(rows)


# =====================================================================
# figure generation (only when run as a script)
# =====================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from analyze_results import classify_families_from_data

    parser = argparse.ArgumentParser(description="Generate summary figures")
    parser.add_argument("--version", type=str, required=True, choices=["v1", "v2", "v3"])
    args = parser.parse_args()
    version = args.version

    # ---------- style ----------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 300,
    })

    FIG_W, FIG_H = 3.5, 2.6

    FIGURES = RESULTS / f"figures_{version}"
    FIGURES.mkdir(parents=True, exist_ok=True)

    # load data
    byz_csv = RESULTS / f"byzantine_{version}" / "all_results.csv"
    soft_csv = RESULTS / f"byzantine_soft_{version}" / "all_results.csv"
    topo_csv = RESULTS / f"topology_{version}" / "all_results.csv"
    topo_silent_csv = RESULTS / f"topology_silent_{version}" / "all_results.csv"

    byz_data = load_csv(byz_csv)
    soft_data = load_csv(soft_csv) if soft_csv.exists() else []
    topo_data = load_csv(topo_csv) if topo_csv.exists() else []
    topo_silent_data = load_csv(topo_silent_csv) if topo_silent_csv.exists() else []

    # Derive FD/PC from data
    FD_FAMILIES, PC_FAMILIES = classify_families_from_data(byz_data)
    fd_label = " + ".join(sorted(FD_FAMILIES)) if FD_FAMILIES else "FD"
    pc_label = " + ".join(sorted(PC_FAMILIES)) if PC_FAMILIES else "PC"
    print(f"Version: {version}")
    print(f"FD families (derived): {sorted(FD_FAMILIES)}")
    print(f"PC families (derived): {sorted(PC_FAMILIES)}")

    # separate byzantine conditions
    byz_by_cond = defaultdict(list)
    for r in byz_data:
        byz_by_cond[r["condition"]].append(r)

    # =================================================================
    # FIGURE 1: Byzantine Learning Curves
    # =================================================================
    fig1, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))

    conditions = [
        ("adv_0", "k=0 (baseline)", "#2ca02c", "o"),
        ("adv_1", "k=1 (hard)", "#d62728", "s"),
        ("adv_2", "k=2 (hard)", "#9467bd", "D"),
    ]

    rounds_range = range(1, 6)

    for cond_key, label, color, marker in conditions:
        rows = honest_rows(byz_by_cond[cond_key])
        rates = []
        for rnd in rounds_range:
            rnd_rows = [r for r in rows if int(r["round"]) == rnd]
            rates.append(coop_rate(rnd_rows) * 100)
        ax1.plot(list(rounds_range), rates, color=color, marker=marker,
                 label=label, markeredgecolor="white", markeredgewidth=0.4)

    # soft byzantine
    soft_honest = honest_rows(soft_data)
    soft_rates = []
    for rnd in rounds_range:
        rnd_rows = [r for r in soft_honest if int(r["round"]) == rnd]
        soft_rates.append(coop_rate(rnd_rows) * 100)
    ax1.plot(list(rounds_range), soft_rates, color="#ff7f0e", marker="^",
             label="k=1 (soft, p=0.5)", markeredgecolor="white",
             markeredgewidth=0.4, linestyle="--")

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Honest-Agent Cooperation Rate (%)")
    ax1.set_xticks(list(rounds_range))
    ax1.set_ylim(-5, 105)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.legend(loc="lower left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig1.tight_layout(pad=0.4)
    fig1.savefig(FIGURES / "byzantine_learning_curves.pdf", bbox_inches="tight")
    fig1.savefig(FIGURES / "byzantine_learning_curves.png", bbox_inches="tight")
    print(f"  [saved] byzantine_learning_curves.pdf/png")


    # =================================================================
    # FIGURE 2: Payoff Gap Bar Chart
    # =================================================================
    fig2, ax2 = plt.subplots(figsize=(FIG_W, FIG_H))

    # compute payoff/round for FD vs PC in each condition
    bar_conditions = [
        ("Hard k=1", honest_rows(byz_by_cond["adv_1"])),
        ("Hard k=2", honest_rows(byz_by_cond["adv_2"])),
        ("Soft k=1", soft_honest),
    ]

    fd_vals, pc_vals = [], []
    for label, rows in bar_conditions:
        fd_rows = [r for r in rows if r["model_family"] in FD_FAMILIES]
        pc_rows = [r for r in rows if r["model_family"] in PC_FAMILIES]
        fd_vals.append(avg_payoff_per_round(fd_rows))
        pc_vals.append(avg_payoff_per_round(pc_rows))

    x = np.arange(len(bar_conditions))
    width = 0.32

    bars_fd = ax2.bar(x - width / 2, fd_vals, width, label=f"FD ({fd_label})",
                      color="#1f77b4", edgecolor="white", linewidth=0.5)
    bars_pc = ax2.bar(x + width / 2, pc_vals, width, label=f"PC ({pc_label})",
                      color="#e377c2", edgecolor="white", linewidth=0.5)

    # add ratio annotations
    for i in range(len(bar_conditions)):
        if pc_vals[i] > 0:
            ratio = fd_vals[i] / pc_vals[i]
            y_top = max(fd_vals[i], pc_vals[i])
            ax2.annotate(f"{ratio:.0f}x" if ratio >= 2.5 else f"{ratio:.1f}x",
                         xy=(x[i], y_top + 0.15),
                         ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax2.set_ylabel("Avg Payoff / Round")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c[0] for c in bar_conditions])
    ax2.set_ylim(0, max(fd_vals) * 1.3)
    ax2.legend(loc="upper left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig2.tight_layout(pad=0.4)
    fig2.savefig(FIGURES / "payoff_gap_bars.pdf", bbox_inches="tight")
    fig2.savefig(FIGURES / "payoff_gap_bars.png", bbox_inches="tight")
    print(f"  [saved] payoff_gap_bars.pdf/png")


    # =================================================================
    # FIGURE 3: Full Results Summary Table (two cooperation metrics)
    # =================================================================
    fig3, ax3 = plt.subplots(figsize=(7.0, 3.8))
    ax3.axis("off")

    # All conditions: (label, condition_key, dataset)
    table_conditions = [
        ("Baseline (k=0)",         "adv_0",     byz_data),
        ("Hard Byz. (k=1)",        "adv_1",     byz_data),
        ("Hard Byz. (k=2)",        "adv_2",     byz_data),
        ("Soft Byz. (k=1, p=.5)",  "soft_p0.5", soft_data),
        ("Explicit broadcast",     "broadcast",  topo_data),
        ("Explicit ring",          "ring",       topo_data),
        ("Explicit star",          "star",       topo_data),
        ("Silent broadcast",       "broadcast",  topo_silent_data),
        ("Silent ring",            "ring",       topo_silent_data),
        ("Silent star",            "star",       topo_silent_data),
    ]

    col_labels = ["Condition", "Trials", "Group\nCoop %",
                  "Honest\nCoop %", "Avg\nPay/Rnd"]
    cell_data = []

    for label, cond_key, all_data in table_conditions:
        cond_rows = [r for r in all_data if r["condition"] == cond_key]
        h_rows = honest_rows(cond_rows)
        n_trials = len(set(r["trial_id"] for r in cond_rows))

        gcr = group_coop_rate(cond_rows) * 100
        hcr = coop_rate(h_rows) * 100
        avg_pay = avg_payoff_per_round(h_rows)

        cell_data.append([
            label, str(n_trials), f"{gcr:.1f}", f"{hcr:.1f}", f"{avg_pay:.2f}",
        ])

    table = ax3.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colWidths=[0.28, 0.10, 0.14, 0.14, 0.14],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.35)

    # style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#d9e2f3")
        cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#999999")

    # style data rows — shade groups
    # rows 0-3 = Byzantine, 4-6 = explicit topo, 7-9 = silent topo
    group_colors = {
        0: "#ffffff", 1: "#f2f2f2", 2: "#ffffff", 3: "#f2f2f2",
        4: "#e8f0e8", 5: "#dce8dc", 6: "#e8f0e8",
        7: "#e8e8f0", 8: "#dcdce8", 9: "#e8e8f0",
    }
    for i in range(len(cell_data)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_edgecolor("#cccccc")
            cell.set_facecolor(group_colors.get(i, "#ffffff"))

    # first column left-aligned
    for i in range(len(cell_data) + 1):
        table[i, 0].set_text_props(ha="left")

    fig3.tight_layout(pad=0.3)
    fig3.savefig(FIGURES / "results_summary_table.pdf", bbox_inches="tight")
    fig3.savefig(FIGURES / "results_summary_table.png", bbox_inches="tight")
    print(f"  [saved] results_summary_table.pdf/png")

    print(f"\nAll figures saved to {FIGURES}/")
