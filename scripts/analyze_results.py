#!/usr/bin/env python3
"""
Comprehensive analysis of all NETYS experiment results.

Reads the four all_results.csv files and prints:
  (a) Behavioral archetype classification (k=1, k=2)
  (b) Per-model-family payoff breakdown (hard k=1, k=2, soft k=1)
  (c) Soft Byzantine split by adversary defection count
  (d) Round-by-round honest cooperation rates for all conditions
  (e) Master summary table

Usage:
  python scripts/analyze_results.py

Output is printed to stdout and saved to results/analysis_report.txt.
"""

# NOTE: results.json contains a bugged cooperation_rate field
# (engine.py divides by len(history) which double-counts
# communication-stage entries in two-stage games, halving
# the true rate). All statistics here are computed from the
# raw action columns in all_results.csv instead.

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict
from io import StringIO

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BYZANTINE_CSV = RESULTS / "byzantine" / "all_results.csv"
SOFT_CSV = RESULTS / "byzantine_soft" / "all_results.csv"
TOPO_CSV = RESULTS / "topology" / "all_results.csv"
TOPO_SILENT_CSV = RESULTS / "topology_silent" / "all_results.csv"

# ---------- reuse helpers from generate_summary_figures ----------
sys.path.insert(0, str(ROOT / "scripts"))
from generate_summary_figures import (
    load_csv, honest_rows, coop_rate, group_coop_rate, avg_payoff_per_round,
    FD_FAMILIES, PC_FAMILIES,
)

# ---------- model family constants ----------
ALL_FAMILIES = ["Mixtral", "Qwen", "Llama", "DeepSeek"]


class TeeOutput:
    """Write to both stdout and a StringIO buffer."""
    def __init__(self):
        self.buffer = StringIO()
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.buffer.write(text)

    def flush(self):
        self.stdout.flush()

    def get_value(self):
        return self.buffer.value if hasattr(self.buffer, 'value') else self.buffer.getvalue()


def family_of(row):
    return row.get("model_family", "Unknown")


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# =====================================================================
# (a) Behavioral archetype classification
# =====================================================================
def classify_archetype(action_sequence, first_betrayal_round):
    """
    Classify an honest agent's behavior after the first round they were
    betrayed (i.e. an adversary defected while they cooperated).

    Returns one of:
      - 'never_switched': agent never chose Hunt Hare after betrayal
      - 'permanently_switched': agent switched to Hunt Hare and never
        returned to Hunt Stag
      - 'mixed': agent alternated between Stag and Hare after betrayal

    If first_betrayal_round is None (no betrayal observed), returns
    'no_betrayal'.  If the betrayal occurred in the final round,
    there are no post-betrayal actions to classify, so returns
    'betrayal_last_round'.
    """
    if first_betrayal_round is None:
        return "no_betrayal"

    # Actions strictly AFTER the first betrayal round
    post_betrayal = [a for rnd, a in action_sequence if rnd > first_betrayal_round]
    if not post_betrayal:
        # Betrayal in the final round — no subsequent actions to classify
        return "betrayal_last_round"

    all_stag = all(a == "Hunt Stag" for a in post_betrayal)
    all_hare = all(a == "Hunt Hare" for a in post_betrayal)

    if all_stag:
        return "never_switched"
    elif all_hare:
        return "permanently_switched"
    else:
        return "mixed"


def analyze_archetypes(byz_data):
    print_separator("(a) Behavioral Archetype Classification")

    byz_by_cond = defaultdict(list)
    for r in byz_data:
        byz_by_cond[r["condition"]].append(r)

    for cond_key, cond_label in [("adv_1", "Hard Byzantine k=1"),
                                  ("adv_2", "Hard Byzantine k=2")]:
        print(f"--- {cond_label} ---\n")
        cond_rows = byz_by_cond[cond_key]

        # Group by trial
        trials = defaultdict(list)
        for r in cond_rows:
            trials[r["trial_id"]].append(r)

        # Per-model-family archetype counts
        archetype_counts = defaultdict(lambda: defaultdict(int))
        archetype_instances = defaultdict(list)

        for trial_id, rows in sorted(trials.items()):
            # Find adversary agents in this trial
            adversary_agents = {r["agent_id"] for r in rows
                                if r["is_adversary"] in ("True", "1", True)}
            honest_agents = {r["agent_id"] for r in rows
                             if r["is_adversary"] in ("False", "0", False)}

            # Build per-agent action sequences
            agent_actions = defaultdict(list)
            round_choices = defaultdict(dict)
            for r in rows:
                agent_actions[r["agent_id"]].append(
                    (int(r["round"]), r["action"]))
                round_choices[int(r["round"])][r["agent_id"]] = r["action"]

            # Find first round where ANY adversary defected (Hunt Hare)
            # while the honest agent cooperated (Hunt Stag)
            for agent in sorted(honest_agents):
                actions = sorted(agent_actions[agent])
                first_betrayal = None
                for rnd, _ in actions:
                    agent_chose_stag = round_choices[rnd].get(agent) == "Hunt Stag"
                    any_adv_hare = any(
                        round_choices[rnd].get(adv) == "Hunt Hare"
                        for adv in adversary_agents
                    )
                    if agent_chose_stag and any_adv_hare:
                        first_betrayal = rnd
                        break

                archetype = classify_archetype(actions, first_betrayal)
                fam = next((r["model_family"] for r in rows
                            if r["agent_id"] == agent), "Unknown")
                archetype_counts[fam][archetype] += 1
                archetype_instances[fam].append(
                    (trial_id, agent, archetype))

        # Print results table
        print(f"  {'Family':<12} {'never_sw':>10} {'perm_sw':>10} "
              f"{'mixed':>10} {'no_betray':>10} {'last_rnd':>10} {'total':>8}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for fam in ALL_FAMILIES:
            counts = archetype_counts[fam]
            total = sum(counts.values())
            print(f"  {fam:<12} {counts['never_switched']:>10} "
                  f"{counts['permanently_switched']:>10} "
                  f"{counts['mixed']:>10} "
                  f"{counts['no_betrayal']:>10} "
                  f"{counts['betrayal_last_round']:>10} {total:>8}")
        print()


# =====================================================================
# (b) Per-model-family payoff breakdown
# =====================================================================
def analyze_payoff_breakdown(byz_data, soft_data):
    print_separator("(b) Per-Model-Family Payoff Breakdown")

    byz_by_cond = defaultdict(list)
    for r in byz_data:
        byz_by_cond[r["condition"]].append(r)

    conditions = [
        ("Hard k=1", honest_rows(byz_by_cond["adv_1"])),
        ("Hard k=2", honest_rows(byz_by_cond["adv_2"])),
        ("Soft k=1", honest_rows(soft_data)),
    ]

    print(f"  {'Condition':<12} {'Family':<12} {'Avg Pay/Rnd':>12} "
          f"{'Group':>8} {'N_obs':>8}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*8} {'-'*8}")

    for cond_label, rows in conditions:
        for fam in ALL_FAMILIES:
            fam_rows = [r for r in rows if family_of(r) == fam]
            avg = avg_payoff_per_round(fam_rows)
            group = "FD" if fam in FD_FAMILIES else "PC"
            print(f"  {cond_label:<12} {fam:<12} {avg:>12.2f} "
                  f"{group:>8} {len(fam_rows):>8}")
        # FD vs PC aggregate
        fd_rows = [r for r in rows if family_of(r) in FD_FAMILIES]
        pc_rows = [r for r in rows if family_of(r) in PC_FAMILIES]
        fd_avg = avg_payoff_per_round(fd_rows)
        pc_avg = avg_payoff_per_round(pc_rows)
        ratio = fd_avg / pc_avg if pc_avg > 0 else float('inf')
        print(f"  {'':<12} {'FD total':<12} {fd_avg:>12.2f}")
        print(f"  {'':<12} {'PC total':<12} {pc_avg:>12.2f}")
        print(f"  {'':<12} {'FD/PC ratio':<12} {ratio:>12.1f}x")
        print()


# =====================================================================
# (c) Soft Byzantine split by adversary defection count
# =====================================================================
def analyze_soft_by_defections(soft_data):
    print_separator("(c) Soft Byzantine: Split by Adversary Defection Count")

    # Load per-trial adversary defection counts from trial JSONs
    soft_dir = RESULTS / "byzantine_soft"
    trial_defection_count = {}
    for t in range(1, 100):
        rfile = soft_dir / f"trial_{t:02d}" / "results.json"
        if not rfile.exists():
            break
        with open(rfile) as f:
            meta = json.load(f).get("metadata", {})
        adv_actions = meta.get("adversary_actions", [])
        n_defections = sum(1 for a in adv_actions
                           if a.get("choice") == "Hunt Hare")
        trial_defection_count[str(t)] = n_defections

    h_rows = honest_rows(soft_data)

    low = [r for r in h_rows
           if trial_defection_count.get(r["trial_id"], 0) <= 2]
    high = [r for r in h_rows
            if trial_defection_count.get(r["trial_id"], 0) >= 3]

    print(f"  Trial defection counts: {dict(sorted(trial_defection_count.items(), key=lambda x: int(x[0])))}")
    print(f"  Low-betrayal (<=2 defections): {len(set(r['trial_id'] for r in low))} trials, "
          f"{len(low)} honest obs")
    print(f"  High-betrayal (>=3 defections): {len(set(r['trial_id'] for r in high))} trials, "
          f"{len(high)} honest obs")
    print()

    print(f"  {'Split':<18} {'Group':>8} {'Avg Pay/Rnd':>12} {'Coop %':>10}")
    print(f"  {'-'*18} {'-'*8} {'-'*12} {'-'*10}")

    for split_label, split_rows in [("Low (<=2)", low), ("High (>=3)", high)]:
        fd = [r for r in split_rows if family_of(r) in FD_FAMILIES]
        pc = [r for r in split_rows if family_of(r) in PC_FAMILIES]
        fd_avg = avg_payoff_per_round(fd)
        pc_avg = avg_payoff_per_round(pc)
        fd_coop = coop_rate(fd) * 100
        pc_coop = coop_rate(pc) * 100
        ratio = fd_avg / pc_avg if pc_avg > 0 else float('inf')
        print(f"  {split_label:<18} {'FD':>8} {fd_avg:>12.2f} {fd_coop:>9.1f}%")
        print(f"  {split_label:<18} {'PC':>8} {pc_avg:>12.2f} {pc_coop:>9.1f}%")
        print(f"  {split_label:<18} {'ratio':>8} {ratio:>12.1f}x")
        print()


# =====================================================================
# (d) Round-by-round honest cooperation rates
# =====================================================================
def analyze_round_by_round(byz_data, soft_data, topo_data, topo_silent_data):
    print_separator("(d) Round-by-Round Honest Cooperation Rates (%)")

    byz_by_cond = defaultdict(list)
    for r in byz_data:
        byz_by_cond[r["condition"]].append(r)

    topo_by_cond = defaultdict(list)
    for r in topo_data:
        topo_by_cond[r["condition"]].append(r)

    topo_s_by_cond = defaultdict(list)
    for r in topo_silent_data:
        topo_s_by_cond[r["condition"]].append(r)

    all_conditions = [
        ("Baseline (k=0)",     honest_rows(byz_by_cond["adv_0"])),
        ("Hard k=1",           honest_rows(byz_by_cond["adv_1"])),
        ("Hard k=2",           honest_rows(byz_by_cond["adv_2"])),
        ("Soft k=1",           honest_rows(soft_data)),
        ("Expl. broadcast",    honest_rows(topo_by_cond["broadcast"])),
        ("Expl. ring",         honest_rows(topo_by_cond["ring"])),
        ("Expl. star",         honest_rows(topo_by_cond["star"])),
        ("Silent broadcast",   honest_rows(topo_s_by_cond["broadcast"])),
        ("Silent ring",        honest_rows(topo_s_by_cond["ring"])),
        ("Silent star",        honest_rows(topo_s_by_cond["star"])),
    ]

    rounds_range = range(1, 6)
    header = f"  {'Condition':<20}" + "".join(f"{'R'+str(r):>8}" for r in rounds_range)
    print(header)
    print(f"  {'-'*20}" + "".join(f"{'-'*8}" for _ in rounds_range))

    for label, rows in all_conditions:
        rates = []
        for rnd in rounds_range:
            rnd_rows = [r for r in rows if int(r["round"]) == rnd]
            rates.append(coop_rate(rnd_rows) * 100)
        vals = "".join(f"{v:>7.1f}%" for v in rates)
        print(f"  {label:<20}{vals}")
    print()


# =====================================================================
# (e) Master summary table
# =====================================================================
def print_summary_table(byz_data, soft_data, topo_data, topo_silent_data):
    print_separator("(e) Master Summary Table")
    print("  NOTE: All rates computed from raw CSV action columns.")
    print("  We never use results.json cooperation_rate (halved by")
    print("  len(history) bug in engine.py for two-stage games).\n")

    byz_by_cond = defaultdict(list)
    for r in byz_data:
        byz_by_cond[r["condition"]].append(r)

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

    print(f"  {'Condition':<25} {'Trials':>7} {'Group%':>8} "
          f"{'Honest%':>9} {'AvgPay':>8}")
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*9} {'-'*8}")

    for label, cond_key, all_data in table_conditions:
        cond_rows = [r for r in all_data if r["condition"] == cond_key]
        h_rows = honest_rows(cond_rows)
        n_trials = len(set(r["trial_id"] for r in cond_rows))
        gcr = group_coop_rate(cond_rows) * 100
        hcr = coop_rate(h_rows) * 100
        avg_pay = avg_payoff_per_round(h_rows)
        print(f"  {label:<25} {n_trials:>7} {gcr:>7.1f}% "
              f"{hcr:>8.1f}% {avg_pay:>8.2f}")
    print()


# =====================================================================
# main
# =====================================================================
def main():
    tee = TeeOutput()
    sys.stdout = tee

    print("NETYS 2026 — Full Analysis Report")
    print(f"Generated from raw CSV data in {RESULTS}/")
    print(f"Date: {__import__('datetime').datetime.now().isoformat()}")

    # Load data
    byz_data = load_csv(BYZANTINE_CSV)
    soft_data = load_csv(SOFT_CSV)
    topo_data = load_csv(TOPO_CSV)
    topo_silent_data = load_csv(TOPO_SILENT_CSV)

    # Run all analyses
    analyze_archetypes(byz_data)
    analyze_payoff_breakdown(byz_data, soft_data)
    analyze_soft_by_defections(soft_data)
    analyze_round_by_round(byz_data, soft_data, topo_data, topo_silent_data)
    print_summary_table(byz_data, soft_data, topo_data, topo_silent_data)

    # Save to file
    sys.stdout = tee.stdout
    report_path = RESULTS / "analysis_report.txt"
    report_path.write_text(tee.get_value())
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
