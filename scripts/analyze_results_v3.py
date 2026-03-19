#!/usr/bin/env python3
"""
Analysis of byzantine_v3 experiment results.

Same analysis as analyze_results.py but targeting results/byzantine_v3/ only.
Model cohort: Mixtral, Qwen, GPT-4o, DeepSeek.

Usage:
  python scripts/analyze_results_v3.py

Output saved to results/analysis_report_v3.txt.
"""

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict
from io import StringIO

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BYZANTINE_V3_CSV = RESULTS / "byzantine_v3" / "all_results.csv"

# ---------- helpers (inline to avoid import issues) ----------
def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def honest_rows(rows):
    return [r for r in rows if r["is_adversary"] in ("False", "0", False)]

def coop_rate(rows):
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["action"] == "Hunt Stag") / len(rows)

def group_coop_rate(rows):
    if not rows:
        return 0.0
    groups = defaultdict(list)
    for r in rows:
        groups[(r["trial_id"], r["round"])].append(r)
    n_all_stag = sum(
        1 for g in groups.values()
        if all(r["action"] == "Hunt Stag" for r in g)
    )
    return n_all_stag / len(groups)

def avg_payoff_per_round(rows):
    if not rows:
        return 0.0
    return sum(float(r["payoff"]) for r in rows) / len(rows)


# ---------- model family constants for v3 ----------
ALL_FAMILIES = ["Mixtral", "Qwen", "GPT-4o", "DeepSeek"]
FD_FAMILIES = {"Mixtral", "DeepSeek"}
PC_FAMILIES = {"Qwen", "GPT-4o"}  # tentative — will be revealed by data


class TeeOutput:
    def __init__(self):
        self.buffer = StringIO()
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.buffer.write(text)

    def flush(self):
        self.stdout.flush()

    def get_value(self):
        return self.buffer.getvalue()


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
    if first_betrayal_round is None:
        return "no_betrayal"
    post_betrayal = [a for rnd, a in action_sequence if rnd > first_betrayal_round]
    if not post_betrayal:
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

    for cond_key, cond_label in [
        ("adv_1", "Hard Byzantine k=1"),
        ("adv_2", "Hard Byzantine k=2"),
    ]:
        print(f"--- {cond_label} ---\n")
        cond_rows = byz_by_cond[cond_key]

        trials = defaultdict(list)
        for r in cond_rows:
            trials[r["trial_id"]].append(r)

        archetype_counts = defaultdict(lambda: defaultdict(int))

        for trial_id, rows in sorted(trials.items()):
            adversary_agents = {
                r["agent_id"] for r in rows if r["is_adversary"] in ("True", "1", True)
            }
            honest_agents = {
                r["agent_id"] for r in rows if r["is_adversary"] in ("False", "0", False)
            }

            agent_actions = defaultdict(list)
            round_choices = defaultdict(dict)
            for r in rows:
                agent_actions[r["agent_id"]].append((int(r["round"]), r["action"]))
                round_choices[int(r["round"])][r["agent_id"]] = r["action"]

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
                fam = next(
                    (r["model_family"] for r in rows if r["agent_id"] == agent),
                    "Unknown",
                )
                archetype_counts[fam][archetype] += 1

        print(
            f"  {'Family':<12} {'never_sw':>10} {'perm_sw':>10} "
            f"{'mixed':>10} {'no_betray':>10} {'last_rnd':>10} {'total':>8}"
        )
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for fam in ALL_FAMILIES:
            counts = archetype_counts[fam]
            total = sum(counts.values())
            print(
                f"  {fam:<12} {counts['never_switched']:>10} "
                f"{counts['permanently_switched']:>10} "
                f"{counts['mixed']:>10} "
                f"{counts['no_betrayal']:>10} "
                f"{counts['betrayal_last_round']:>10} {total:>8}"
            )
        print()


# =====================================================================
# (b) Per-model-family payoff breakdown (byzantine only, no soft)
# =====================================================================
def analyze_payoff_breakdown(byz_data):
    print_separator("(b) Per-Model-Family Payoff Breakdown")

    byz_by_cond = defaultdict(list)
    for r in byz_data:
        byz_by_cond[r["condition"]].append(r)

    conditions = [
        ("Hard k=1", honest_rows(byz_by_cond["adv_1"])),
        ("Hard k=2", honest_rows(byz_by_cond["adv_2"])),
    ]

    print(
        f"  {'Condition':<12} {'Family':<12} {'Avg Pay/Rnd':>12} "
        f"{'Coop %':>10} {'N_obs':>8}"
    )
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")

    for cond_label, rows in conditions:
        for fam in ALL_FAMILIES:
            fam_rows = [r for r in rows if family_of(r) == fam]
            avg = avg_payoff_per_round(fam_rows)
            cr = coop_rate(fam_rows) * 100
            print(
                f"  {cond_label:<12} {fam:<12} {avg:>12.2f} "
                f"{cr:>9.1f}% {len(fam_rows):>8}"
            )
        # FD vs PC aggregate
        fd_rows = [r for r in rows if family_of(r) in FD_FAMILIES]
        pc_rows = [r for r in rows if family_of(r) in PC_FAMILIES]
        fd_avg = avg_payoff_per_round(fd_rows)
        pc_avg = avg_payoff_per_round(pc_rows)
        ratio = fd_avg / pc_avg if pc_avg > 0 else float("inf")
        print(f"  {'':<12} {'FD total':<12} {fd_avg:>12.2f}")
        print(f"  {'':<12} {'PC total':<12} {pc_avg:>12.2f}")
        print(f"  {'':<12} {'FD/PC ratio':<12} {ratio:>12.1f}x")
        print()


# =====================================================================
# (c) Round-by-round honest cooperation rates
# =====================================================================
def analyze_round_by_round(byz_data):
    print_separator("(c) Round-by-Round Honest Cooperation Rates (%)")

    byz_by_cond = defaultdict(list)
    for r in byz_data:
        byz_by_cond[r["condition"]].append(r)

    all_conditions = [
        ("Baseline (k=0)", honest_rows(byz_by_cond["adv_0"])),
        ("Hard k=1", honest_rows(byz_by_cond["adv_1"])),
        ("Hard k=2", honest_rows(byz_by_cond["adv_2"])),
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

    # Per-model round-by-round for k=1
    print()
    print("  --- Per-model round-by-round for k=1 ---\n")
    k1_honest = honest_rows(byz_by_cond["adv_1"])
    header = f"  {'Model':<12}" + "".join(f"{'R'+str(r):>8}" for r in rounds_range)
    print(header)
    print(f"  {'-'*12}" + "".join(f"{'-'*8}" for _ in rounds_range))
    for fam in ALL_FAMILIES:
        fam_rows = [r for r in k1_honest if family_of(r) == fam]
        rates = []
        for rnd in rounds_range:
            rnd_rows = [r for r in fam_rows if int(r["round"]) == rnd]
            rates.append(coop_rate(rnd_rows) * 100)
        vals = "".join(f"{v:>7.1f}%" for v in rates)
        print(f"  {fam:<12}{vals}")
    print()


# =====================================================================
# (d) Master summary table
# =====================================================================
def print_summary_table(byz_data):
    print_separator("(d) Master Summary Table")
    print("  NOTE: All rates computed from raw CSV action columns.\n")

    byz_by_cond = defaultdict(list)
    for r in byz_data:
        byz_by_cond[r["condition"]].append(r)

    table_conditions = [
        ("Baseline (k=0)", "adv_0", byz_data),
        ("Hard Byz. (k=1)", "adv_1", byz_data),
        ("Hard Byz. (k=2)", "adv_2", byz_data),
    ]

    print(
        f"  {'Condition':<25} {'Trials':>7} {'Group%':>8} "
        f"{'Honest%':>9} {'AvgPay':>8}"
    )
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*9} {'-'*8}")

    for label, cond_key, all_data in table_conditions:
        cond_rows = [r for r in all_data if r["condition"] == cond_key]
        h_rows = honest_rows(cond_rows)
        n_trials = len(set(r["trial_id"] for r in cond_rows))
        gcr = group_coop_rate(cond_rows) * 100
        hcr = coop_rate(h_rows) * 100
        avg_pay = avg_payoff_per_round(h_rows)
        print(
            f"  {label:<25} {n_trials:>7} {gcr:>7.1f}% "
            f"{hcr:>8.1f}% {avg_pay:>8.2f}"
        )
    print()


# =====================================================================
# (e) Per-model cooperation rates (all conditions)
# =====================================================================
def analyze_per_model(byz_data):
    print_separator("(e) Per-Model Cooperation Rates")

    byz_by_cond = defaultdict(list)
    for r in byz_data:
        byz_by_cond[r["condition"]].append(r)

    print(
        f"  {'Condition':<18} {'Model':<12} {'Coop%':>8} "
        f"{'AvgPay':>8} {'N_obs':>7}"
    )
    print(f"  {'-'*18} {'-'*12} {'-'*8} {'-'*8} {'-'*7}")

    for cond_label, cond_key in [
        ("Baseline (k=0)", "adv_0"),
        ("Hard k=1", "adv_1"),
        ("Hard k=2", "adv_2"),
    ]:
        h_rows = honest_rows(byz_by_cond[cond_key])
        for fam in ALL_FAMILIES:
            fam_rows = [r for r in h_rows if family_of(r) == fam]
            cr = coop_rate(fam_rows) * 100
            avg = avg_payoff_per_round(fam_rows)
            print(
                f"  {cond_label:<18} {fam:<12} {cr:>7.1f}% "
                f"{avg:>8.2f} {len(fam_rows):>7}"
            )
        print()


# =====================================================================
# main
# =====================================================================
def main():
    tee = TeeOutput()
    sys.stdout = tee

    print("NETYS 2026 — Byzantine v3 Analysis Report")
    print(f"Generated from: {BYZANTINE_V3_CSV}")
    print(f"Model cohort: {', '.join(ALL_FAMILIES)}")
    print(f"Date: {__import__('datetime').datetime.now().isoformat()}")

    byz_data = load_csv(BYZANTINE_V3_CSV)

    # Quick data sanity check
    families_found = sorted(set(r["model_family"] for r in byz_data))
    conditions_found = sorted(set(r["condition"] for r in byz_data))
    print(f"\nFamilies in data: {families_found}")
    print(f"Conditions in data: {conditions_found}")
    print(f"Total rows: {len(byz_data)}")

    analyze_archetypes(byz_data)
    analyze_payoff_breakdown(byz_data)
    analyze_round_by_round(byz_data)
    print_summary_table(byz_data)
    analyze_per_model(byz_data)

    # Save to file
    sys.stdout = tee.stdout
    report_path = RESULTS / "analysis_report_v3.txt"
    report_path.write_text(tee.get_value())
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
