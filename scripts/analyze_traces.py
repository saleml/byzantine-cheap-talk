#!/usr/bin/env python3
"""
Qualitative analysis of reasoning traces across all experiments.

Scans all trial JSONs, flags erroneous traces, identifies common reasoning
themes, and compares behavioral patterns across model families and versions.

Output: results/trace_analysis.txt

Usage:
  python scripts/analyze_traces.py
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
from io import StringIO

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


# =====================================================================
# Trace extraction
# =====================================================================

def extract_traces(results_dir):
    """Extract all action-stage reasoning traces from a results directory."""
    traces = []
    for jf in sorted(results_dir.rglob("results.json")):
        with open(jf) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        agent_family = {a["name"]: a.get("model_family", "?") for a in meta.get("agents", [])}
        adversaries = set(meta.get("adversary_agents", []))
        version = meta.get("version", "?")
        experiment = meta.get("experiment", "?")
        trial = meta.get("trial", "?")

        for rd in data.get("rounds_data", []):
            if rd.get("stage") == "communication" or ("communications" in rd and "choices" not in rd):
                continue
            rn = rd.get("round", "?")
            choices = rd.get("choices", {})
            for agent, dec in rd.get("full_decisions", {}).items():
                if not isinstance(dec, dict):
                    continue
                reasoning = dec.get("reasoning", "")
                action = dec.get("action")
                choice = action.get("choice", "") if isinstance(action, dict) else ""
                traces.append({
                    "dir": results_dir.name,
                    "file": str(jf.relative_to(RESULTS)),
                    "version": version,
                    "experiment": experiment,
                    "trial": trial,
                    "round": rn,
                    "agent": agent,
                    "family": agent_family.get(agent, "?"),
                    "is_adversary": agent in adversaries,
                    "choice": choice,
                    "reasoning": reasoning,
                    "failed": dec.get("action_parsing_failed", False),
                })
    return traces


# =====================================================================
# Error / anomaly detection
# =====================================================================

def check_trace_errors(trace):
    """Return list of issue labels for a single trace."""
    issues = []
    r = trace["reasoning"]
    choice = trace["choice"]

    if trace["is_adversary"]:
        return []  # skip adversary traces

    if trace["failed"]:
        issues.append("PARSING_FAILED")
        return issues

    if not r:
        issues.append("EMPTY_REASONING")
        return issues

    # Nested JSON (model wrapped response in extra JSON)
    if r.strip().startswith('{') and '"reasoning"' in r[:50]:
        issues.append("NESTED_JSON")

    # Reasoning says one thing, action does another
    r_lower = r.lower()
    if choice == "Hunt Stag" and ("hunt hare" in r_lower and "i will" in r_lower and "hunt hare" in r_lower[-100:]):
        issues.append("REASONING_ACTION_MISMATCH_stag_but_said_hare")
    if choice == "Hunt Hare" and ("hunt stag" in r_lower and "i will" in r_lower and "hunt stag" in r_lower[-100:]):
        issues.append("REASONING_ACTION_MISMATCH_hare_but_said_stag")

    # Self-reference in third person
    agent = trace["agent"]
    if f"{agent} said" in r or f"{agent} broadcast" in r or f"{agent} chose" in r:
        issues.append("SELF_REFERENCE_THIRD_PERSON")

    # Extremely short reasoning (< 50 chars)
    if len(r.strip()) < 50:
        issues.append("VERY_SHORT_REASONING")

    # Extremely long reasoning (> 2000 chars — possible context confusion)
    if len(r.strip()) > 2000:
        issues.append("VERY_LONG_REASONING")

    return issues


# =====================================================================
# Thematic analysis
# =====================================================================

THEME_PATTERNS = {
    "trust_signal": r"trust|believe|honest|reliable|credible",
    "betrayal_detection": r"betray|defect|lied|broke|unreliable|deceptive",
    "risk_aversion": r"safe|guaranteed|risk|certain|secure|avoid.*0",
    "coordination_appeal": r"coordinate|cooperate|together|collective|mutual",
    "payoff_calculation": r"10 points|3 points|0 points|maximize|payoff",
    "game_theory": r"nash|equilibrium|dominant|rational|optimal|strategy",
    "history_analysis": r"round [0-9]|previous round|history|pattern|consistently",
    "uncertainty": r"uncertain|unsure|unknown|might|could|possible|cannot see",
    "topology_awareness": r"hub|spoke|neighbor|visible|hidden|cannot see|only see",
    "defection_rationale": r"safe.*hare|guarantee.*3|hunt hare.*safe|protect.*score",
    "cooperation_rationale": r"stag.*10|cooperate.*10|all.*hunt stag|collective.*best",
    "forgiveness": r"give.*chance|try again|one more|despite|still.*cooperate",
}


def classify_themes(reasoning):
    """Return set of theme labels present in a reasoning trace."""
    r_lower = reasoning.lower()
    return {theme for theme, pattern in THEME_PATTERNS.items() if re.search(pattern, r_lower)}


# =====================================================================
# Archetype analysis (improved — accounts for preemptive defectors)
# =====================================================================

def analyze_archetypes_detailed(traces):
    """Classify agents considering preemptive defection."""
    # Group by (dir, trial, agent)
    agent_trials = defaultdict(list)
    for t in traces:
        if t["is_adversary"] or t["failed"]:
            continue
        key = (t["dir"], t["trial"], t["agent"])
        agent_trials[key].append(t)

    family_stats = defaultdict(lambda: defaultdict(int))

    for (dirn, trial, agent), trial_traces in agent_trials.items():
        trial_traces.sort(key=lambda x: x["round"])
        family = trial_traces[0]["family"]
        actions = [t["choice"] for t in trial_traces]

        if not actions:
            continue

        if actions[0] == "Hunt Hare" and all(a == "Hunt Hare" for a in actions):
            family_stats[family]["preemptive_defector"] += 1
        elif all(a == "Hunt Stag" for a in actions):
            family_stats[family]["always_cooperate"] += 1
        elif actions[0] == "Hunt Stag":
            # Started cooperating — check if switched permanently
            first_hare = next((i for i, a in enumerate(actions) if a == "Hunt Hare"), None)
            if first_hare is not None:
                after = actions[first_hare:]
                if all(a == "Hunt Hare" for a in after):
                    family_stats[family]["fast_defector"] += 1
                else:
                    family_stats[family]["mixed"] += 1
            else:
                family_stats[family]["always_cooperate"] += 1
        else:
            family_stats[family]["other"] += 1

    return family_stats


# =====================================================================
# Main
# =====================================================================

def main():
    buf = StringIO()
    def out(s=""):
        print(s)
        buf.write(s + "\n")

    out("=" * 80)
    out("  TRACE ANALYSIS REPORT")
    out("=" * 80)
    out()

    # Collect all traces
    all_traces = []
    dirs = sorted([d for d in RESULTS.iterdir()
                   if d.is_dir() and d.name != "figures" and not d.name.startswith(".")])

    for d in dirs:
        traces = extract_traces(d)
        if traces:
            all_traces.extend(traces)
            out(f"  {d.name}: {len(traces)} traces extracted")

    honest_traces = [t for t in all_traces if not t["is_adversary"] and not t["failed"]]
    out(f"\n  Total: {len(all_traces)} traces ({len(honest_traces)} honest, non-failed)")

    # ---- Error detection ----
    out("\n" + "=" * 80)
    out("  FLAGGED TRACES (errors/anomalies)")
    out("=" * 80 + "\n")

    error_counts = defaultdict(int)
    flagged = []
    for t in all_traces:
        issues = check_trace_errors(t)
        if issues:
            flagged.append((t, issues))
            for iss in issues:
                error_counts[iss] += 1

    if error_counts:
        out("  Issue summary:")
        for iss, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            out(f"    {iss}: {count}")
        out()
        out(f"  Detailed flagged traces ({len(flagged)} total):")
        for t, issues in flagged[:50]:  # cap at 50
            out(f"    {t['dir']}/{t['file'].split('/',1)[-1]}  round={t['round']}  "
                f"{t['agent']} ({t['family']})  issues={issues}")
            out(f"      choice={t['choice']}  reasoning={t['reasoning'][:120]}...")
            out()
    else:
        out("  No flagged traces.")

    # ---- Theme analysis ----
    out("\n" + "=" * 80)
    out("  THEMATIC ANALYSIS")
    out("=" * 80 + "\n")

    # By model family
    family_themes = defaultdict(Counter)
    family_counts = Counter()
    for t in honest_traces:
        themes = classify_themes(t["reasoning"])
        family_themes[t["family"]].update(themes)
        family_counts[t["family"]] += 1

    families = sorted(family_counts.keys())
    themes = sorted(THEME_PATTERNS.keys())

    out(f"  {'Theme':<25}" + "".join(f"{f:>15}" for f in families))
    out(f"  {'-'*25}" + "".join(f"{'-'*15}" for _ in families))
    for theme in themes:
        vals = []
        for fam in families:
            count = family_themes[fam][theme]
            total = family_counts[fam]
            pct = count / total * 100 if total else 0
            vals.append(f"{pct:>13.1f}%")
        out(f"  {theme:<25}" + "".join(vals))

    # By choice (cooperators vs defectors)
    out("\n  --- Theme prevalence by action ---\n")
    choice_themes = defaultdict(Counter)
    choice_counts = Counter()
    for t in honest_traces:
        themes = classify_themes(t["reasoning"])
        choice_themes[t["choice"]].update(themes)
        choice_counts[t["choice"]] += 1

    out(f"  {'Theme':<25} {'Hunt Stag':>15} {'Hunt Hare':>15}")
    out(f"  {'-'*25} {'-'*15} {'-'*15}")
    for theme in themes:
        stag_pct = choice_themes["Hunt Stag"][theme] / choice_counts["Hunt Stag"] * 100 if choice_counts["Hunt Stag"] else 0
        hare_pct = choice_themes["Hunt Hare"][theme] / choice_counts["Hunt Hare"] * 100 if choice_counts["Hunt Hare"] else 0
        out(f"  {theme:<25} {stag_pct:>13.1f}% {hare_pct:>13.1f}%")

    # ---- Archetype analysis (improved) ----
    out("\n" + "=" * 80)
    out("  ARCHETYPE ANALYSIS (improved — includes preemptive defectors)")
    out("=" * 80 + "\n")

    for version_prefix in ["v1", "v2", "v3"]:
        version_traces = [t for t in all_traces
                          if t["dir"].endswith(f"_{version_prefix}")
                          and "byzantine" in t["dir"] and "soft" not in t["dir"]
                          and "star" not in t["dir"]]
        if not version_traces:
            continue

        # Filter to k=1 only (adv_1 condition)
        k1_traces = [t for t in version_traces
                      if "/adv_1/" in t["file"]]
        if not k1_traces:
            continue

        stats = analyze_archetypes_detailed(k1_traces)
        out(f"  --- {version_prefix} (k=1 hard Byzantine) ---\n")
        out(f"  {'Family':<15} {'preempt_def':>12} {'fast_def':>12} {'mixed':>12} "
            f"{'always_coop':>12} {'total':>8}")
        out(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
        for fam in sorted(stats.keys()):
            s = stats[fam]
            total = sum(s.values())
            out(f"  {fam:<15} {s['preemptive_defector']:>12} {s['fast_defector']:>12} "
                f"{s['mixed']:>12} {s['always_cooperate']:>12} {total:>8}")
        out()

    # ---- Archetype consistency across versions ----
    out("\n" + "=" * 80)
    out("  ARCHETYPE CONSISTENCY ACROSS VERSIONS")
    out("=" * 80 + "\n")

    out("  Model         v1            v2            v3            Consistent?")
    out("  " + "-" * 70)

    model_archetypes = {}
    for version_prefix in ["v1", "v2", "v3"]:
        version_traces = [t for t in all_traces
                          if t["dir"].endswith(f"_{version_prefix}")
                          and "byzantine" in t["dir"] and "soft" not in t["dir"]
                          and "star" not in t["dir"]]
        k1_traces = [t for t in version_traces if "/adv_1/" in t["file"]]
        stats = analyze_archetypes_detailed(k1_traces)
        for fam, s in stats.items():
            total = sum(s.values())
            dp_count = s["preemptive_defector"] + s["fast_defector"]
            label = "DP" if dp_count / total > 0.5 else "CP" if total > 0 else "?"
            model_archetypes.setdefault(fam, {})[version_prefix] = f"{label} ({dp_count}/{total})"

    all_models = sorted(model_archetypes.keys())
    for fam in all_models:
        arcs = model_archetypes[fam]
        v1 = arcs.get("v1", "—")
        v2 = arcs.get("v2", "—")
        v3 = arcs.get("v3", "—")
        labels = [a.split(" ")[0] for a in [v1, v2, v3] if a != "—"]
        consistent = "YES" if len(set(labels)) <= 1 else "NO"
        out(f"  {fam:<14} {v1:<14}{v2:<14}{v3:<14}{consistent}")

    # ---- Noteworthy traces ----
    out("\n" + "=" * 80)
    out("  NOTEWORTHY TRACES")
    out("=" * 80 + "\n")

    # Find best examples of each archetype
    for label, desc in [
        ("Fast defector — betrayal detection", "explicitly names betrayer and switches"),
        ("Persistent cooperator — forgiveness", "continues cooperating despite exploitation"),
        ("Preemptive defector — round 1 defection", "defects from the very first round"),
    ]:
        out(f"  --- {label} ---\n")
        found = 0
        for t in honest_traces:
            themes = classify_themes(t["reasoning"])
            if label.startswith("Fast") and t["round"] == 2 and t["choice"] == "Hunt Hare" and "betrayal_detection" in themes:
                if found < 2:
                    out(f"    {t['dir']} trial={t['trial']} round={t['round']} {t['agent']} ({t['family']})")
                    out(f"    Choice: {t['choice']}")
                    out(f"    Reasoning: {t['reasoning'][:300]}...")
                    out()
                    found += 1
            elif label.startswith("Persistent") and t["round"] >= 4 and t["choice"] == "Hunt Stag" and "forgiveness" in themes:
                if found < 2:
                    out(f"    {t['dir']} trial={t['trial']} round={t['round']} {t['agent']} ({t['family']})")
                    out(f"    Choice: {t['choice']}")
                    out(f"    Reasoning: {t['reasoning'][:300]}...")
                    out()
                    found += 1
            elif label.startswith("Preemptive") and t["round"] == 1 and t["choice"] == "Hunt Hare" and "risk_aversion" in themes:
                if found < 2:
                    out(f"    {t['dir']} trial={t['trial']} round={t['round']} {t['agent']} ({t['family']})")
                    out(f"    Choice: {t['choice']}")
                    out(f"    Reasoning: {t['reasoning'][:300]}...")
                    out()
                    found += 1
        if found == 0:
            out(f"    (no examples found)\n")

    # Save
    report_path = RESULTS / "trace_analysis.txt"
    report_path.write_text(buf.getvalue())
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()