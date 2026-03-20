#!/usr/bin/env python3
"""
Diagnostic script: scan ALL trial JSON files for malformed action-stage outputs.

Checks:
  1. WRONG_SCHEMA:  action has "type"/"word" instead of "choice"
  2. TRUNCATED_REASONING: reasoning starts with '{\\n' or contains nested JSON
  3. MISSING_ACTION: action is null/empty or missing "choice" key
  4. INVALID_CHOICE: choice is not "Hunt Stag" or "Hunt Hare"

Usage:
  python scripts/check_output_format.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Auto-discover all subdirectories in results/ that contain trial data
DIRS = sorted([
    d.name for d in RESULTS.iterdir()
    if d.is_dir() and d.name != "figures" and not d.name.startswith(".")
])

VALID_CHOICES = {"hunt stag", "hunt hare"}


def is_action_stage(rd):
    """True if this round entry is an action stage (not communication)."""
    if rd.get("stage") == "communication":
        return False
    # Also skip comm-only entries that lack choices
    if "communications" in rd and "choices" not in rd:
        return False
    return True


def check_decision(agent_name, decision, model_family):
    """Return list of (issue_type, detail) tuples for one agent's decision."""
    issues = []
    if not isinstance(decision, dict):
        issues.append(("MISSING_ACTION", "decision is not a dict"))
        return issues

    action = decision.get("action")
    reasoning = decision.get("reasoning", "")

    # --- Check reasoning ---
    if isinstance(reasoning, str):
        stripped = reasoning.strip()
        if stripped.startswith('{\\n') or stripped.startswith('{\\n') or stripped.startswith('{\n'):
            issues.append(("TRUNCATED_REASONING", f"starts with '{{\\n': {stripped[:80]}..."))
        elif stripped.startswith('{') and '"reasoning"' in stripped:
            issues.append(("TRUNCATED_REASONING", f"nested JSON detected: {stripped[:80]}..."))

    # --- Check action ---
    if action is None:
        issues.append(("MISSING_ACTION", "action is null"))
        return issues

    if not isinstance(action, dict):
        issues.append(("MISSING_ACTION", f"action is {type(action).__name__}: {str(action)[:60]}"))
        return issues

    # Wrong schema: has type/word (communication schema) instead of choice
    if "type" in action and "word" in action and "choice" not in action:
        issues.append(("WRONG_SCHEMA", f"action has type/word instead of choice: {action}"))
        return issues

    if "choice" not in action:
        # May have other unexpected keys
        issues.append(("MISSING_ACTION", f"no 'choice' key in action: {action}"))
        return issues

    choice = action["choice"]
    if not isinstance(choice, str) or choice.strip().lower() not in VALID_CHOICES:
        issues.append(("INVALID_CHOICE", f"choice={choice!r}"))

    return issues


def scan_directory(dir_path):
    """Scan all trial JSONs in a directory tree. Returns list of issue dicts."""
    issues = []
    rounds_checked = 0

    json_files = sorted(dir_path.rglob("results.json"))
    if not json_files:
        return issues, rounds_checked, len(json_files)

    for jf in json_files:
        rel = jf.relative_to(RESULTS)
        try:
            with open(jf) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            issues.append({
                "file": str(rel),
                "round": None,
                "agent_id": None,
                "model_family": None,
                "issue_type": "JSON_PARSE_ERROR",
                "detail": str(e),
            })
            continue

        metadata = data.get("metadata", {})
        # Build agent->family map
        agent_family = {}
        for ag in metadata.get("agents", []):
            agent_family[ag["name"]] = ag.get("model_family", "Unknown")

        rounds_data = data.get("rounds_data", [])
        for rd in rounds_data:
            if not is_action_stage(rd):
                continue

            rn = rd.get("round", "?")
            full_decisions = rd.get("full_decisions", {})
            rounds_checked += 1

            for agent_name, decision in full_decisions.items():
                mf = agent_family.get(agent_name, "Unknown")
                agent_issues = check_decision(agent_name, decision, mf)
                for issue_type, detail in agent_issues:
                    issues.append({
                        "file": str(rel),
                        "round": rn,
                        "agent_id": agent_name,
                        "model_family": mf,
                        "issue_type": issue_type,
                        "detail": detail,
                    })

    return issues, rounds_checked, len(json_files)


def main():
    print("=" * 90)
    print("  Output Format Diagnostic — Scanning all trial JSONs")
    print("=" * 90)
    print()

    all_issues = []
    summary = []

    for dirname in DIRS:
        dir_path = RESULTS / dirname
        if not dir_path.exists():
            print(f"[skip] {dirname}/ — does not exist")
            continue

        issues, rounds_checked, n_files = scan_directory(dir_path)

        if n_files == 0:
            print(f"[skip] {dirname}/ — no results.json files found")
            continue

        affected_agents = set()
        affected_families = set()
        type_counts = defaultdict(int)
        for iss in issues:
            affected_agents.add(iss["agent_id"])
            affected_families.add(iss["model_family"])
            type_counts[iss["issue_type"]] += 1

        summary.append({
            "dir": dirname,
            "files": n_files,
            "rounds": rounds_checked,
            "issues": len(issues),
            "agents": affected_agents,
            "families": affected_families,
            "type_counts": dict(type_counts),
            "pct": (len(issues) / rounds_checked * 100) if rounds_checked > 0 else 0,
        })

        if issues:
            print(f"\n--- {dirname}/ ({n_files} files, {rounds_checked} action-rounds, "
                  f"{len(issues)} issues) ---")
            for iss in issues:
                print(f"  {iss['file']}  round={iss['round']}  "
                      f"{iss['agent_id']} ({iss['model_family']})  "
                      f"{iss['issue_type']}: {iss['detail']}")
        else:
            print(f"[ok]   {dirname}/ — {n_files} files, {rounds_checked} action-rounds, 0 issues")

        all_issues.extend(issues)

    # ---- Summary table ----
    print()
    print("=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    print()
    print(f"  {'Directory':<35} {'Files':>6} {'Rounds':>7} {'Issues':>7} "
          f"{'%':>6}  {'Affected families'}")
    print(f"  {'-'*35} {'-'*6} {'-'*7} {'-'*7} {'-'*6}  {'-'*30}")

    flagged = []
    for s in summary:
        fam_str = ", ".join(sorted(s["families"])) if s["families"] else "—"
        flag = " *** FLAGGED" if s["pct"] > 5 else ""
        print(f"  {s['dir']:<35} {s['files']:>6} {s['rounds']:>7} {s['issues']:>7} "
              f"{s['pct']:>5.1f}%  {fam_str}{flag}")
        if s["pct"] > 5:
            flagged.append(s)

    # ---- Issue type breakdown ----
    print()
    total_type = defaultdict(int)
    for s in summary:
        for k, v in s["type_counts"].items():
            total_type[k] += v
    if total_type:
        print("  Issue type breakdown (all directories):")
        for k, v in sorted(total_type.items(), key=lambda x: -x[1]):
            print(f"    {k:<25} {v:>6}")
    else:
        print("  No issues found across any directory.")

    # ---- Flagged dirs ----
    if flagged:
        print()
        print(f"  *** {len(flagged)} directories FLAGGED (>5% issue rate):")
        for s in flagged:
            print(f"      {s['dir']}  ({s['issues']}/{s['rounds']} = {s['pct']:.1f}%)")

    print()
    print(f"  Total: {sum(s['rounds'] for s in summary)} action-rounds checked, "
          f"{len(all_issues)} issues found across {sum(s['files'] for s in summary)} files.")


if __name__ == "__main__":
    main()
