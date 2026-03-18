#!/usr/bin/env python3
"""
Experiment: Byzantine × Star Topology Crossing

Combines a single hard Byzantine adversary with a star communication
topology, testing whether the adversary's structural position matters.

Two sub-conditions (run sequentially in the same script):
  - hub_is_adversary: the Byzantine agent is always assigned as
    the hub (sees all messages, visible to all spokes)
  - hub_is_honest: the hub is always an honest agent; the Byzantine
    agent is one of the three spokes

In both cases:
  - 1 Byzantine adversary (always broadcasts "stag", always hunts hare)
  - Star topology with explicit visibility announcement
  - Same 4 heterogeneous models as other experiments
  - 10 trials, 5 rounds each
  - Adversary identity (which model) randomized per trial within the
    positional constraint (hub vs spoke)

Results saved to:
  results/byzantine_star/hub_is_adversary/trial_XX/results.json
  results/byzantine_star/hub_is_honest/trial_XX/results.json
  results/byzantine_star/all_results.csv

Usage:
  python scripts/run_byzantine_star.py                    # full run
  python scripts/run_byzantine_star.py --trials 5         # fewer trials
  python scripts/run_byzantine_star.py --dry-run          # verify setup, no API calls
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Set, Optional

# Ensure repo root and src/ on path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Load .env before importing engine (module-level OpenAI client needs API key)
_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k, _v)

from src.engine import GameEngine
from games import StagHuntWithCommunication

# Reuse topology class and helpers from run_topology
from run_topology import (
    TopologyStagHunt, build_visibility, enrich_model_family,
    add_words_visible,
)

# Reuse Byzantine patch from run_byzantine
from run_byzantine import patch_byzantine


# ---------- agents (same heterogeneous cohort as the paper) ----------
AGENTS = [
    {"name": "Agent_1", "model": "mistralai/Mixtral-8x22B-Instruct-v0.1", "model_family": "Mixtral"},
    {"name": "Agent_2", "model": "Qwen/Qwen2.5-72B-Instruct", "model_family": "Qwen"},
    {"name": "Agent_3", "model": "meta-llama/Llama-3.3-70B-Instruct", "model_family": "Llama"},
    {"name": "Agent_4", "model": "deepseek-ai/DeepSeek-V3", "model_family": "DeepSeek"},
]

AGENT_FAMILY = {a["name"]: a["model_family"] for a in AGENTS}


def run_one_trial(condition: str, trial_num: int, out_dir: Path,
                  rounds: int):
    """
    Run a single trial for a given condition.

    condition: "hub_is_adversary" or "hub_is_honest"
    """
    agent_names = [a["name"] for a in AGENTS]

    if condition == "hub_is_adversary":
        # Adversary is the hub; pick which agent is adversary (= hub)
        adversary_name = random.choice(agent_names)
        hub_name = adversary_name
    elif condition == "hub_is_honest":
        # Hub is honest; pick hub first, then pick adversary from spokes
        hub_name = random.choice(agent_names)
        spoke_names = [n for n in agent_names if n != hub_name]
        adversary_name = random.choice(spoke_names)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    adversary_names = {adversary_name}

    adversary_detail = [{
        "agent_id": adversary_name,
        "model_family": AGENT_FAMILY[adversary_name],
    }]
    hub_detail = {
        "agent_id": hub_name,
        "model_family": AGENT_FAMILY[hub_name],
    }

    # Create game with star topology and explicit visibility
    game = TopologyStagHunt(
        agents=AGENTS, rounds=rounds,
        topology="star", hub_name=hub_name,
    )
    engine = GameEngine(game)
    patch_byzantine(engine, game, adversary_names)

    results = engine.run()

    # Fix total_rounds (engine double-counts two-stage entries)
    results["total_rounds"] = rounds

    # Build visibility map for enrichment
    vis_map = build_visibility("star", agent_names, hub_name=hub_name)

    # Enrich reasoning traces
    enrich_model_family(results)
    add_words_visible(results, vis_map)

    # Attach experiment metadata
    results["metadata"] = {
        "experiment": "byzantine_star",
        "condition": condition,
        "n_adversaries": 1,
        "adversary_agents": [adversary_name],
        "adversary_detail": adversary_detail,
        "topology": "star",
        "hub_agent": hub_detail,
        "adversary_is_hub": condition == "hub_is_adversary",
        "visibility_map": {k: sorted(v) for k, v in vis_map.items()},
        "trial": trial_num,
        "rounds": rounds,
        "agents": AGENTS,
        "timestamp": datetime.now().isoformat(),
    }

    # Save
    trial_dir = out_dir / f"trial_{trial_num:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    with open(trial_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return trial_num


def run_condition(condition: str, trials: int, out_dir: Path,
                  workers: int, rounds: int):
    """Run all trials for one condition."""
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for t in range(1, trials + 1):
        if (out_dir / f"trial_{t:02d}" / "results.json").exists():
            print(f"  [skip] {condition} trial {t:02d} already exists")
            continue
        todo.append(t)

    if not todo:
        print(f"  All {trials} trials already complete for {condition}")
        return

    print(f"  Running {len(todo)} trials for {condition} (workers={workers})")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one_trial, condition, t, out_dir, rounds): t
            for t in todo
        }
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                fut.result()
                print(f"  [done] {condition} trial {t:02d}")
            except Exception as e:
                print(f"  [FAIL] {condition} trial {t:02d}: {e}")


def generate_csv(out_root: Path, conditions: List[str], trials: int):
    """
    Generate a flat CSV combining both conditions.
    One row per (trial, round, agent).
    """
    csv_path = out_root / "all_results.csv"
    fieldnames = [
        "trial_id", "condition", "round", "agent_id", "model_family",
        "word_broadcast", "action", "payoff", "reasoning",
        "is_adversary", "hub_agent_id",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cond in conditions:
            cond_dir = out_root / cond
            for t in range(1, trials + 1):
                results_file = cond_dir / f"trial_{t:02d}" / "results.json"
                if not results_file.exists():
                    continue
                with open(results_file) as rf:
                    results = json.load(rf)

                metadata = results.get("metadata", {})
                adversary_set = set(metadata.get("adversary_agents", []))
                hub_detail = metadata.get("hub_agent")
                hub_agent_id = hub_detail["agent_id"] if hub_detail else ""
                rounds_data = results.get("rounds_data", [])

                # Pair communication + action entries by round number
                comm_by_round: Dict[int, Dict] = {}
                action_by_round: Dict[int, Dict] = {}
                for rd in rounds_data:
                    rn = rd.get("round")
                    if rd.get("stage") == "communication" or (
                        "communications" in rd and "choices" not in rd
                    ):
                        comm_by_round[rn] = rd
                    elif "choices" in rd:
                        action_by_round[rn] = rd

                for rn in sorted(action_by_round):
                    act = action_by_round[rn]
                    comm = comm_by_round.get(rn, {})

                    communications = act.get(
                        "communications", comm.get("communications", {})
                    )
                    choices = act.get("choices", {})
                    payoffs = act.get("payoffs", {})

                    comm_decisions = comm.get("full_decisions", {})
                    act_decisions = act.get("full_decisions", {})

                    for agent in [a["name"] for a in AGENTS]:
                        comm_reason = (
                            comm_decisions.get(agent, {}).get("reasoning", "")
                            if isinstance(comm_decisions.get(agent), dict)
                            else ""
                        )
                        act_reason = (
                            act_decisions.get(agent, {}).get("reasoning", "")
                            if isinstance(act_decisions.get(agent), dict)
                            else ""
                        )
                        reasoning = f"COMM: {comm_reason} | ACTION: {act_reason}"

                        writer.writerow({
                            "trial_id": t,
                            "condition": cond,
                            "round": rn,
                            "agent_id": agent,
                            "model_family": AGENT_FAMILY.get(agent, "Unknown"),
                            "word_broadcast": communications.get(agent, ""),
                            "action": choices.get(agent, ""),
                            "payoff": payoffs.get(agent, 0),
                            "reasoning": reasoning,
                            "is_adversary": agent in adversary_set,
                            "hub_agent_id": hub_agent_id,
                        })

    print(f"  CSV written to {csv_path}")


def dry_run():
    """
    Print the visibility map and agent assignments for one trial of each
    condition, without making any API calls.
    """
    conditions = ["hub_is_adversary", "hub_is_honest"]
    agent_names = [a["name"] for a in AGENTS]

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"  DRY RUN — Condition: {condition}")
        print(f"{'='*60}")

        if condition == "hub_is_adversary":
            adversary_name = random.choice(agent_names)
            hub_name = adversary_name
        else:
            hub_name = random.choice(agent_names)
            spoke_names = [n for n in agent_names if n != hub_name]
            adversary_name = random.choice(spoke_names)

        vis_map = build_visibility("star", agent_names, hub_name=hub_name)

        print(f"\n  Hub agent:       {hub_name} ({AGENT_FAMILY[hub_name]})")
        print(f"  Adversary agent: {adversary_name} ({AGENT_FAMILY[adversary_name]})")
        print(f"  Adversary is hub: {adversary_name == hub_name}")
        print()

        # Verify constraint
        if condition == "hub_is_adversary":
            assert adversary_name == hub_name, \
                f"FAIL: adversary ({adversary_name}) != hub ({hub_name})"
            print("  CONSTRAINT CHECK: adversary == hub  ✓")
        else:
            assert adversary_name != hub_name, \
                f"FAIL: adversary ({adversary_name}) == hub ({hub_name})"
            print("  CONSTRAINT CHECK: adversary != hub  ✓")

        print(f"\n  Visibility map (star topology):")
        for agent, visible in sorted(vis_map.items()):
            role_parts = []
            if agent == hub_name:
                role_parts.append("HUB")
            else:
                role_parts.append("spoke")
            if agent == adversary_name:
                role_parts.append("ADVERSARY")
            else:
                role_parts.append("honest")
            role = ", ".join(role_parts)
            print(f"    {agent} ({AGENT_FAMILY[agent]:>8}, {role}): "
                  f"sees {sorted(visible)}")

        # Show what the adversary would do
        print(f"\n  Adversary behavior:")
        print(f"    Communication: always broadcasts 'stag'")
        print(f"    Action: always chooses 'Hunt Hare'")

        # Show what the topology prompt says for each role
        print(f"\n  Prompt visibility descriptions:")
        for agent in agent_names:
            if agent == hub_name:
                desc = "You are the central hub. You can see all players' messages."
            else:
                desc = (f"You can only see the hub player's ({hub_name}) message. "
                        f"Other players' messages are hidden from you.")
            is_adv = " [ADVERSARY — prompt never sent, patched]" if agent == adversary_name else ""
            print(f"    {agent}: {desc}{is_adv}")

    print(f"\n{'='*60}")
    print("  DRY RUN COMPLETE — No API calls were made.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment: Byzantine × Star Topology")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", type=str, default="results/byzantine_star")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers per condition")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print setup for one trial per condition, no API calls")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    out_root = Path(args.out)
    conditions = ["hub_is_adversary", "hub_is_honest"]

    for condition in conditions:
        print(f"\n{'='*50}")
        print(f"Condition: {condition}")
        print(f"{'='*50}")
        run_condition(condition, args.trials, out_root / condition,
                      args.workers, args.rounds)

    # Generate combined flat CSV
    print(f"\n{'='*50}")
    print("Generating combined CSV...")
    generate_csv(out_root, conditions, args.trials)

    print("\nAll Byzantine × Star conditions complete.")
    print(f"Results in {out_root}/")


if __name__ == "__main__":
    main()
