#!/usr/bin/env python3
"""
Experiment A: Byzantine Cheap Talk in 4-Player Stag Hunt

Adversarial agents always broadcast "stag" but always choose Hunt Hare.
Tests how deceptive communication undermines cooperation.

Conditions:
  - n_adversaries=0: baseline (all 4 agents are honest LLM players)
  - n_adversaries=1: one randomly-chosen agent is adversarial
  - n_adversaries=2: two randomly-chosen agents are adversarial

Adversary assignment is randomized per trial.

Usage:
  python scripts/run_byzantine.py                          # defaults: 10 trials, 5 rounds
  python scripts/run_byzantine.py --trials 30 --rounds 5
  python scripts/run_byzantine.py --trials 2 --workers 1   # quick smoke test
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
from typing import Dict, List, Any, Set

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


# ---------- agents (same heterogeneous cohort as the paper) ----------
AGENTS = [
    {"name": "Agent_1", "model": "mistralai/Mixtral-8x22B-Instruct-v0.1", "model_family": "Mixtral"},
    {"name": "Agent_2", "model": "Qwen/Qwen2.5-72B-Instruct", "model_family": "Qwen"},
    {"name": "Agent_3", "model": "meta-llama/Llama-3.3-70B-Instruct", "model_family": "Llama"},
    {"name": "Agent_4", "model": "deepseek-ai/DeepSeek-V3", "model_family": "DeepSeek"},
]

AGENT_FAMILY = {a["name"]: a["model_family"] for a in AGENTS}


def patch_byzantine(engine: GameEngine, game: StagHuntWithCommunication,
                    adversary_names: Set[str]):
    """
    Monkey-patch engine.call_agent so that adversarial agents skip the LLM
    and return a fixed deceptive strategy:
      - Communication phase: broadcast "stag"
      - Action phase: choose "Hunt Hare"
    """
    if not adversary_names:
        return  # nothing to patch

    original_call = engine.call_agent

    def patched_call(agent_name, model, prompt, agent_config=None, max_retries=3):
        if agent_name in adversary_names:
            if game.stage == "communication":
                return {
                    "reasoning": "Byzantine agent: always signal cooperation to deceive others.",
                    "action": {"type": "communicate", "word": "stag"}
                }
            else:  # action stage
                return {
                    "reasoning": "Byzantine agent: always defect regardless of signals.",
                    "action": {"choice": "Hunt Hare"}
                }
        return original_call(agent_name, model, prompt, agent_config, max_retries)

    engine.call_agent = patched_call  # type: ignore


def enrich_model_family(results: Dict[str, Any]):
    """Add model_family to every agent entry in full_decisions throughout rounds_data."""
    for round_data in results.get("rounds_data", []):
        for agent_name, decision in round_data.get("full_decisions", {}).items():
            if isinstance(decision, dict):
                decision["model_family"] = AGENT_FAMILY.get(agent_name, "Unknown")


def add_words_visible(results: Dict[str, Any]):
    """Add words_visible to each agent's full_decisions in action-stage entries.

    Byzantine experiment uses broadcast topology: every agent sees all words.
    """
    for round_data in results.get("rounds_data", []):
        if "choices" not in round_data:  # skip communication-stage entries
            continue
        communications = round_data.get("communications", {})
        for agent_name, decision in round_data.get("full_decisions", {}).items():
            if isinstance(decision, dict):
                decision["words_visible"] = [[a, w] for a, w in communications.items()]


def run_one_trial(n_adversaries: int, trial_num: int, out_dir: Path,
                  rounds: int):
    """Run a single trial for a given adversary count."""
    # Randomize adversary assignment: draw n_adversaries agents uniformly
    all_names = [a["name"] for a in AGENTS]
    adversary_names_list = sorted(random.sample(all_names, n_adversaries))
    adversary_names = set(adversary_names_list)

    # Build adversary detail list for metadata
    adversary_detail = [
        {"agent_id": name, "model_family": AGENT_FAMILY[name]}
        for name in adversary_names_list
    ]

    game = StagHuntWithCommunication(agents=AGENTS, rounds=rounds)
    engine = GameEngine(game)
    patch_byzantine(engine, game, adversary_names)

    results = engine.run()

    # Fix total_rounds: engine counts history entries (2x for two-stage games)
    results["total_rounds"] = rounds

    # Enrich reasoning traces with model_family and words_visible
    enrich_model_family(results)
    add_words_visible(results)

    # Attach experiment metadata
    results["metadata"] = {
        "experiment": "byzantine_cheap_talk",
        "n_adversaries": n_adversaries,
        "adversary_agents": adversary_names_list,
        "adversary_detail": adversary_detail,
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


def run_condition(n_adversaries: int, trials: int, out_dir: Path,
                  workers: int, rounds: int):
    """Run all trials for one adversary-count condition."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which trials still need to run
    todo = []
    for t in range(1, trials + 1):
        if (out_dir / f"trial_{t:02d}" / "results.json").exists():
            print(f"  [skip] n_adv={n_adversaries} trial {t:02d} already exists")
            continue
        todo.append(t)

    if not todo:
        print(f"  All {trials} trials already complete for n_adv={n_adversaries}")
        return

    print(f"  Running {len(todo)} trials for n_adv={n_adversaries} (workers={workers})")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one_trial, n_adversaries, t, out_dir, rounds): t
            for t in todo
        }
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                fut.result()
                print(f"  [done] n_adv={n_adversaries} trial {t:02d}")
            except Exception as e:
                print(f"  [FAIL] n_adv={n_adversaries} trial {t:02d}: {e}")


def generate_csv(out_root: Path, conditions: List[int], trials: int):
    """
    Generate a flat CSV combining all conditions and trials.
    One row per (trial, round, agent).
    """
    csv_path = out_root / "all_results.csv"
    fieldnames = [
        "trial_id", "condition", "round", "agent_id", "model_family",
        "word_broadcast", "action", "payoff", "reasoning", "is_adversary",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n_adv in conditions:
            cond_dir = out_root / f"adv_{n_adv}"
            for t in range(1, trials + 1):
                results_file = cond_dir / f"trial_{t:02d}" / "results.json"
                if not results_file.exists():
                    continue
                with open(results_file) as rf:
                    results = json.load(rf)

                metadata = results.get("metadata", {})
                adversary_set = set(metadata.get("adversary_agents", []))
                rounds_data = results.get("rounds_data", [])

                # Pair communication + action entries by round number.
                # Engine appends comm then action for each round.
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
                            "condition": f"adv_{n_adv}",
                            "round": rn,
                            "agent_id": agent,
                            "model_family": AGENT_FAMILY.get(agent, "Unknown"),
                            "word_broadcast": communications.get(agent, ""),
                            "action": choices.get(agent, ""),
                            "payoff": payoffs.get(agent, 0),
                            "reasoning": reasoning,
                            "is_adversary": agent in adversary_set,
                        })

    print(f"  CSV written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Experiment A: Byzantine Cheap Talk")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", type=str, default="results/byzantine")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers per condition")
    args = parser.parse_args()

    out_root = Path(args.out)
    conditions = [0, 1, 2]

    for n_adv in conditions:
        print(f"\n{'='*50}")
        print(f"Condition: {n_adv} adversaries")
        print(f"{'='*50}")
        run_condition(n_adv, args.trials, out_root / f"adv_{n_adv}",
                      args.workers, args.rounds)

    # Generate combined flat CSV
    print(f"\n{'='*50}")
    print("Generating combined CSV...")
    generate_csv(out_root, conditions, args.trials)

    print("\nAll Byzantine experiment conditions complete.")
    print(f"Results in {out_root}/")


if __name__ == "__main__":
    main()