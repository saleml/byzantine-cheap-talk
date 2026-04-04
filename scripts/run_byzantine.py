#!/usr/bin/env python3
"""
Experiment A: Byzantine Cheap Talk in N-Player Stag Hunt

Runs a Stag Hunt with one-word cheap-talk communication, where some agents
may be Byzantine: they always broadcast "stag" in the communication phase
but always choose "Hunt Hare" in the action phase.

Each trial consists of multiple rounds (default 5).
Each round has two stages:
  1. communication: every agent broadcasts one word
  2. action: every agent chooses Hunt Stag or Hunt Hare

--------------------------------------------------
KEY FLAGS / TAGS
--------------------------------------------------

Model selection (choose exactly one):
  --version {v1,v2,v3}
      Use one predefined 4-model cohort from config.py:
        v1 = Mixtral, Qwen, Llama, DeepSeek
        v2 = Mixtral, Qwen, GPT-4o, Claude Sonnet
        v3 = Mixtral, Qwen, GPT-4o, DeepSeek

  --num_players N
      Ignore --version and instead use the first N agents from the
      ordered master model pool in config.py.
      Example:
        --num_players 5
      means use the first 5 models from MASTER_AGENT_POOL.

  NOTE:
    --version and --num_players are mutually exclusive.
    You must provide exactly one of them.

Adversary condition selection:
  --condition_set {k0,k1,k2,first2,all}
      k0      = run only adv_0 (no adversaries / baseline)
      k1      = run only adv_1 (one adversary)
      k2      = run only adv_2 (two adversaries)
      first2  = run adv_0 and adv_1
      all     = run adv_0, adv_1, and adv_2

Lowercasing option:
  --lowercase
      Lowercase communication words in the action-stage prompt
      before agents reason over them.
      This is useful for testing whether capitalization differences
      affect coordination behavior.
      If not mentioned, defaults to False (i.e. comm words appear as-is in prompts and logs).

Standard runtime flags:
  --trials T
      Number of trials per condition (default: 10)

  --rounds R
      Number of rounds per trial (default: 5)

  --workers W
      Number of parallel workers per condition (default: 2)

  --max-retries M
      Maximum retry attempts for crashed / missing trials (default: 3)

  --out PATH
      Output directory. If omitted, a default path is built automatically:
        results/byzantine_<version>
        results/byzantine_n<num_players>
      and appends _lowercase if --lowercase is used.

--------------------------------------------------
CONDITIONS
--------------------------------------------------

  adv_0
      Baseline condition: all agents are honest.

  adv_1
      One randomly chosen agent is Byzantine in each trial.

  adv_2
      Two randomly chosen agents are Byzantine in each trial.

Adversary assignment is randomized independently per trial.

--------------------------------------------------
EXAMPLES
--------------------------------------------------

  python scripts/run_byzantine.py --version v1
  python scripts/run_byzantine.py --version v3
  python scripts/run_byzantine.py --version v2 --lowercase
  python scripts/run_byzantine.py --version v3 --condition_set k1
  python scripts/run_byzantine.py --version v3 --condition_set first2
  python scripts/run_byzantine.py --num_players 5
  python scripts/run_byzantine.py --version v1 --trials 2
  python scripts/run_byzantine.py --version v3 --out results/custom_dir
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
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Load .env before importing engine
_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k, _v)

from src.engine import GameEngine
from src.games import StagHuntWithCommunication
# get_first_n_agents supports selecting the first N models from MASTER_AGENT_POOL.
from config import get_agents, get_agent_family_map, get_first_n_agents


def patch_byzantine(engine: GameEngine, game: StagHuntWithCommunication,
                    adversary_names: Set[str]):
    """
    Monkey-patch engine.call_agent so that adversarial agents skip the LLM
    and return a fixed deceptive strategy:
      - Communication phase: broadcast "stag"
      - Action phase: choose "Hunt Hare"
    """
    if not adversary_names:
        return

    original_call = engine.call_agent

    def patched_call(agent_name, model, prompt, agent_config=None, max_retries=3):
        if agent_name in adversary_names:
            if game.stage == "communication":
                return {
                    "reasoning": "Byzantine agent: always signal cooperation to deceive others.",
                    "action": {"type": "communicate", "word": "stag"}
                }
            else:
                return {
                    "reasoning": "Byzantine agent: always defect regardless of signals.",
                    "action": {"choice": "Hunt Hare"}
                }
        return original_call(agent_name, model, prompt, agent_config, max_retries)

    engine.call_agent = patched_call


def enrich_model_family(results: Dict[str, Any], agent_family: Dict[str, str]):
    """Add model_family to every agent entry in full_decisions."""
    for round_data in results.get("rounds_data", []):
        for agent_name, decision in round_data.get("full_decisions", {}).items():
            if isinstance(decision, dict):
                decision["model_family"] = agent_family.get(agent_name, "Unknown")


def add_words_visible(results: Dict[str, Any]):
    """Add words_visible to each agent's full_decisions in action-stage entries.

    Byzantine experiment uses broadcast topology: every agent sees all words.
    """
    for round_data in results.get("rounds_data", []):
        if "choices" not in round_data:
            continue
        communications = round_data.get("communications", {})
        for agent_name, decision in round_data.get("full_decisions", {}).items():
            if isinstance(decision, dict):
                decision["words_visible"] = [[a, w] for a, w in communications.items()]


def run_one_trial(agents, agent_family, version, n_adversaries, trial_num,
                  out_dir, rounds, lowercase=False):
    """Run a single trial for a given adversary count."""
    all_names = [a["name"] for a in agents]
    adversary_names_list = sorted(random.sample(all_names, n_adversaries))
    adversary_names = set(adversary_names_list)

    adversary_detail = [
        {"agent_id": name, "model_family": agent_family[name]}
        for name in adversary_names_list
    ]

    game = StagHuntWithCommunication(agents=agents, rounds=rounds,
                                     lowercase_comms=lowercase)
    engine = GameEngine(game)
    patch_byzantine(engine, game, adversary_names)

    results = engine.run()

    # Fix total_rounds: engine counts history entries (2x for two-stage games)
    results["total_rounds"] = rounds

    enrich_model_family(results, agent_family)
    add_words_visible(results)

    results["metadata"] = {
        "experiment": "byzantine_cheap_talk",
        "version": version,
        "n_adversaries": n_adversaries,
        "adversary_agents": adversary_names_list,
        "adversary_detail": adversary_detail,
        "trial": trial_num,
        "rounds": rounds,
        "agents": agents,
        "timestamp": datetime.now().isoformat(),
    }

    trial_dir = out_dir / f"trial_{trial_num:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    with open(trial_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return trial_num


def run_condition(agents, agent_family, version, n_adversaries, trials,
                  out_dir, workers, rounds, lowercase=False, max_retries=3):
    """Run all trials for one adversary-count condition.

    Automatically retries crashed trials (those that never saved results.json)
    up to max_retries times. Completed trials are always skipped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for retry in range(max_retries):
        todo = [t for t in range(1, trials + 1)
                if not (out_dir / f"trial_{t:02d}" / "results.json").exists()]

        if not todo:
            if retry == 0:
                print(f"  All {trials} trials already complete for n_adv={n_adversaries}")
            return

        label = f" (retry {retry})" if retry > 0 else ""
        print(f"  Running {len(todo)} trials for n_adv={n_adversaries}{label} (workers={workers})")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_one_trial, agents, agent_family, version,
                            n_adversaries, t, out_dir, rounds, lowercase): t
                for t in todo
            }
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    fut.result()
                    print(f"  [done] n_adv={n_adversaries} trial {t:02d}")
                except Exception as e:
                    print(f"  [FAIL] n_adv={n_adversaries} trial {t:02d}: {e}")

    # Final check
    still_missing = [t for t in range(1, trials + 1)
                     if not (out_dir / f"trial_{t:02d}" / "results.json").exists()]
    if still_missing:
        print(f"  WARNING: {len(still_missing)} trials still missing after {max_retries} "
              f"attempts: {still_missing}")


def generate_csv(out_root: Path, conditions: List[int], trials: int,
                 agents, agent_family):
    """Generate a flat CSV combining all conditions and trials.
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

                    for agent in [a["name"] for a in agents]:
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
                            "model_family": agent_family.get(agent, "Unknown"),
                            "word_broadcast": communications.get(agent, ""),
                            "action": choices.get(agent, ""),
                            "payoff": payoffs.get(agent, 0),
                            "reasoning": reasoning,
                            "is_adversary": agent in adversary_set,
                        })

    print(f"  CSV written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Experiment A: Byzantine Cheap Talk")
    # Enforce that exactly one model-selection mode is used.
    selector_group = parser.add_mutually_exclusive_group(required=True)
    # Version mode selects one predefined 4-model cohort.
    selector_group.add_argument("--version", type=str, choices=["v1", "v2", "v3"],
                                help="Model cohort version (v1=Mixtral/Qwen/Llama/DeepSeek, "
                                     "v2=Mixtral/Qwen/GPT-4o/Claude-Sonnet, "
                                     "v3=Mixtral/Qwen/GPT-4o/DeepSeek)")
    # Num-players mode selects the first N models from the ordered master pool.
    selector_group.add_argument("--num_players", type=int, default=None,
                                help="If set, ignore --version cohort and use first N agents "
                                     "from config MASTER_AGENT_POOL")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir (default: results/byzantine_{version})")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers per condition")
    parser.add_argument("--condition_set", type=str, default="all",
                        choices=["k0", "k1", "k2", "first2", "all"],
                        help="Which adversary conditions to run: "
                             "k0->adv_0 only, k1->adv_1 only, k2->adv_2 only, "
                             "first2->adv_0+adv_1, all->adv_0+adv_1+adv_2")
    parser.add_argument("--lowercase", action="store_true",
                        help="Lowercase communication words in action-stage prompts")
    # if --lowercase is mentioned in the command then it will be set to True, otherwise it will be False
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retry attempts for crashed trials (default: 3)")
    args = parser.parse_args()

    # In num-players mode, build agents dynamically from the master pool.
    if args.num_players is not None:
        # Select the first N configured models and assign Agent_1..Agent_N names.
        agents = get_first_n_agents(args.num_players)
        # Build name->family mapping for downstream metadata and CSV output.
        agent_family = {a["name"]: a["model_family"] for a in agents}
    else:
        # In version mode, use the fixed cohort defined for that version.
        agents = get_agents(args.version)
        # Keep existing family mapping behavior for versioned cohorts.
        agent_family = get_agent_family_map(args.version)

    # Use version-based output suffix unless num-players mode is active.
    suffix_core = f"_{args.version}" if args.num_players is None else f"_n{args.num_players}"
    # Preserve lowercase suffix behavior independent of selection mode.
    suffix = suffix_core + ("_lowercase" if args.lowercase else "")
    # Respect explicit --out; otherwise construct default output directory.
    out_root = Path(args.out) if args.out else Path(f"results/byzantine{suffix}")
    requested_conditions = {
        "k0": [0],
        "k1": [1],
        "k2": [2],
        "first2": [0, 1],
        "all": [0, 1, 2],
    }[args.condition_set]
    # Fail fast if requested adversary counts are infeasible for the selected player count.
    invalid_conditions = [n for n in requested_conditions if n >= len(agents)]
    if invalid_conditions:
        parser.error("Invalid condition selection for current player count: "
                     f"requested {[f'adv_{k}' for k in requested_conditions]} with "
                     f"{len(agents)} player(s). "
                     f"Infeasible: {[f'adv_{k}' for k in invalid_conditions]}.")
    conditions = requested_conditions

    # Version is None in num-players mode, so print a clear label.
    print(f"Version: {args.version if args.version is not None else 'N/A (using --num_players)'}")
    # Report final selected player count, including source mode when applicable.
    print(f"Num players: {len(agents)}" + (f" (from first {args.num_players} in master pool)"
                                         if args.num_players is not None else ""))
    print(f"Condition set: {args.condition_set} -> {[f'adv_{k}' for k in requested_conditions]}")
    print(f"Models: {[a['model_family'] for a in agents]}")
    print(f"Lowercase comms: {args.lowercase}")
    print(f"Output: {out_root}")

    for n_adv in conditions:
        print(f"\n{'='*50}")
        print(f"Condition: {n_adv} adversaries")
        print(f"{'='*50}")
        run_condition(agents, agent_family, args.version, n_adv,
                      args.trials, out_root / f"adv_{n_adv}",
                      args.workers, args.rounds, args.lowercase,
                      args.max_retries)

    print(f"\n{'='*50}")
    print("Generating combined CSV...")
    generate_csv(out_root, conditions, args.trials, agents, agent_family)

    print(f"\nAll Byzantine experiment conditions complete.")
    print(f"Results in {out_root}/")


if __name__ == "__main__":
    main()
