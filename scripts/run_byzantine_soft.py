#!/usr/bin/env python3
"""
Experiment A-soft: Soft Byzantine Cheap Talk (probabilistic defection).

Adversarial agents always broadcast "stag" but defect with probability
--defect_prob (default 0.5) rather than always.  Tests whether deterministic
vs probabilistic deception matters.

Only runs n_adversaries=1, 30 trials.

Usage:
  python scripts/run_byzantine_soft.py                    # defaults
  python scripts/run_byzantine_soft.py --defect_prob 0.3 --trials 30
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
from games_lowercase import StagHuntWithCommunication


# ---------- agents (same heterogeneous cohort as the paper) ----------
# AGENTS = [
#     {"name": "Agent_1", "model": "mistralai/Mixtral-8x22B-Instruct-v0.1", "model_family": "Mixtral"},
#     {"name": "Agent_2", "model": "Qwen/Qwen2.5-72B-Instruct", "model_family": "Qwen"},
#     {"name": "Agent_3", "model": "meta-llama/Llama-3.3-70B-Instruct", "model_family": "Llama"},
#     {"name": "Agent_4", "model": "deepseek-ai/DeepSeek-V3", "model_family": "DeepSeek"},
# ]

# for reference: we have the following keys: DEEPINFRA_API_KEY, ANTHROPIC_API_KEY, and OPENAI_API_KEY 

# add GPT and Claude Sonnet instead of LLaMA and DeepSeek 
AGENTS = [
    {"name": "Agent_1", "model": "mistralai/Mixtral-8x22B-Instruct-v0.1", "model_family": "Mixtral"},
    {"name": "Agent_2", "model": "Qwen/Qwen2.5-72B-Instruct", "model_family": "Qwen"},
    {"name": "Agent_3", "model": "gpt-4o", "model_family": "GPT-4o"},
    {"name": "Agent_4", "model": "claude-sonnet-4-6", "model_family": "Claude Sonnet"},
]


AGENT_FAMILY = {a["name"]: a["model_family"] for a in AGENTS}


def patch_soft_byzantine(engine: GameEngine, game: StagHuntWithCommunication,
                         adversary_names: Set[str], defect_prob: float):
    """
    Monkey-patch engine.call_agent so adversarial agents:
      - Communication phase: always broadcast "stag"
      - Action phase: choose "Hunt Hare" with probability defect_prob,
                      "Hunt Stag" otherwise
    """
    if not adversary_names:
        return

    original_call = engine.call_agent

    def patched_call(agent_name, model, prompt, agent_config=None, max_retries=3):
        if agent_name in adversary_names:
            if game.stage == "communication":
                return {
                    "reasoning": "Soft Byzantine agent: always signal cooperation.",
                    "action": {"type": "communicate", "word": "stag"}
                }
            else:  # action stage
                defects = random.random() < defect_prob
                if defects:
                    return {
                        "reasoning": f"Soft Byzantine agent: defecting this round (p={defect_prob}).",
                        "action": {"choice": "Hunt Hare"}
                    }
                else:
                    return {
                        "reasoning": f"Soft Byzantine agent: cooperating this round (p={1-defect_prob}).",
                        "action": {"choice": "Hunt Stag"}
                    }
        return original_call(agent_name, model, prompt, agent_config, max_retries)

    engine.call_agent = patched_call  # type: ignore


def enrich_model_family(results: Dict[str, Any]):
    for round_data in results.get("rounds_data", []):
        for agent_name, decision in round_data.get("full_decisions", {}).items():
            if isinstance(decision, dict):
                decision["model_family"] = AGENT_FAMILY.get(agent_name, "Unknown")


def add_words_visible(results: Dict[str, Any]):
    """Broadcast topology: every agent sees all words."""
    for round_data in results.get("rounds_data", []):
        if "choices" not in round_data:
            continue
        communications = round_data.get("communications", {})
        for agent_name, decision in round_data.get("full_decisions", {}).items():
            if isinstance(decision, dict):
                decision["words_visible"] = [[a, w] for a, w in communications.items()]


def run_one_trial(trial_num: int, out_dir: Path, rounds: int,
                  defect_prob: float):
    """Run a single trial with one soft-Byzantine adversary."""
    all_names = [a["name"] for a in AGENTS]
    adversary_name = random.choice(all_names)
    adversary_names = {adversary_name}
    adversary_detail = [
        {"agent_id": adversary_name, "model_family": AGENT_FAMILY[adversary_name]}
    ]

    game = StagHuntWithCommunication(agents=AGENTS, rounds=rounds)
    engine = GameEngine(game)
    patch_soft_byzantine(engine, game, adversary_names, defect_prob)

    results = engine.run()

    results["total_rounds"] = rounds
    enrich_model_family(results)
    add_words_visible(results)

    # Record per-round adversary actions for analysis
    adv_actions = []
    for rd in results.get("rounds_data", []):
        if "choices" not in rd:
            continue
        choice = rd["choices"].get(adversary_name, "")
        adv_actions.append({"round": rd["round"], "choice": choice})

    results["metadata"] = {
        "experiment": "byzantine_soft",
        "n_adversaries": 1,
        "defect_prob": defect_prob,
        "adversary_agents": [adversary_name],
        "adversary_detail": adversary_detail,
        "adversary_actions": adv_actions,
        "trial": trial_num,
        "rounds": rounds,
        "agents": AGENTS,
        "timestamp": datetime.now().isoformat(),
    }

    trial_dir = out_dir / f"trial_{trial_num:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    with open(trial_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return trial_num


def run_all(trials: int, out_dir: Path, workers: int, rounds: int,
            defect_prob: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for t in range(1, trials + 1):
        if (out_dir / f"trial_{t:02d}" / "results.json").exists():
            print(f"  [skip] trial {t:02d} already exists")
            continue
        todo.append(t)
    if not todo:
        print(f"  All {trials} trials already complete")
        return
    print(f"  Running {len(todo)} trials (defect_prob={defect_prob}, workers={workers})")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one_trial, t, out_dir, rounds, defect_prob): t
            for t in todo
        }
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                fut.result()
                print(f"  [done] trial {t:02d}")
            except Exception as e:
                print(f"  [FAIL] trial {t:02d}: {e}")


def generate_csv(out_dir: Path, trials: int):
    csv_path = out_dir / "all_results.csv"
    fieldnames = [
        "trial_id", "condition", "round", "agent_id", "model_family",
        "word_broadcast", "action", "payoff", "reasoning", "is_adversary",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in range(1, trials + 1):
            results_file = out_dir / f"trial_{t:02d}" / "results.json"
            if not results_file.exists():
                continue
            with open(results_file) as rf:
                results = json.load(rf)
            metadata = results.get("metadata", {})
            adversary_set = set(metadata.get("adversary_agents", []))
            defect_prob = metadata.get("defect_prob", 0.5)
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
                communications = act.get("communications", comm.get("communications", {}))
                choices = act.get("choices", {})
                payoffs = act.get("payoffs", {})
                comm_decisions = comm.get("full_decisions", {})
                act_decisions = act.get("full_decisions", {})
                for agent in [a["name"] for a in AGENTS]:
                    comm_reason = (
                        comm_decisions.get(agent, {}).get("reasoning", "")
                        if isinstance(comm_decisions.get(agent), dict) else ""
                    )
                    act_reason = (
                        act_decisions.get(agent, {}).get("reasoning", "")
                        if isinstance(act_decisions.get(agent), dict) else ""
                    )
                    writer.writerow({
                        "trial_id": t,
                        "condition": f"soft_p{defect_prob}",
                        "round": rn,
                        "agent_id": agent,
                        "model_family": AGENT_FAMILY.get(agent, "Unknown"),
                        "word_broadcast": communications.get(agent, ""),
                        "action": choices.get(agent, ""),
                        "payoff": payoffs.get(agent, 0),
                        "reasoning": f"COMM: {comm_reason} | ACTION: {act_reason}",
                        "is_adversary": agent in adversary_set,
                    })
    print(f"  CSV written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment A-soft: Soft Byzantine Cheap Talk")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--defect_prob", type=float, default=0.5,
                        help="Probability adversary defects each round (default 0.5)")
    parser.add_argument("--out", type=str, default="results/byzantine_soft")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers")
    args = parser.parse_args()

    out_dir = Path(args.out)

    print(f"{'='*50}")
    print(f"Soft Byzantine: n_adv=1, defect_prob={args.defect_prob}")
    print(f"{'='*50}")
    run_all(args.trials, out_dir, args.workers, args.rounds, args.defect_prob)

    print(f"\n{'='*50}")
    print("Generating CSV...")
    generate_csv(out_dir, args.trials)

    print(f"\nSoft Byzantine experiment complete. Results in {out_dir}/")


if __name__ == "__main__":
    main()
