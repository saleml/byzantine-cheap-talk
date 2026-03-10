#!/usr/bin/env python3
"""
Run IPGG+Punishment baselines with and without one-word communication.
Results are stored under results_rebuttal/ipgg_baselines/<condition>/trial_XX/.

Usage:
  python scripts/run_ipgg_baselines.py --trials 10

Conditions:
  - plain: IteratedPublicGoodsGame (10 rounds, punishment on)
  - comm:  IteratedPublicGoodsGameWithCommunication (same params)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.engine import GameEngine
from src.games import IteratedPublicGoodsGame, IteratedPublicGoodsGameWithCommunication


def make_agents(model: str = None):
    if model is None or str(model).lower() == "none":
        return [
            {"name": "Agent_1", "model": "mistralai/Mixtral-8x22B-Instruct-v0.1", "model_family": "Mixtral"},
            {"name": "Agent_2", "model": "Qwen/Qwen2.5-72B-Instruct", "model_family": "Qwen"},
            {"name": "Agent_3", "model": "meta-llama/Llama-3.3-70B-Instruct", "model_family": "Llama"},
            {"name": "Agent_4", "model": "deepseek-ai/DeepSeek-V3", "model_family": "DeepSeek"},
        ]
    return [
        {"name": f"Agent_{i}", "model": model, "model_family": model} for i in range(1, 5)
    ]

def maybe_patch_mock(engine: GameEngine, rounds: int):
    """If QUICK_MOCK is set, bypass API and emit deterministic cooperative actions."""
    if not os.environ.get("QUICK_MOCK"):
        return

    def _mock_call(agent_name, model, prompt, agent_config=None, max_retries=3):
        # Contribution 15, punish none; for comm phase, word 'cooperate'
        if "communicate" in prompt.lower():
            return {"reasoning": "mock", "action": {"type": "communicate", "word": "cooperate"}}
        if "punish" in prompt.lower():
            return {"reasoning": "mock", "action": {"type": "punish", "targets": []}}
        return {"reasoning": "mock", "action": {"type": "contribute", "amount": 15}}
    engine.call_agent = _mock_call  # type: ignore


def run_condition(condition: str, trials: int, out_dir: Path, rounds: int, multiplier: float, workers: int, model_override: str = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    agents = make_agents(model_override)
    # Collect remaining trials
    todo = []
    for t in range(1, trials + 1):
        trial_dir = out_dir / f"trial_{t:02d}"
        results_file = trial_dir / "results.json"
        if results_file.exists():
            print(f"[skip] {condition} trial {t:02d} already exists")
            continue
        todo.append(t)

    def run_one(trial_num: int):
        if condition == "plain":
            game = IteratedPublicGoodsGame(agents=agents, rounds=rounds, enable_punishment=True, multiplier=multiplier)
        elif condition == "comm":
            game = IteratedPublicGoodsGameWithCommunication(agents=agents, rounds=rounds)
            game.multiplier = multiplier  # override default
        else:
            raise ValueError(f"Unknown condition {condition}")

        engine = GameEngine(game)
        maybe_patch_mock(engine, rounds)
        results = engine.run()
        results["metadata"] = {
            "condition": condition,
            "trial": trial_num,
            "timestamp": datetime.now().isoformat(),
            "agents": agents,
            "rounds": rounds,
        }

        trial_dir = out_dir / f"trial_{trial_num:02d}"
        trial_dir.mkdir(exist_ok=True)
        with open(trial_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        return trial_num

    # parallel execution (configurable workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_one, t): t for t in todo}
        for fut in as_completed(futures):
            tdone = futures[fut]
            try:
                fut.result()
                print(f"[done] {condition} trial {tdone:02d}")
            except Exception as e:
                print(f"[fail] {condition} trial {tdone:02d}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--out", type=str, default="results_rebuttal/ipgg_baselines")
    parser.add_argument("--model", type=str, default=None, help="Override model for all agents")
    parser.add_argument("--multiplier", type=float, default=1.6, help="Public goods multiplier")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers per condition")
    args = parser.parse_args()

    out_root = Path(args.out)
    run_condition("plain", args.trials, out_root / "plain", args.rounds, args.multiplier, args.workers, args.model)
    run_condition("comm", args.trials, out_root / "comm", args.rounds, args.multiplier, args.workers, args.model)


if __name__ == "__main__":
    main()
