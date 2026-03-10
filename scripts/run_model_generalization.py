#!/usr/bin/env python3
"""
Quick smoke tests for stronger models in heterogeneous 4-agent settings.
Runs:
  1) Stag Hunt (3 rounds)
  2) Iterated Public Goods Game with Punishment (IPGG+P, 10 rounds, multiplier 1.6)
  3) Optional communication variant for IPGG+P (cheap-talk channel)

Defaults: 5 trials each, agents = DeepSeek-R1, GPT-4.1-mini, Qwen2.5-72B, Llama-3.3-70B.
Stores results under results_rebuttal/model_generalization/{stag_hunt,ipgg_punish,ipgg_comm}/trial_XX/results.json
Skips trials that already have results.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.engine import GameEngine
from src.games import StagHuntGame, IteratedPublicGoodsGame, IteratedPublicGoodsGameWithCommunication


def parse_agents(agent_str: str):
    models = [m.strip() for m in agent_str.split(",") if m.strip()]
    agents = []
    for i, model in enumerate(models, 1):
        agents.append(
            {
                "name": f"Agent_{i}",
                "model": model,
                "model_family": model,
            }
        )
    return agents


def run_stag_hunt(out_dir: Path, agents, trials: int, workers: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for t in range(1, trials + 1):
        results_file = out_dir / f"trial_{t:02d}" / "results.json"
        if results_file.exists():
            print(f"[skip] stag_hunt trial {t:02d}")
            continue
        todo.append(t)

    def _run(trial_num: int):
        game = StagHuntGame(agents=agents, rounds=3)
        engine = GameEngine(game)
        results = engine.run()
        results["metadata"] = {
            "game": "StagHunt",
            "trial": trial_num,
            "timestamp": datetime.now().isoformat(),
            "agents": agents,
            "rounds": 3,
        }
        td = out_dir / f"trial_{trial_num:02d}"
        td.mkdir(exist_ok=True)
        with open(td / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run, t): t for t in todo}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                fut.result()
                print(f"[done] stag_hunt trial {t:02d}")
            except Exception as e:
                print(f"[fail] stag_hunt trial {t:02d}: {e}")


def run_ipgg(out_dir: Path, agents, trials: int, workers: int, multiplier: float, with_comm: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for t in range(1, trials + 1):
        results_file = out_dir / f"trial_{t:02d}" / "results.json"
        if results_file.exists():
            print(f"[skip] ipgg_{'comm' if with_comm else 'punish'} trial {t:02d}")
            continue
        todo.append(t)

    def _run(trial_num: int):
        if with_comm:
            game = IteratedPublicGoodsGameWithCommunication(agents=agents, rounds=10)
            game.multiplier = multiplier
        else:
            game = IteratedPublicGoodsGame(
                agents=agents,
                rounds=10,
                enable_punishment=True,
                multiplier=multiplier,
            )
        engine = GameEngine(game)
        results = engine.run()
        results["metadata"] = {
            "game": "IPGG_Comm" if with_comm else "IPGG_Punish",
            "trial": trial_num,
            "timestamp": datetime.now().isoformat(),
            "agents": agents,
            "rounds": 10,
            "multiplier": multiplier,
        }
        td = out_dir / f"trial_{trial_num:02d}"
        td.mkdir(exist_ok=True)
        with open(td / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run, t): t for t in todo}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                fut.result()
                print(f"[done] ipgg_{'comm' if with_comm else 'punish'} trial {t:02d}")
            except Exception as e:
                print(f"[fail] ipgg_{'comm' if with_comm else 'punish'} trial {t:02d}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agents",
        type=str,
        default="deepseek-ai/DeepSeek-R1,gpt-4.1-mini,Qwen/Qwen2.5-72B-Instruct,meta-llama/Llama-3.3-70B-Instruct",
        help="Comma-separated model IDs for the 4 agents (order matters).",
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--multiplier", type=float, default=1.6)
    parser.add_argument("--comm", action="store_true", help="Also run IPGG with communication (cheap talk)")
    parser.add_argument("--out", type=str, default="results_rebuttal/model_generalization")
    args = parser.parse_args()

    agents = parse_agents(args.agents)
    if len(agents) != 4:
        raise ValueError("Please provide exactly 4 models for 4 agents.")

    out_root = Path(args.out)
    run_stag_hunt(out_root / "stag_hunt", agents, args.trials, args.workers)
    run_ipgg(out_root / "ipgg_punish", agents, args.trials, args.workers, args.multiplier, with_comm=False)
    if args.comm:
        run_ipgg(out_root / "ipgg_comm", agents, args.trials, args.workers, args.multiplier, with_comm=True)


if __name__ == "__main__":
    main()
