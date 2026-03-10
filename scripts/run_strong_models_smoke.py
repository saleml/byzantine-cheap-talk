#!/usr/bin/env python3
"""
Quick smoke tests with stronger models on Stag Hunt (communication) and IPGG+P (with comm).
Stores results under results_rebuttal/strong_models/<model_alias>/<game>/trial_XX/.

Usage:
  python scripts/run_strong_models_smoke.py --trials 5
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.engine import GameEngine
from src.games import StagHuntWithCommunication, IteratedPublicGoodsGameWithCommunication


MODELS = {
    "deepseek_r1": "deepseek-ai/DeepSeek-R1",
    "o3_mini": "gpt-4.1-mini",
    "gpt4o_mini": "gpt-4o-mini",
}


def make_agents(model_id: str):
    return [
        {"name": f"Agent_{i}", "model": model_id, "model_family": model_id.split("/")[-1]} 
        for i in range(1, 5)
    ]


def run_game(game_cls, game_name: str, model_alias: str, model_id: str, trials: int, out_root: Path):
    out_dir = out_root / model_alias / game_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in range(1, trials + 1):
        agents = make_agents(model_id)
        if game_name.startswith("ipgg"):
            game = game_cls(agents=agents, rounds=10)
        else:
            game = game_cls(agents=agents, rounds=3)

        engine = GameEngine(game)
        results = engine.run()
        results["metadata"] = {
            "model_id": model_id,
            "trial": t,
            "game": game_name,
            "timestamp": datetime.now().isoformat(),
        }

        trial_dir = out_dir / f"trial_{t:02d}"
        trial_dir.mkdir(exist_ok=True)
        with open(trial_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out", default="results_rebuttal/strong_models")
    args = parser.parse_args()

    out_root = Path(args.out)
    for alias, mid in MODELS.items():
        run_game(StagHuntWithCommunication, "stag_hunt_comm", alias, mid, args.trials, out_root)
        run_game(IteratedPublicGoodsGameWithCommunication, "ipgg_p_comm", alias, mid, args.trials, out_root)


if __name__ == "__main__":
    main()
