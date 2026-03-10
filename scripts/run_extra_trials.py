#!/usr/bin/env python3
"""
Add +10 trials for control_group, full_curriculum, direct_precursor using existing configs.
Stores in results_rebuttal/extra_trials/<condition>/trial_XX/.

Usage:
  python scripts/run_extra_trials.py --trials 10
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.engine import CurriculumEngine


CONDITIONS = {
    "control_group": "config/control_group.json",
    "full_curriculum": "config/curriculum_full.json",
    "direct_precursor": "config/curriculum_direct_precursor.json",
}


def run_condition(name: str, config_path: str, trials: int, out_root: Path):
    with open(config_path) as f:
        cfg = json.load(f)

    engine = CurriculumEngine(curriculum_config=cfg, trials=trials)

    engine.results_dir = out_root / name
    engine.results_dir.mkdir(parents=True, exist_ok=True)

    engine.run_all_trials()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--out", default="results_rebuttal/extra_trials")
    args = parser.parse_args()

    out_root = Path(args.out)
    for name, path in CONDITIONS.items():
        run_condition(name, path, args.trials, out_root)


if __name__ == "__main__":
    main()
