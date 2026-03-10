#!/usr/bin/env python3
"""
Run curriculum with neutral, static lessons (Claude bypass) to ablate lesson-content effects.
Stores results in results_rebuttal/curriculum_static/<curriculum_name>/trial_XX/.

Usage:
  python scripts/run_curriculum_static_lesson.py --config config/curriculum_full.json --trials 5
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.engine import CurriculumEngine


NEUTRAL_LESSON = (
    "Lesson: cooperate when payoffs favor group welfare; punish only clear free-riders; "
    "re-evaluate after each round instead of assuming defection."
)


def patch_claude(engine: CurriculumEngine):
    """Monkeypatch ClaudeLessonGenerator.generate_lesson to return a neutral string."""

    def _neutral_lesson(stage_results, stage_config, stage_num, previous_lessons=None):
        return NEUTRAL_LESSON

    engine.claude_generator.generate_lesson = _neutral_lesson  # type: ignore


def run_curriculum(config_path: Path, trials: int, out_root: Path):
    with open(config_path) as f:
        curriculum_config = json.load(f)

    engine = CurriculumEngine(curriculum_config=curriculum_config, trials=trials)
    patch_claude(engine)

    # Redirect results directory
    engine.results_dir = out_root / curriculum_config.get("name", "curriculum")
    engine.results_dir.mkdir(parents=True, exist_ok=True)

    engine.run_all_trials()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to curriculum JSON")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out", default="results_rebuttal/curriculum_static")
    args = parser.parse_args()

    run_curriculum(Path(args.config), args.trials, Path(args.out))


if __name__ == "__main__":
    main()
