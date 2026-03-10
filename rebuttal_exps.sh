#!/usr/bin/env bash
set -euo pipefail

# Source API keys
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

OUT_ROOT="results_rebuttal"

echo "Running IPGG baselines (comm vs plain)..."
python scripts/run_ipgg_baselines.py --trials 10 --out "${OUT_ROOT}/ipgg_baselines"

echo "Running curriculum lesson ablation (neutral lessons, full curriculum)..."
python scripts/run_curriculum_static_lesson.py --config config/curriculum_full.json --trials 5 --out "${OUT_ROOT}/curriculum_static"

echo "Running strong-model smoke tests (stag_hunt_comm + ipgg_p_comm)..."
python scripts/run_strong_models_smoke.py --trials 5 --out "${OUT_ROOT}/strong_models"

echo "Running cheap-talk IPGG+P done above; plain vs comm already handled."

echo "Adding +10 trials for control/full/direct precursor (fresh path)..."
python scripts/run_extra_trials.py --trials 10 --out "${OUT_ROOT}/extra_trials"

echo "All rebuttal experiments launched. Outputs in ${OUT_ROOT}/"
