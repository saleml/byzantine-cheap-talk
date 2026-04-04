# Byzantine Cheap Talk and Communication Topology in LLM-Mediated Coordination Games

Code and data for the NETYS 2026 submission. We study how adversarial agents (Byzantine cheap talk) and restricted communication topologies affect cooperation in a 4-player Stag Hunt with pre-play communication, using heterogeneous LLM agents.

## Setup

```bash
git clone <repo-url> && cd gameth_llm_tmp
pip install -r requirements.txt
```

Create a `.env` file with your API keys:

```
DEEPINFRA_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here        # needed for GPT-4o in v2/v3
ANTHROPIC_API_KEY=your_key_here     # needed for Claude Sonnet in v2
```


Before running any scripts, load the environment variables into your terminal session:
```
source .env
```

## Model Versions

All experiment scripts take a `--version` flag that selects the model cohort:

| Version | Models | Notes |
|---------|--------|-------|
| `v1` | Mixtral-8x22B, Qwen2.5-72B, Llama-3.3-70B, DeepSeek-V3 | Original cohort (all via DeepInfra). Used for NETYS paper. |
| `v2` | Mixtral-8x22B, Qwen2.5-72B, GPT-4o, Claude Sonnet | Replaced Llama+DeepSeek. Claude Sonnet had severe output schema bugs (73-97% malformed). |
| `v3` | Mixtral-8x22B, Qwen2.5-72B, GPT-4o, DeepSeek-V3 | Replaced Claude Sonnet with DeepSeek. Current cohort for ongoing experiments. |

Model cohorts are defined in `scripts/config.py`.

## Experiments

All scripts require `--version v1|v2|v3`. Results go to `results/{experiment}_{version}/`.

| Script | Description | Key flags | Example |
|--------|-------------|-----------|---------|
| `run_byzantine.py` | Hard Byzantine (k=0,1,2) | `--version`, `--lowercase`, `--trials`, `--rounds` | `python scripts/run_byzantine.py --version v1 --trials 10` |
| `run_byzantine_soft.py` | Soft Byzantine | `--version`, `--defect_prob`, `--trials` | `python scripts/run_byzantine_soft.py --version v3 --defect_prob 0.3` |
| `run_topology.py` | Explicit topology | `--version`, `--lowercase`, `--trials` | `python scripts/run_topology.py --version v1 --trials 10` |
| `run_topology_silent.py` | Silent topology | `--version`, `--lowercase`, `--trials` | `python scripts/run_topology_silent.py --version v1` |
| `run_byzantine_star.py` | Byzantine × star | `--version`, `--dry-run`, `--trials` | `python scripts/run_byzantine_star.py --version v1 --dry-run` |

### Analysis

```bash
python scripts/analyze_results.py --version v1    # analyze v1 results
python scripts/analyze_results.py --version v3    # analyze v3 results
python scripts/generate_summary_figures.py        # generate figures
python scripts/check_output_format.py             # diagnose malformed outputs
```

Reports are saved to `results/analysis_report_{version}.txt`.

## Results Directory

Results are organized as `results/{experiment}_{version}/`:

```
results/
  byzantine_v1/          # Hard Byzantine, v1 models (10 trials × 3 conditions)
  byzantine_v2/          # Hard Byzantine, v2 models
  byzantine_v3/          # Hard Byzantine, v3 models
  byzantine_soft_v2/     # Soft Byzantine, v2 models (15 trials)
  byzantine_soft_v3/     # Soft Byzantine, v3 models
  byzantine_star_v1/     # Byzantine × star, v1 models (10 trials × 2 conditions)
  topology_v1/           # Explicit topology, v1 models (10 trials × 3 topologies)
  topology_v2/           # Explicit topology, v2 models
  topology_v3/           # Explicit topology, v3 models
  topology_silent_v1/    # Silent topology, v1 models
  topology_silent_v2/    # Silent topology, v2 models
  topology_silent_v3/    # Silent topology, v3 models
  *_v2_lowercase/        # v2 models with lowercased communication words
  *_test/                # Quick smoke-test runs (3 trials)
  figures/               # Generated PDF/PNG figures
```

Each experiment directory contains per-trial `results.json` files and an `all_results.csv` summary.

**Note on the paper:** The NETYS 2026 paper reports v1 results (Mixtral, Qwen, Llama, DeepSeek). There is no `byzantine_soft_v1/` because soft Byzantine was first run with v2 models.

## Project Structure

```
src/
  engine.py              # GameEngine: runs games, handles multi-provider LLM API calls
  games.py               # Game implementations (StagHuntWithCommunication, etc.)
                         #   lowercase_comms=True flag for lowercase prompt variant
scripts/
  config.py              # Centralized version config (model cohorts, DP/CP families)
  run_byzantine.py       # Experiment A: hard Byzantine
  run_byzantine_soft.py  # Experiment A-soft: probabilistic Byzantine
  run_byzantine_star.py  # Experiment: Byzantine × star topology crossing
  run_topology.py        # Experiment B: explicit topology
  run_topology_silent.py # Experiment B-silent: silent topology
  analyze_results.py     # Comprehensive analysis (--version flag)
  generate_summary_figures.py  # Publication figures
  check_output_format.py # Diagnostic for malformed LLM outputs
paper_netys/             # NETYS 2026 paper (LaTeX)
paper_eacl/              # EACL 2026 paper (prior work)
```


