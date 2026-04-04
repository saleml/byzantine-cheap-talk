# Byzantine Cheap Talk and Communication Topology in LLM-Mediated Coordination Games

Code and data for the NETYS 2026 paper experiments on LLM-mediated coordination in repeated Stag Hunt with communication.

This repository is organized as the **full experimental package reported in the paper** (code + results for all experiment families below).

## LLMs Used (Across Cohorts)

The experiments in this repository use the following model set (configured in `scripts/config.py`):

- `mistralai/Mixtral-8x22B-Instruct-v0.1` (Mixtral)
- `Qwen/Qwen2.5-72B-Instruct` (Qwen)
- `meta-llama/Llama-3.3-70B-Instruct` (Llama)
- `deepseek-ai/DeepSeek-V3` (DeepSeek)
- `gpt-4o` (GPT-4o)
- `claude-sonnet-4-6` (Claude Sonnet)

## Paper Scope (Full Experiment Set)

The paper covers the following experiment suite:

1. **Hard Byzantine**
- `k=0`, `k=1`, `k=2`
- 20 trials per `k`
- 4-player setting
- Additional archetype-stability analysis for `k=1` across `{2,3,5,6}` players (20 trials each)

2. **Soft Byzantine**
- `k=1`, `p=0.5`
- 20 trials

3. **Explicit Topology**
- topologies: `broadcast`, `ring`, `star`
- 20 trials per topology

4. **Silent Topology**
- topologies: `broadcast`, `ring`, `star`
- 20 trials per topology

5. **Byzantine × Star**
- star topology with one hard Byzantine adversary
- two conditions: `hub_is_adversary` vs `hub_is_honest`

## Repository Layout

```text
├── src/
│   ├── engine.py                 # Game engine + LLM API integration
│   └── games.py                  
├── scripts/
│   ├── config.py                 # Model cohorts and agent pools
│   ├── run_byzantine.py          # Hard Byzantine (k=0/1/2, supports N-player)
│   ├── run_byzantine_soft.py     # Soft Byzantine (k=1, probabilistic defect)
│   ├── run_topology.py           # Explicit topology (broadcast/ring/star)
│   ├── run_topology_silent.py    # Silent topology variant
│   ├── run_byzantine_star.py     # Byzantine × star crossing
│   ├── analyze_results.py        # Main analysis pipeline
│   ├── analyze_traces.py         # Trace-level analysis helpers
│   ├── generate_summary_figures.py
│   ├── heatmap.py
│   └── check_output_format.py
├── results/                      # Experiment outputs and analysis artifacts
├── paper_netys/
│   └── main.tex                  # NETYS paper source
├── requirements.txt
└── README.md
```

## Code → Result Mapping

Use these scripts for each paper experiment family:

1. `scripts/run_byzantine.py`
- Hard Byzantine (`k=0/1/2`, 4-player main setting)
- Also supports the `k=1` multi-group-size analysis (`N in {2,3,4,5,6}` via `--num_players`)
- Typical outputs example:
  - `results/byzantine_v1/` (or `v2`/`v3`)
  - `results/k1_5_groups/` (group-size analysis for k=1 and a group of 5 players)

2. `scripts/run_byzantine_soft.py`
- Soft Byzantine (`k=1`, `p=0.5`)
- Typical output: `results/byzantine_soft_v*/`

3. `scripts/run_topology.py`
- Explicit topology (`broadcast`, `ring`, `star`)
- Typical output: `results/topology_v*/`

4. `scripts/run_topology_silent.py`
- Silent topology (`broadcast`, `ring`, `star`)
- Typical output: `results/topology_silent_v*/`

5. `scripts/run_byzantine_star.py`
- Byzantine × star (`hub_is_adversary`, `hub_is_honest`)
- Typical output: `results/byzantine_star_v*/`

6. `scripts/analyze_results.py` and `scripts/generate_summary_figures.py`
- Consolidated reports/figures from experiment outputs

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` with provider keys (as needed by chosen model cohort):

```bash
DEEPINFRA_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

Load env vars before running:

```bash
source .env
```

## Model Cohorts

Cohorts are defined in `scripts/config.py`.

You can run experiments in two ways:

1. Use predefined cohort versions with `--version` (`v1`, `v2`, `v3`).
2. Define a custom player setup by selecting the first `N` models from the master pool with `--num_players N` (up to 6 players/models).

Predefined `--version` cohorts:

- `v1`: Mixtral-8x22B, Qwen2.5-72B, Llama-3.3-70B, DeepSeek-V3
- `v2`: Mixtral-8x22B, Qwen2.5-72B, GPT-4o, Claude Sonnet
- `v3`: Mixtral-8x22B, Qwen2.5-72B, GPT-4o, DeepSeek-V3

## Reproducing the Paper Experiment Set

Example commands (20-trial paper configuration):

```bash
# Hard Byzantine (4 players): k=0,1,2
python scripts/run_byzantine.py --version v1 --condition_set all --trials 20 --rounds 5

# k=1 archetype stability across N={2,3,4,5,6}
python scripts/run_byzantine.py --num_players 2 --condition_set k1 --trials 20 --rounds 5
python scripts/run_byzantine.py --num_players 3 --condition_set k1 --trials 20 --rounds 5
python scripts/run_byzantine.py --num_players 4 --condition_set k1 --trials 20 --rounds 5
python scripts/run_byzantine.py --num_players 5 --condition_set k1 --trials 20 --rounds 5
python scripts/run_byzantine.py --num_players 6 --condition_set k1 --trials 20 --rounds 5

# Soft Byzantine (k=1, p=0.5)
python scripts/run_byzantine_soft.py --version v1 --trials 20 --defect_prob 0.5 --rounds 5

# Explicit topology
python scripts/run_topology.py --version v1 --trials 20 --rounds 5

# Silent topology
python scripts/run_topology_silent.py --version v1 --trials 20 --rounds 5

# Byzantine × star
python scripts/run_byzantine_star.py --version v1 --trials 20 --rounds 5
```

## Analysis

```bash
python scripts/analyze_results.py --version v1
python scripts/analyze_results.py --version v2
python scripts/analyze_results.py --version v3
python scripts/analyze_results.py --num_of_players 2 3 4 5 6
python scripts/generate_summary_figures.py
python scripts/check_output_format.py
```

Outputs include CSV summaries, text reports, and figure assets under `results/`.

## Notes

- Main paper source is in `paper_netys/main.tex`.
- The repository is structured so experiment scripts, generated artifacts, and paper text remain directly cross-referenced.
