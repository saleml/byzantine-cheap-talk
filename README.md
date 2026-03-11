# Byzantine Cheap Talk and Communication Topology in LLM-Mediated Coordination Games

Code and data for the NETYS 2026 submission. We study how adversarial agents (Byzantine cheap talk) and restricted communication topologies affect cooperation in a 4-player Stag Hunt with pre-play communication, using four heterogeneous LLM agents (Mixtral-8x22B, Qwen2.5-72B, Llama-3.3-70B, DeepSeek-V3). Key findings: a single deterministic liar destroys group cooperation entirely; probabilistic deception is far more survivable; topology restrictions collapse coordination only when agents are explicitly told about visibility limits (meta-reasoning effect, not information loss).

## Setup

```bash
git clone <repo-url> && cd gameth_llm_tmp
pip install -r requirements.txt
```

Create a `.env` file with your DeepInfra API key:

```
DEEPINFRA_API_KEY=your_key_here
```

## Experiments

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/run_byzantine.py` | Hard Byzantine cheap talk (k=0,1,2 adversaries) | `python scripts/run_byzantine.py --trials 10 --workers 4` |
| `scripts/run_byzantine_soft.py` | Soft Byzantine (probabilistic defection, p=0.5) | `python scripts/run_byzantine_soft.py --trials 15 --workers 4` |
| `scripts/run_topology.py` | Explicit communication topology (broadcast/ring/star) | `python scripts/run_topology.py --trials 10 --workers 4` |
| `scripts/run_topology_silent.py` | Silent topology (same filtering, no visibility cues) | `python scripts/run_topology_silent.py --trials 10 --workers 4` |
| `scripts/generate_summary_figures.py` | Generate publication figures from results CSVs | `python scripts/generate_summary_figures.py` |
| `scripts/analyze_results.py` | Reproduces all paper statistics from raw CSVs | `python scripts/analyze_results.py` |

## Pre-computed Results

The `results/` directory contains pre-computed experiment outputs:

- `results/byzantine/` -- 10 trials each for k=0, k=1, k=2
- `results/byzantine_soft/` -- 15 trials, k=1 soft (p=0.5)
- `results/topology/` -- 10 trials each for broadcast, ring, star
- `results/topology_silent/` -- 10 trials each for broadcast, ring, star (silent)
- `results/figures/` -- Generated PDF/PNG figures

Each experiment directory contains per-trial JSON results and an `all_results.csv` summary.

## Project Structure

```
src/
  engine.py                 # GameEngine: runs games, handles LLM API calls
  games.py                  # Game implementations (StagHuntWithCommunication, etc.)
scripts/                    # Experiment runners and analysis (see table above)
results/                    # Raw trial data + figures
paper_netys/
  outline.md                # Paper outline with all results
```
