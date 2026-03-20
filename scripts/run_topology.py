#!/usr/bin/env python3
"""
Experiment B: Communication Topology in 4-Player Stag Hunt

Tests how restricting message visibility affects coordination.

Topology conditions:
  - broadcast: every agent sees all 4 words (baseline, same as paper)
  - ring:      each agent sees only its two neighbors' words
               Agent_i sees Agent_{i-1} and Agent_{i+1} (mod 4)
  - star:      one randomly-chosen hub agent sees all words;
               spokes see only the hub's word

Hub assignment is randomized per trial in the star condition.

Usage:
  python scripts/run_topology.py --version v1              # Mixtral/Qwen/Llama/DeepSeek
  python scripts/run_topology.py --version v3              # Mixtral/Qwen/GPT-4o/DeepSeek
  python scripts/run_topology.py --version v1 --trials 2   # quick test
  python scripts/run_topology.py --version v3 --out results/custom_dir
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
from typing import Dict, List, Any, Set, Optional

# Ensure repo root and src/ on path
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
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
from src.games import StagHuntWithCommunication
from config import get_agents, get_agent_family_map


def build_visibility(topology: str, agent_names: List[str],
                     hub_name: Optional[str] = None) -> Dict[str, Set[str]]:
    """
    Build a visibility map: for each agent, which agents' messages it can see
    (including its own).

    Args:
        topology: "broadcast", "ring", or "star"
        agent_names: ordered list of agent names
        hub_name: required for "star" -- the designated hub agent

    Returns:
        {agent_name: set of agent_names whose messages are visible}
    """
    n = len(agent_names)

    if topology == "broadcast":
        return {name: set(agent_names) for name in agent_names}

    elif topology == "ring":
        vis = {}
        for i, name in enumerate(agent_names):
            left = agent_names[(i - 1) % n]
            right = agent_names[(i + 1) % n]
            vis[name] = {name, left, right}
        return vis

    elif topology == "star":
        if hub_name is None:
            raise ValueError("hub_name required for star topology")
        vis = {hub_name: set(agent_names)}
        for name in agent_names:
            if name != hub_name:
                vis[name] = {name, hub_name}
        return vis

    else:
        raise ValueError(f"Unknown topology: {topology}")


class TopologyStagHunt(StagHuntWithCommunication):
    """
    Stag Hunt with Communication where message visibility is governed by
    a topology graph.  Only overrides the action-stage prompt to filter
    which communications each agent can see.
    """

    def __init__(self, agents: List[Dict[str, Any]], rounds: int = 5,
                 topology: str = "broadcast",
                 hub_name: Optional[str] = None):
        super().__init__(agents, rounds)
        self.topology = topology
        self.hub_name = hub_name
        self.visibility = build_visibility(
            topology, [a["name"] for a in agents], hub_name=hub_name
        )

    def get_agent_prompt(
        self,
        agent_name: str,
        agent_config: Dict[str, Any],
        round_num: int,
        history: List[Dict[str, Any]]
    ) -> str:
        """
        Override: in the action stage, filter communications to only those
        visible under the topology.  Communication stage is unchanged
        (every agent still broadcasts one word).
        """
        if self.stage == "communication":
            return super().get_agent_prompt(agent_name, agent_config,
                                            round_num, history)

        # --- Action stage: filter visible communications ---
        player_id = int(agent_name.split('_')[1])
        visible = self.visibility[agent_name]

        # Format history -- show only visible communications per round
        history_str = ""
        if self.round_history:
            for r in self.round_history:
                history_str += f"Round {r['round']}:\n"
                filtered_comms = {k: v for k, v in r['communications'].items()
                                  if k in visible}
                history_str += f"  Communications you saw: {filtered_comms}\n"
                history_str += f"  Actions: {r['choices']}\n"
                history_str += f"  Payoffs: {r['payoffs']}\n"
        else:
            history_str = "None yet"

        # Build filtered communication string for current round
        visible_comms = {k: v for k, v in self.current_communications.items()
                         if k in visible}
        comm_str = ""
        for agent, word in visible_comms.items():
            comm_str += f"  {agent}: '{word}'\n"

        n_total = len(self.agents)

        # Topology-aware visibility explanation
        if self.topology == "broadcast":
            vis_description = f"All {n_total} players' words are visible to you."
        elif self.topology == "ring":
            neighbors = sorted(visible - {agent_name})
            vis_description = (
                f"You can only see messages from your neighbors: "
                f"{', '.join(neighbors)}. Other players' messages are hidden."
            )
        elif self.topology == "star":
            if agent_name == self.hub_name:
                vis_description = (
                    "You are the central hub. You can see all players' messages."
                )
            else:
                vis_description = (
                    f"You can only see the hub player's ({self.hub_name}) message. "
                    f"Other players' messages are hidden from you."
                )
        else:
            vis_description = ""

        prompt = f"""You are Player {player_id}. This is the ACTION phase of Round {round_num}.

### OBJECTIVE
Your sole objective is to maximize your personal score.

### GAME RULES
1.  **Players:** There are {n_total} players in total.
2.  **Actions:** You must choose one of two actions: 'Hunt Stag' or 'Hunt Hare'.
3.  **Payoffs:**
    *   If ALL {n_total} players choose 'Hunt Stag', everyone gets 10 points. This is the best collective outcome.
    *   If ANY player chooses 'Hunt Hare', the stag escapes. Players who chose 'Hunt Stag' get 0 points. Players who chose 'Hunt Hare' get 3 points each. This is the safe, low-risk outcome.

### COMMUNICATION VISIBILITY
{vis_description}

### MESSAGES YOU CAN SEE
The words broadcast by visible players in the communication phase are:
{comm_str}

### GAME HISTORY
The history of communications and actions from previous rounds is:
{history_str}

### YOUR TASK
Based on the words you have seen (noting that you may NOT see all players' messages), provide your step-by-step reasoning for your action.
Then, declare your final action for this round.

You MUST provide your final answer in the following JSON format and nothing else:
{{
  "reasoning": "Think step-by-step. Analyze the words you can see. Consider what unseen players might have said. Is it safe to cooperate?",
  "action": {{
    "choice": "<'Hunt Stag' or 'Hunt Hare'>"
  }}
}}"""

        return prompt


def enrich_model_family(results: Dict[str, Any], agent_family: Dict[str, str]):
    """Add model_family to every agent entry in full_decisions throughout rounds_data."""
    for round_data in results.get("rounds_data", []):
        for agent_name, decision in round_data.get("full_decisions", {}).items():
            if isinstance(decision, dict):
                decision["model_family"] = agent_family.get(agent_name, "Unknown")


def add_words_visible(results: Dict[str, Any],
                      visibility: Dict[str, Set[str]]):
    """Add words_visible to each agent's full_decisions in action-stage entries."""
    for round_data in results.get("rounds_data", []):
        if "choices" not in round_data:  # skip communication-stage entries
            continue
        communications = round_data.get("communications", {})
        for agent_name, decision in round_data.get("full_decisions", {}).items():
            if isinstance(decision, dict):
                visible = visibility.get(agent_name, set())
                decision["words_visible"] = [[a, w] for a, w in communications.items() if a in visible]


def run_one_trial(agents, agent_family, version, topology: str,
                  trial_num: int, out_dir: Path, rounds: int):
    """Run a single trial for a given topology condition."""
    # Randomize hub for star topology
    agent_names = [a["name"] for a in agents]
    hub_name: Optional[str] = None
    hub_detail: Optional[Dict[str, str]] = None

    if topology == "star":
        hub_name = random.choice(agent_names)
        hub_detail = {
            "agent_id": hub_name,
            "model_family": agent_family[hub_name],
        }

    game = TopologyStagHunt(
        agents=agents, rounds=rounds,
        topology=topology, hub_name=hub_name,
    )
    engine = GameEngine(game)

    results = engine.run()

    # Fix total_rounds: engine counts history entries (2x for two-stage games)
    results["total_rounds"] = rounds

    # Build visibility map (needed for enrichment and metadata)
    vis_map = build_visibility(topology, agent_names, hub_name=hub_name)

    # Enrich reasoning traces with model_family and words_visible
    enrich_model_family(results, agent_family)
    add_words_visible(results, vis_map)

    # Attach experiment metadata
    results["metadata"] = {
        "experiment": "communication_topology",
        "version": version,
        "topology": topology,
        "hub_agent": hub_detail,
        "visibility_map": {k: sorted(v) for k, v in vis_map.items()},
        "trial": trial_num,
        "rounds": rounds,
        "agents": agents,
        "timestamp": datetime.now().isoformat(),
    }

    # Save
    trial_dir = out_dir / f"trial_{trial_num:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    with open(trial_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return trial_num


def run_condition(agents, agent_family, version, topology: str, trials: int,
                  out_dir: Path, workers: int, rounds: int, max_retries=3):
    """Run all trials for one topology condition.

    Automatically retries crashed trials (those that never saved results.json)
    up to max_retries times. Completed trials are always skipped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for retry in range(max_retries):
        todo = [t for t in range(1, trials + 1)
                if not (out_dir / f"trial_{t:02d}" / "results.json").exists()]

        if not todo:
            if retry == 0:
                print(f"  All {trials} trials already complete for {topology}")
            return

        label = f" (retry {retry})" if retry > 0 else ""
        print(f"  Running {len(todo)} trials for {topology}{label} (workers={workers})")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_one_trial, agents, agent_family, version,
                            topology, t, out_dir, rounds): t
                for t in todo
            }
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    fut.result()
                    print(f"  [done] {topology} trial {t:02d}")
                except Exception as e:
                    print(f"  [FAIL] {topology} trial {t:02d}: {e}")

    # Final check
    still_missing = [t for t in range(1, trials + 1)
                     if not (out_dir / f"trial_{t:02d}" / "results.json").exists()]
    if still_missing:
        print(f"  WARNING: {len(still_missing)} trials still missing after {max_retries} "
              f"attempts: {still_missing}")


def generate_csv(out_root: Path, topologies: List[str], trials: int,
                 agents, agent_family):
    """
    Generate a flat CSV combining all conditions and trials.
    One row per (trial, round, agent).
    """
    csv_path = out_root / "all_results.csv"
    fieldnames = [
        "trial_id", "condition", "round", "agent_id", "model_family",
        "word_broadcast", "action", "payoff", "reasoning",
        "is_adversary", "hub_agent_id",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for topo in topologies:
            cond_dir = out_root / topo
            for t in range(1, trials + 1):
                results_file = cond_dir / f"trial_{t:02d}" / "results.json"
                if not results_file.exists():
                    continue
                with open(results_file) as rf:
                    results = json.load(rf)

                metadata = results.get("metadata", {})
                hub_detail = metadata.get("hub_agent")
                hub_agent_id = hub_detail["agent_id"] if hub_detail else ""
                rounds_data = results.get("rounds_data", [])

                # Pair communication + action entries by round number
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

                    communications = act.get(
                        "communications", comm.get("communications", {})
                    )
                    choices = act.get("choices", {})
                    payoffs = act.get("payoffs", {})

                    comm_decisions = comm.get("full_decisions", {})
                    act_decisions = act.get("full_decisions", {})

                    for agent in [a["name"] for a in agents]:
                        comm_reason = (
                            comm_decisions.get(agent, {}).get("reasoning", "")
                            if isinstance(comm_decisions.get(agent), dict)
                            else ""
                        )
                        act_reason = (
                            act_decisions.get(agent, {}).get("reasoning", "")
                            if isinstance(act_decisions.get(agent), dict)
                            else ""
                        )
                        reasoning = f"COMM: {comm_reason} | ACTION: {act_reason}"

                        writer.writerow({
                            "trial_id": t,
                            "condition": topo,
                            "round": rn,
                            "agent_id": agent,
                            "model_family": agent_family.get(agent, "Unknown"),
                            "word_broadcast": communications.get(agent, ""),
                            "action": choices.get(agent, ""),
                            "payoff": payoffs.get(agent, 0),
                            "reasoning": reasoning,
                            "is_adversary": False,  # no adversaries in topology exp
                            "hub_agent_id": hub_agent_id,
                        })

    print(f"  CSV written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment B: Communication Topology")
    parser.add_argument("--version", type=str, required=True, choices=["v1", "v2", "v3"],
                        help="Model cohort version (v1=Mixtral/Qwen/Llama/DeepSeek, "
                             "v2=Mixtral/Qwen/GPT-4o/Claude-Sonnet, "
                             "v3=Mixtral/Qwen/GPT-4o/DeepSeek)")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir (default: results/topology_{version})")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers per condition")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retry attempts for crashed trials (default: 3)")
    args = parser.parse_args()

    agents = get_agents(args.version)
    agent_family = get_agent_family_map(args.version)
    out_root = Path(args.out) if args.out else Path(f"results/topology_{args.version}")
    topologies = ["broadcast", "ring", "star"]

    print(f"Version: {args.version}")
    print(f"Models: {[a['model_family'] for a in agents]}")
    print(f"Output: {out_root}")

    for topology in topologies:
        print(f"\n{'='*50}")
        print(f"Condition: {topology}")
        print(f"{'='*50}")
        run_condition(agents, agent_family, args.version, topology,
                      args.trials, out_root / topology,
                      args.workers, args.rounds, args.max_retries)

    # Generate combined flat CSV
    print(f"\n{'='*50}")
    print("Generating combined CSV...")
    generate_csv(out_root, topologies, args.trials, agents, agent_family)

    print("\nAll topology experiment conditions complete.")
    print(f"Results in {out_root}/")


if __name__ == "__main__":
    main()
