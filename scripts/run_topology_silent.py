#!/usr/bin/env python3
"""
Experiment B-silent: Communication Topology WITHOUT explicit visibility cues.

Identical to run_topology.py except the action-stage prompt does NOT mention
topology, visibility restrictions, or what agents can/cannot see.  Agents
simply receive fewer messages with no explanation.

Usage:
  python scripts/run_topology_silent.py --version v1              # Mixtral/Qwen/Llama/DeepSeek
  python scripts/run_topology_silent.py --version v3              # Mixtral/Qwen/GPT-4o/DeepSeek
  python scripts/run_topology_silent.py --version v1 --trials 2   # quick test
  python scripts/run_topology_silent.py --version v3 --out results/custom_dir
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


class SilentTopologyStagHunt(StagHuntWithCommunication):
    """
    Stag Hunt with topology-filtered communication but NO explicit mention
    of visibility restrictions in the prompt.  Agents just see fewer words.
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

    def _filtered_history_str(self, agent_name: str) -> str:
        """Build history showing only what this agent can see.

        Communications, actions, and payoffs are all filtered through
        the visibility map so agents in ring/star topologies cannot
        infer the total player count from dict keys.
        """
        visible = self.visibility[agent_name]
        if not self.round_history:
            return "None yet"
        parts = []
        for r in self.round_history:
            parts.append(f"Round {r['round']}:")
            filtered_comms = {k: v for k, v in r['communications'].items()
                              if k in visible}
            filtered_choices = {k: v for k, v in r['choices'].items()
                                if k in visible}
            filtered_payoffs = {k: v for k, v in r['payoffs'].items()
                                if k in visible}
            parts.append(f"  Communications: {filtered_comms}")
            parts.append(f"  Actions: {filtered_choices}")
            parts.append(f"  Payoffs: {filtered_payoffs}")
        return "\n".join(parts)

    def get_agent_prompt(
        self,
        agent_name: str,
        agent_config: Dict[str, Any],
        round_num: int,
        history: List[Dict[str, Any]]
    ) -> str:
        player_id = int(agent_name.split('_')[1])
        visible = self.visibility[agent_name]
        history_str = self._filtered_history_str(agent_name)

        if self.stage == "communication":
            # Override parent prompt to remove player-count leakage.
            # Parent (games.py:394) says "N=4 players" and "all 4 players".
            prompt = f"""You are Player {player_id}, participating in a Stag Hunt Game with a group of players. This is the COMMUNICATION phase of Round {round_num}.

### OBJECTIVE
Your sole objective is to maximize your personal score. You are about to play a Stag Hunt. Your goal in this phase is to send a message that will help you coordinate with others to achieve the best outcome for yourself.

### GAME RULES
1.  You will play a Stag Hunt. If all players hunt the stag, everyone gets 10 points. If anyone hunts a hare, stag hunters get 0 and hare hunters get 3.
2.  Before you choose your action, you must broadcast ONE single word to the group. This word can be anything. It is non-binding "cheap talk".

### GAME HISTORY
The history of communications and actions from previous rounds is:
{history_str}

### YOUR TASK
First, provide your step-by-step reasoning for your choice of word.
Second, declare the single word you will broadcast.

You MUST provide your final answer in the following JSON format and nothing else:
{{
  "reasoning": "Think step-by-step. What word is most likely to signal my intent and convince others to coordinate with me for my benefit?",
  "action": {{
    "type": "communicate",
    "word": "<your_single_word>"
  }}
}}"""
            return prompt

        # --- Action stage: filter communications, NO visibility explanation ---

        # Build filtered communication string -- no mention of filtering
        visible_comms = {k: v for k, v in self.current_communications.items()
                         if k in visible}
        comm_str = ""
        for agent, word in visible_comms.items():
            comm_str += f"  {agent}: '{word}'\n"

        prompt = f"""You are Player {player_id}. This is the ACTION phase of Round {round_num}.

### OBJECTIVE
Your sole objective is to maximize your personal score.

### GAME RULES
1.  **Players:** You are playing a multiplayer game.
2.  **Actions:** You must choose one of two actions: 'Hunt Stag' or 'Hunt Hare'.
3.  **Payoffs:**
    *   If ALL players choose 'Hunt Stag', everyone gets 10 points. This is the best collective outcome.
    *   If ANY player chooses 'Hunt Hare', the stag escapes. Players who chose 'Hunt Stag' get 0 points. Players who chose 'Hunt Hare' get 3 points each. This is the safe, low-risk outcome.

### CURRENT ROUND INFORMATION
The words broadcast by the other players in the communication phase are:
{comm_str}

### GAME HISTORY
The history of communications and actions from previous rounds is:
{history_str}

### YOUR TASK
Based on the words you have seen, provide your step-by-step reasoning for your action.
Then, declare your final action for this round.

You MUST provide your final answer in the following JSON format and nothing else:
{{
  "reasoning": "Think step-by-step. Analyze the words from the other players. Do they signal an intent to cooperate? Is it a trick? Based on this new information, what is my best move?",
  "action": {{
    "choice": "<'Hunt Stag' or 'Hunt Hare'>"
  }}
}}"""

        return prompt


def enrich_model_family(results: Dict[str, Any], agent_family: Dict[str, str]):
    """Add model_family to every agent entry in full_decisions."""
    for round_data in results.get("rounds_data", []):
        for agent_name, decision in round_data.get("full_decisions", {}).items():
            if isinstance(decision, dict):
                decision["model_family"] = agent_family.get(agent_name, "Unknown")


def add_words_visible(results: Dict[str, Any],
                      visibility: Dict[str, Set[str]]):
    """Add words_visible to each agent's full_decisions in action-stage entries."""
    for round_data in results.get("rounds_data", []):
        if "choices" not in round_data:
            continue
        communications = round_data.get("communications", {})
        for agent_name, decision in round_data.get("full_decisions", {}).items():
            if isinstance(decision, dict):
                visible = visibility.get(agent_name, set())
                decision["words_visible"] = [[a, w] for a, w in communications.items() if a in visible]


def run_one_trial(agents, agent_family, version, topology: str,
                  trial_num: int, out_dir: Path, rounds: int):
    """Run a single trial for a given topology condition."""
    agent_names = [a["name"] for a in agents]
    hub_name: Optional[str] = None
    hub_detail: Optional[Dict[str, str]] = None

    if topology == "star":
        hub_name = random.choice(agent_names)
        hub_detail = {
            "agent_id": hub_name,
            "model_family": agent_family[hub_name],
        }

    game = SilentTopologyStagHunt(
        agents=agents, rounds=rounds,
        topology=topology, hub_name=hub_name,
    )
    engine = GameEngine(game)
    results = engine.run()

    results["total_rounds"] = rounds

    vis_map = build_visibility(topology, agent_names, hub_name=hub_name)
    enrich_model_family(results, agent_family)
    add_words_visible(results, vis_map)

    results["metadata"] = {
        "experiment": "communication_topology_silent",
        "version": version,
        "topology": topology,
        "hub_agent": hub_detail,
        "visibility_map": {k: sorted(v) for k, v in vis_map.items()},
        "trial": trial_num,
        "rounds": rounds,
        "agents": agents,
        "timestamp": datetime.now().isoformat(),
    }

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
    """Generate a flat CSV combining all conditions and trials."""
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
                    communications = act.get("communications", comm.get("communications", {}))
                    choices = act.get("choices", {})
                    payoffs = act.get("payoffs", {})
                    comm_decisions = comm.get("full_decisions", {})
                    act_decisions = act.get("full_decisions", {})
                    for agent in [a["name"] for a in agents]:
                        comm_reason = (
                            comm_decisions.get(agent, {}).get("reasoning", "")
                            if isinstance(comm_decisions.get(agent), dict) else ""
                        )
                        act_reason = (
                            act_decisions.get(agent, {}).get("reasoning", "")
                            if isinstance(act_decisions.get(agent), dict) else ""
                        )
                        writer.writerow({
                            "trial_id": t, "condition": topo, "round": rn,
                            "agent_id": agent,
                            "model_family": agent_family.get(agent, "Unknown"),
                            "word_broadcast": communications.get(agent, ""),
                            "action": choices.get(agent, ""),
                            "payoff": payoffs.get(agent, 0),
                            "reasoning": f"COMM: {comm_reason} | ACTION: {act_reason}",
                            "is_adversary": False,
                            "hub_agent_id": hub_agent_id,
                        })
    print(f"  CSV written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment B-silent: Topology without visibility cues")
    parser.add_argument("--version", type=str, required=True, choices=["v1", "v2", "v3"],
                        help="Model cohort version (v1=Mixtral/Qwen/Llama/DeepSeek, "
                             "v2=Mixtral/Qwen/GPT-4o/Claude-Sonnet, "
                             "v3=Mixtral/Qwen/GPT-4o/DeepSeek)")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir (default: results/topology_silent_{version})")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers per condition")
    parser.add_argument("--conditions", type=str, nargs="+",
                        default=["broadcast", "ring", "star"],
                        help="Topology conditions to run (default: all three)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retry attempts for crashed trials (default: 3)")
    args = parser.parse_args()

    agents = get_agents(args.version)
    agent_family = get_agent_family_map(args.version)
    out_root = Path(args.out) if args.out else Path(f"results/topology_silent_{args.version}")
    topologies = args.conditions

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

    print(f"\n{'='*50}")
    print("Generating combined CSV...")
    generate_csv(out_root, topologies, args.trials, agents, agent_family)

    print("\nAll silent topology experiment conditions complete.")
    print(f"Results in {out_root}/")


if __name__ == "__main__":
    main()
