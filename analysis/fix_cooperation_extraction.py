#!/usr/bin/env python3
"""
Fix cooperation rate extraction by parsing round-level data
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Paths
RESULTS_DIR = Path("/home/salem.lahlou/salem/gameth/framework_exploration/results")
OUTPUT_DIR = Path("/home/salem.lahlou/salem/gameth/framework_exploration/analysis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_contributions_from_round(round_data):
    """Extract contribution amounts from round data"""
    contributions = []

    if 'full_decisions' in round_data:
        for agent, decision in round_data['full_decisions'].items():
            if 'action' in decision:
                action = decision['action']
                # Check for contribution amount
                if isinstance(action, dict) and 'amount' in action:
                    contributions.append(action['amount'])
                elif isinstance(action, dict) and 'type' in action and action['type'] == 'contribute':
                    if 'amount' in action:
                        contributions.append(action['amount'])

    return contributions

def extract_actions_from_round(round_data):
    """Extract actions for coordination games (Hunt Stag, Hunt Hare, etc.)"""
    actions = []

    if 'full_decisions' in round_data:
        for agent, decision in round_data['full_decisions'].items():
            if 'action' in decision:
                action = decision['action']
                if isinstance(action, dict) and 'choice' in action:
                    actions.append(action['choice'])
                elif isinstance(action, str):
                    actions.append(action)

    return actions

def compute_cooperation_rate_ipgg(stages_or_rounds_data, cooperation_threshold=15):
    """Compute cooperation rate for IPGG games (threshold on contributions)"""
    if not stages_or_rounds_data:
        return 0.0

    total_decisions = 0
    cooperative_decisions = 0

    for round_data in stages_or_rounds_data:
        contributions = extract_contributions_from_round(round_data)
        if contributions:
            total_decisions += len(contributions)
            cooperative_decisions += sum(1 for c in contributions if c >= cooperation_threshold)

    if total_decisions == 0:
        return 0.0

    return cooperative_decisions / total_decisions

def compute_cooperation_rate_coordination(stages_or_rounds_data, cooperative_actions=None):
    """Compute cooperation rate for coordination games"""
    if cooperative_actions is None:
        cooperative_actions = ['Hunt Stag', 'Opera', 'Football', 'Volunteer']

    if not stages_or_rounds_data:
        return 0.0

    total_decisions = 0
    cooperative_decisions = 0

    for round_data in stages_or_rounds_data:
        actions = extract_actions_from_round(round_data)
        if actions:
            total_decisions += len(actions)
            cooperative_decisions += sum(1 for a in actions if a in cooperative_actions)

    if total_decisions == 0:
        return 0.0

    return cooperative_decisions / total_decisions

def identify_game_type(stage_name, game_name):
    """Identify whether game is IPGG-type or coordination-type"""
    ipgg_keywords = ['public goods', 'ipgg', 'contribution']
    coordination_keywords = ['stag hunt', 'battle', 'volunteer']

    text = (stage_name + ' ' + game_name).lower()

    if any(kw in text for kw in ipgg_keywords):
        return 'ipgg'
    elif any(kw in text for kw in coordination_keywords):
        return 'coordination'
    else:
        return 'unknown'

def load_curriculum_with_fixed_cooperation():
    """Load curriculum results with properly computed cooperation rates"""
    curriculum_data = defaultdict(list)

    curriculum_dir = RESULTS_DIR / "curriculum"
    if not curriculum_dir.exists():
        return curriculum_data

    for condition_dir in curriculum_dir.iterdir():
        if not condition_dir.is_dir():
            continue

        condition_name = condition_dir.name

        for trial_dir in condition_dir.iterdir():
            if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                continue

            results_file = trial_dir / "results.json"
            if not results_file.exists():
                continue

            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)

                    # Recompute cooperation rates for each stage
                    if 'stages' in data:
                        for stage in data['stages']:
                            stage_name = stage.get('stage_name', '')
                            game_name = stage.get('game', '')
                            game_type = identify_game_type(stage_name, game_name)

                            if 'rounds_data' in stage and stage['rounds_data']:
                                if game_type == 'ipgg':
                                    stage['cooperation_rate_fixed'] = compute_cooperation_rate_ipgg(stage['rounds_data'])
                                elif game_type == 'coordination':
                                    stage['cooperation_rate_fixed'] = compute_cooperation_rate_coordination(stage['rounds_data'])
                                else:
                                    stage['cooperation_rate_fixed'] = stage.get('cooperation_rate', 0.0)

                                print(f"{condition_name}/{trial_dir.name}/{stage_name}: coop_rate={stage['cooperation_rate_fixed']:.3f}")

                    curriculum_data[condition_name].append(data)

            except Exception as e:
                print(f"Error loading {results_file}: {e}")

    return curriculum_data

def load_game_with_fixed_cooperation():
    """Load communication game results with fixed cooperation rates"""
    game_data = defaultdict(lambda: defaultdict(list))

    comm_games = ["battle_of_sexes", "ipgg_communication", "volunteers_dilemma"]

    for game in comm_games:
        game_dir = RESULTS_DIR / game
        if not game_dir.exists():
            continue

        for setting_dir in game_dir.iterdir():
            if not setting_dir.is_dir():
                continue

            setting_name = setting_dir.name

            for trial_dir in setting_dir.iterdir():
                if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                    continue

                exp_file = trial_dir / "experiment_data.json"
                if not exp_file.exists():
                    continue

                try:
                    with open(exp_file, 'r') as f:
                        data = json.load(f)

                        # Recompute cooperation rate
                        if 'rounds' in data:
                            game_type = 'ipgg' if 'ipgg' in game else 'coordination'

                            if game_type == 'ipgg':
                                coop_rate = compute_cooperation_rate_ipgg(data['rounds'])
                            else:
                                coop_rate = compute_cooperation_rate_coordination(data['rounds'])

                            # Store in metrics
                            if 'metrics' not in data:
                                data['metrics'] = {}
                            data['metrics']['cooperation_rate_fixed'] = coop_rate

                            print(f"{game}/{setting_name}/{trial_dir.name}: coop_rate={coop_rate:.3f}")

                        game_data[game][setting_name].append(data)

                except Exception as e:
                    print(f"Error loading {exp_file}: {e}")

    return game_data

def extract_target_task_metrics_fixed(curriculum_results):
    """Extract target task metrics using fixed cooperation rates"""
    metrics = defaultdict(lambda: defaultdict(list))

    for condition, trials in curriculum_results.items():
        for trial in trials:
            if 'stages' in trial and trial['stages']:
                # Use last stage (target task)
                target_stage = trial['stages'][-1]

                coop_rate = target_stage.get('cooperation_rate_fixed',
                                            target_stage.get('cooperation_rate', 0.0))

                metrics[condition]['cooperation_rate'].append(coop_rate)
                metrics[condition]['average_payoff'].append(
                    target_stage.get('average_payoff', 0.0)
                )

    return metrics

# [Rest of the visualization functions remain the same but use cooperation_rate_fixed]

def main():
    print("Loading and fixing cooperation rates...")

    curriculum_results = load_curriculum_with_fixed_cooperation()
    game_data = load_game_with_fixed_cooperation()

    print(f"\n✓ Loaded {len(curriculum_results)} curriculum conditions")
    print(f"✓ Loaded {len(game_data)} communication games")

    # Generate summary of fixed rates
    print("\n" + "="*80)
    print("FIXED COOPERATION RATES SUMMARY")
    print("="*80)

    for condition, trials in sorted(curriculum_results.items()):
        if trials:
            metrics = extract_target_task_metrics_fixed({condition: trials})
            if metrics[condition]['cooperation_rate']:
                coop_rates = metrics[condition]['cooperation_rate']
                print(f"{condition:40s}: {np.mean(coop_rates):.3f} ± {np.std(coop_rates):.3f} (n={len(coop_rates)})")

    print("="*80)

if __name__ == "__main__":
    main()
