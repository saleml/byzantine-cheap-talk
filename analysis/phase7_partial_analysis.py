#!/usr/bin/env python3
"""
Phase 7 Partial Results Analysis
Generates figures and tables from available trial data
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

def load_curriculum_results():
    """Load all curriculum trial results"""
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
                    curriculum_data[condition_name].append(data)
            except Exception as e:
                print(f"Error loading {results_file}: {e}")

    return curriculum_data

def load_game_results():
    """Load communication game results"""
    game_data = defaultdict(lambda: defaultdict(list))

    # Games with communication
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
                        game_data[game][setting_name].append(data)
                except Exception as e:
                    print(f"Error loading {exp_file}: {e}")

    return game_data

def extract_target_task_metrics(curriculum_results):
    """Extract target task (final stage) metrics from curriculum results"""
    metrics = defaultdict(lambda: defaultdict(list))

    for condition, trials in curriculum_results.items():
        for trial in trials:
            # Try to get final metrics from top level first
            if 'final_cooperation_rate' in trial and 'final_avg_payoff' in trial:
                metrics[condition]['cooperation_rate'].append(
                    trial.get('final_cooperation_rate', 0.0)
                )
                metrics[condition]['average_payoff'].append(
                    trial.get('final_avg_payoff', 0.0)
                )
            # Otherwise, extract from stages list
            elif 'stages' in trial and trial['stages']:
                # Use the last stage (target task)
                target_stage = trial['stages'][-1]
                metrics[condition]['cooperation_rate'].append(
                    target_stage.get('cooperation_rate', 0.0)
                )
                metrics[condition]['average_payoff'].append(
                    target_stage.get('average_payoff', 0.0)
                )

    return metrics

def create_curriculum_comparison_figure(curriculum_results):
    """Create figure comparing curriculum conditions on target task"""
    metrics = extract_target_task_metrics(curriculum_results)

    if not metrics:
        print("No curriculum metrics available")
        return

    # Prepare data for plotting
    conditions = []
    coop_rates = []
    payoffs = []
    n_trials = []

    for condition, vals in metrics.items():
        if vals['cooperation_rate']:
            conditions.append(condition.replace('_', ' ').title())
            coop_rates.append(vals['cooperation_rate'])
            payoffs.append(vals['average_payoff'])
            n_trials.append(len(vals['cooperation_rate']))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Cooperation Rate
    positions = range(len(conditions))
    bp1 = ax1.boxplot(coop_rates, positions=positions, widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))

    ax1.set_xticks(positions)
    ax1.set_xticklabels(conditions, rotation=45, ha='right')
    ax1.set_ylabel('Cooperation Rate (Target Task)', fontsize=12, fontweight='bold')
    ax1.set_title('Cooperation Rate on Target Task by Curriculum', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3)

    # Add n= annotations
    for i, n in enumerate(n_trials):
        ax1.text(i, -0.1, f'n={n}', ha='center', fontsize=9, style='italic')

    # Average Payoff
    bp2 = ax2.boxplot(payoffs, positions=positions, widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))

    ax2.set_xticks(positions)
    ax2.set_xticklabels(conditions, rotation=45, ha='right')
    ax2.set_ylabel('Average Payoff (Target Task)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Payoff on Target Task by Curriculum', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add n= annotations
    for i, n in enumerate(n_trials):
        ax2.text(i, ax2.get_ylim()[0] - 5, f'n={n}', ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'curriculum_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'curriculum_comparison.png'}")
    plt.close()

def create_communication_game_figure(game_data):
    """Create figure showing communication game results"""
    if not game_data:
        print("No communication game data available")
        return

    # Map game names to readable labels
    game_labels = {
        'battle_of_sexes': 'Battle of the Sexes',
        'ipgg_communication': 'IPGG + Communication',
        'volunteers_dilemma': "Volunteer's Dilemma"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (game, settings) in enumerate(sorted(game_data.items())):
        ax = axes[idx]

        # Extract cooperation rates
        setting_names = []
        coop_data = []
        n_trials = []

        for setting, trials in sorted(settings.items()):
            if not trials:
                continue

            setting_label = "Heterogeneous" if "heterogeneous" in setting else "Homogeneous"
            setting_names.append(setting_label)

            # Extract cooperation/success rate
            rates = []
            for trial in trials:
                if 'metrics' in trial and 'cooperation_rate' in trial['metrics']:
                    rates.append(trial['metrics']['cooperation_rate'])
                elif 'summary' in trial and 'cooperation_rate' in trial['summary']:
                    rates.append(trial['summary']['cooperation_rate'])

            coop_data.append(rates)
            n_trials.append(len(rates))

        if coop_data:
            positions = range(len(setting_names))
            bp = ax.boxplot(coop_data, positions=positions, widths=0.5, patch_artist=True,
                           boxprops=dict(facecolor='salmon', alpha=0.7),
                           medianprops=dict(color='darkred', linewidth=2))

            ax.set_xticks(positions)
            ax.set_xticklabels(setting_names)
            ax.set_ylabel('Cooperation Rate', fontsize=11, fontweight='bold')
            ax.set_title(game_labels.get(game, game), fontsize=13, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)

            # Add n= annotations
            for i, n in enumerate(n_trials):
                ax.text(i, -0.1, f'n={n}', ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'communication_games_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'communication_games_comparison.png'}")
    plt.close()

def create_stage_progression_figure(curriculum_results):
    """Create figure showing cooperation progression through curriculum stages"""

    # Focus on key curricula
    key_curricula = [
        'communication_only_curriculum',
        'success_driven_curriculum',
        'cooperation_first_curriculum',
        'control_group'
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, curriculum in enumerate(key_curricula):
        if idx >= len(axes) or curriculum not in curriculum_results:
            continue

        ax = axes[idx]
        trials = curriculum_results[curriculum]

        # Collect stage-by-stage cooperation rates
        stage_data = defaultdict(list)

        for trial in trials:
            if 'stages' not in trial:
                continue

            for stage_idx, stage in enumerate(trial['stages']):
                stage_num = stage.get('stage', stage_idx + 1)
                coop_rate = stage.get('cooperation_rate', 0.0)
                stage_data[stage_num].append(coop_rate)

        if stage_data:
            stages = sorted(stage_data.keys())
            means = [np.mean(stage_data[s]) for s in stages]
            stds = [np.std(stage_data[s]) for s in stages]

            ax.plot(stages, means, marker='o', linewidth=2, markersize=8, label='Mean')
            ax.fill_between(stages,
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.3)

            ax.set_xlabel('Stage Number', fontsize=11, fontweight='bold')
            ax.set_ylabel('Cooperation Rate', fontsize=11, fontweight='bold')
            ax.set_title(curriculum.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

            # Add n= annotation
            n = len(trials)
            ax.text(0.02, 0.98, f'n={n} trials', transform=ax.transAxes,
                   verticalalignment='top', fontsize=9, style='italic',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'stage_progression.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'stage_progression.png'}")
    plt.close()

def create_summary_table(curriculum_results, game_data):
    """Create summary statistics table"""

    rows = []

    # Curriculum conditions
    for condition, trials in sorted(curriculum_results.items()):
        metrics = extract_target_task_metrics({condition: trials})

        if condition in metrics and metrics[condition]['cooperation_rate']:
            coop_rates = metrics[condition]['cooperation_rate']
            payoffs = metrics[condition]['average_payoff']

            rows.append({
                'Condition Type': 'Curriculum',
                'Condition': condition.replace('_', ' ').title(),
                'N Trials': len(coop_rates),
                'Mean Cooperation': f"{np.mean(coop_rates):.3f}",
                'Std Cooperation': f"{np.std(coop_rates):.3f}",
                'Mean Payoff': f"{np.mean(payoffs):.1f}",
                'Std Payoff': f"{np.std(payoffs):.1f}"
            })

    # Communication games
    game_labels = {
        'battle_of_sexes': 'Battle of Sexes + Comm',
        'ipgg_communication': 'IPGG + Communication',
        'volunteers_dilemma': 'Volunteers Dilemma + Comm'
    }

    for game, settings in sorted(game_data.items()):
        for setting, trials in sorted(settings.items()):
            if not trials:
                continue

            # Extract cooperation rates
            rates = []
            for trial in trials:
                if 'metrics' in trial and 'cooperation_rate' in trial['metrics']:
                    rates.append(trial['metrics']['cooperation_rate'])
                elif 'summary' in trial and 'cooperation_rate' in trial['summary']:
                    rates.append(trial['summary']['cooperation_rate'])

            if rates:
                setting_label = "Heterogeneous" if "heterogeneous" in setting else "Homogeneous"

                rows.append({
                    'Condition Type': 'Communication Game',
                    'Condition': f"{game_labels.get(game, game)} ({setting_label})",
                    'N Trials': len(rates),
                    'Mean Cooperation': f"{np.mean(rates):.3f}",
                    'Std Cooperation': f"{np.std(rates):.3f}",
                    'Mean Payoff': 'N/A',
                    'Std Payoff': 'N/A'
                })

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = OUTPUT_DIR / 'phase7_summary_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Print to console
    print("\n" + "="*100)
    print("PHASE 7 PARTIAL RESULTS SUMMARY TABLE")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    return df

def create_latex_table(df):
    """Create LaTeX-formatted table"""

    latex = r"""\begin{table}[ht]
\centering
\caption{Phase 7 Partial Results: Cooperation Rates and Payoffs}
\label{tab:phase7_partial}
\begin{tabular}{llrrrrr}
\toprule
\textbf{Type} & \textbf{Condition} & \textbf{N} & \textbf{Coop. Mean} & \textbf{Coop. SD} & \textbf{Payoff Mean} & \textbf{Payoff SD} \\
\midrule
"""

    for _, row in df.iterrows():
        latex += f"{row['Condition Type']} & {row['Condition']} & {row['N Trials']} & "
        latex += f"{row['Mean Cooperation']} & {row['Std Cooperation']} & "
        latex += f"{row['Mean Payoff']} & {row['Std Payoff']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    latex_path = OUTPUT_DIR / 'phase7_summary_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex)

    print(f"\nSaved LaTeX table: {latex_path}")

def main():
    print("Loading Phase 7 partial results...")

    # Load data
    curriculum_results = load_curriculum_results()
    game_data = load_game_results()

    print(f"Loaded {len(curriculum_results)} curriculum conditions")
    for condition, trials in curriculum_results.items():
        print(f"  - {condition}: {len(trials)} trials")

    print(f"\nLoaded {len(game_data)} communication games")
    for game, settings in game_data.items():
        for setting, trials in settings.items():
            print(f"  - {game} ({setting}): {len(trials)} trials")

    print("\nGenerating figures...")

    # Create figures
    create_curriculum_comparison_figure(curriculum_results)
    create_communication_game_figure(game_data)
    create_stage_progression_figure(curriculum_results)

    # Create tables
    print("\nGenerating tables...")
    df = create_summary_table(curriculum_results, game_data)
    create_latex_table(df)

    print("\n✓ Analysis complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
