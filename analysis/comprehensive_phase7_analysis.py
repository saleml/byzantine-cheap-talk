#!/usr/bin/env python3
"""
Comprehensive Phase 7 Analysis with Proper Cooperation Rate Extraction
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
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# Paths
RESULTS_DIR = Path("/home/salem.lahlou/salem/gameth/framework_exploration/results")
OUTPUT_DIR = Path("/home/salem.lahlou/salem/gameth/framework_exploration/analysis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_cooperation_from_contributions_dict(rounds_data, threshold=15):
    """Extract cooperation rate from Phase 6 format (contributions dict)"""
    total_decisions = 0
    cooperative_decisions = 0

    for round_data in rounds_data:
        if 'contributions' in round_data and isinstance(round_data['contributions'], dict):
            for agent, amount in round_data['contributions'].items():
                total_decisions += 1
                if amount >= threshold:
                    cooperative_decisions += 1

    return cooperative_decisions / total_decisions if total_decisions > 0 else 0.0

def extract_cooperation_from_decisions(rounds_data, threshold=15):
    """Extract cooperation rate from Phase 7 format (full_decisions nested)"""
    total_decisions = 0
    cooperative_decisions = 0

    for round_data in rounds_data:
        if 'full_decisions' in round_data:
            for agent, decision in round_data['full_decisions'].items():
                if 'action' in decision and isinstance(decision['action'], dict):
                    action = decision['action']
                    # Check for contribution amount
                    if 'amount' in action:
                        total_decisions += 1
                        if action['amount'] >= threshold:
                            cooperative_decisions += 1

    return cooperative_decisions / total_decisions if total_decisions > 0 else 0.0

def load_curriculum_with_proper_cooperation():
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
                            if 'rounds_data' in stage and stage['rounds_data']:
                                # Try Phase 6 format first (contributions dict)
                                coop_rate = extract_cooperation_from_contributions_dict(stage['rounds_data'])

                                # If that yields 0, try Phase 7 format
                                if coop_rate == 0.0:
                                    coop_rate = extract_cooperation_from_decisions(stage['rounds_data'])

                                stage['cooperation_rate_computed'] = coop_rate

                        # Update final cooperation rate
                        if data['stages']:
                            data['final_cooperation_rate_computed'] = data['stages'][-1].get('cooperation_rate_computed', 0.0)

                    curriculum_data[condition_name].append(data)

            except Exception as e:
                print(f"Error loading {results_file}: {e}")

    return curriculum_data

def create_payoff_and_cooperation_figure(curriculum_results):
    """Create comprehensive comparison figure"""

    # Focus on key conditions for paper
    key_conditions = [
        'success_driven_curriculum',
        'control_group',
        'communication_only_curriculum',
        'cooperation_first_curriculum',
        'direct_precursor',
        'scrambled_curriculum',
        'full_curriculum'
    ]

    # Prepare data
    condition_labels = []
    payoffs_data = []
    coop_data = []
    n_trials = []
    colors = []

    # Define colors: green for success, orange for neutral, red for failure
    color_map = {
        'success_driven_curriculum': '#2ecc71',
        'communication_only_curriculum': '#3498db',
        'cooperation_first_curriculum': '#9b59b6',
        'control_group': '#95a5a6',
        'direct_precursor': '#e67e22',
        'scrambled_curriculum': '#e74c3c',
        'full_curriculum': '#c0392b'
    }

    for condition in key_conditions:
        if condition not in curriculum_results:
            continue

        trials = curriculum_results[condition]
        if not trials:
            continue

        # Extract payoffs
        payoffs = [t['final_avg_payoff'] for t in trials if 'final_avg_payoff' in t]

        # Extract cooperation rates
        coop_rates = [t.get('final_cooperation_rate_computed', 0.0) for t in trials]

        if payoffs:
            # Format label
            label = condition.replace('_', ' ').title()
            label = label.replace('Ipgg', 'IPGG')

            condition_labels.append(label)
            payoffs_data.append(payoffs)
            coop_data.append(coop_rates)
            n_trials.append(len(payoffs))
            colors.append(color_map.get(condition, '#95a5a6'))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Payoff comparison
    positions = range(len(condition_labels))
    bp1 = ax1.boxplot(payoffs_data, positions=positions, widths=0.6, patch_artist=True,
                       boxprops=dict(alpha=0.7),
                       medianprops=dict(color='black', linewidth=2))

    # Color boxes
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(condition_labels, rotation=45, ha='right')
    ax1.set_ylabel('Average Payoff (tokens)', fontsize=13, fontweight='bold')
    ax1.set_title('Target Task Performance by Curriculum', fontsize=15, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add control line
    control_idx = condition_labels.index('Control Group') if 'Control Group' in condition_labels else None
    if control_idx is not None and payoffs_data[control_idx]:
        control_mean = np.mean(payoffs_data[control_idx])
        ax1.axhline(control_mean, color='gray', linestyle='--', linewidth=2, alpha=0.5, label=f'Control Mean: {control_mean:.1f}')
        ax1.legend()

    # Add n= annotations
    for i, n in enumerate(n_trials):
        y_min = ax1.get_ylim()[0]
        ax1.text(i, y_min + 5, f'n={n}', ha='center', fontsize=9, style='italic')

    # Cooperation rate comparison
    bp2 = ax2.boxplot(coop_data, positions=positions, widths=0.6, patch_artist=True,
                       boxprops=dict(alpha=0.7),
                       medianprops=dict(color='black', linewidth=2))

    # Color boxes
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(condition_labels, rotation=45, ha='right')
    ax2.set_ylabel('Cooperation Rate', fontsize=13, fontweight='bold')
    ax2.set_title('Cooperation Rate on Target Task', fontsize=15, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(axis='y', alpha=0.3)

    # Add n= annotations
    for i, n in enumerate(n_trials):
        ax2.text(i, -0.08, f'n={n}', ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comprehensive_curriculum_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'comprehensive_curriculum_comparison.png'}")
    plt.close()

def create_contribution_trajectory_figure(curriculum_results):
    """Create round-by-round contribution trajectory figure"""

    key_conditions = [
        'success_driven_curriculum',
        'communication_only_curriculum',
        'control_group',
        'direct_precursor',
        'full_curriculum'
    ]

    color_map = {
        'success_driven_curriculum': '#2ecc71',
        'communication_only_curriculum': '#3498db',
        'control_group': '#95a5a6',
        'direct_precursor': '#e67e22',
        'full_curriculum': '#c0392b'
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    for condition in key_conditions:
        if condition not in curriculum_results:
            continue

        trials = curriculum_results[condition]
        if not trials:
            continue

        # Collect round-by-round contributions
        all_round_contributions = defaultdict(list)

        for trial in trials:
            if 'stages' not in trial or not trial['stages']:
                continue

            # Use last stage (target task)
            target_stage = trial['stages'][-1]

            if 'rounds_data' in target_stage:
                for round_data in target_stage['rounds_data']:
                    round_num = round_data.get('round', 0)

                    # Extract contributions
                    if 'contributions' in round_data and isinstance(round_data['contributions'], dict):
                        contribs = list(round_data['contributions'].values())
                        if contribs:
                            all_round_contributions[round_num].append(np.mean(contribs))

        if all_round_contributions:
            rounds = sorted(all_round_contributions.keys())
            means = [np.mean(all_round_contributions[r]) for r in rounds]
            stds = [np.std(all_round_contributions[r]) for r in rounds]

            label = condition.replace('_', ' ').title()
            color = color_map.get(condition, '#95a5a6')

            ax.plot(rounds, means, marker='o', linewidth=2.5, markersize=8,
                   label=label, color=color, alpha=0.8)
            ax.fill_between(rounds,
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.2, color=color)

    ax.set_xlabel('Round', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Contribution (tokens)', fontsize=13, fontweight='bold')
    ax.set_title('Contribution Trajectories in Target Task (IPGG+P)', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'contribution_trajectories_phase7.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'contribution_trajectories_phase7.png'}")
    plt.close()

def create_summary_table(curriculum_results):
    """Create comprehensive summary table"""

    rows = []

    for condition, trials in sorted(curriculum_results.items()):
        if not trials:
            continue

        payoffs = [t.get('final_avg_payoff', 0.0) for t in trials if 'final_avg_payoff' in t]
        coop_rates = [t.get('final_cooperation_rate_computed', 0.0) for t in trials]

        if payoffs:
            # Calculate vs. control
            control_mean = np.mean([t.get('final_avg_payoff', 0) for t in curriculum_results.get('control_group', [])])
            vs_control = ((np.mean(payoffs) - control_mean) / control_mean * 100) if control_mean > 0 else 0

            rows.append({
                'Condition': condition.replace('_', ' ').title(),
                'N': len(payoffs),
                'Mean Payoff': f"{np.mean(payoffs):.1f}",
                'SD Payoff': f"{np.std(payoffs):.1f}",
                'vs. Control': f"{vs_control:+.1f}%",
                'Mean Coop': f"{np.mean(coop_rates):.3f}",
                'SD Coop': f"{np.std(coop_rates):.3f}"
            })

    df = pd.DataFrame(rows)

    # Sort by mean payoff
    df['_sort'] = df['Mean Payoff'].astype(float)
    df = df.sort_values('_sort', ascending=False).drop('_sort', axis=1)

    # Save
    csv_path = OUTPUT_DIR / 'phase7_comprehensive_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")

    # Print
    print("\n" + "="*100)
    print("PHASE 7 COMPREHENSIVE RESULTS")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    # Create LaTeX table
    latex = r"""\begin{table*}[ht]
\centering
\caption{Phase 7 Comprehensive Results: Curriculum Performance on Target Task}
\label{tab:phase7_results}
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Condition} & \textbf{N} & \textbf{Payoff Mean} & \textbf{Payoff SD} & \textbf{vs. Control} & \textbf{Coop. Mean} & \textbf{Coop. SD} \\
\midrule
"""

    for _, row in df.iterrows():
        latex += f"{row['Condition']} & {row['N']} & {row['Mean Payoff']} & {row['SD Payoff']} & {row['vs. Control']} & {row['Mean Coop']} & {row['SD Coop']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""

    latex_path = OUTPUT_DIR / 'phase7_comprehensive_results.tex'
    with open(latex_path, 'w') as f:
        f.write(latex)

    print(f"✓ Saved LaTeX table: {latex_path}\n")

    return df

def main():
    print("="*80)
    print("COMPREHENSIVE PHASE 7 ANALYSIS")
    print("="*80)
    print("\nLoading curriculum results with proper cooperation extraction...")

    curriculum_results = load_curriculum_with_proper_cooperation()

    print(f"\n✓ Loaded {len(curriculum_results)} curriculum conditions")
    for condition, trials in sorted(curriculum_results.items()):
        print(f"  • {condition}: {len(trials)} trials")

    print("\nGenerating figures...")
    create_payoff_and_cooperation_figure(curriculum_results)
    create_contribution_trajectory_figure(curriculum_results)

    print("\nGenerating summary table...")
    df = create_summary_table(curriculum_results)

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  • comprehensive_curriculum_comparison.png (payoff + cooperation)")
    print("  • contribution_trajectories_phase7.png (round-by-round)")
    print("  • phase7_comprehensive_results.csv")
    print("  • phase7_comprehensive_results.tex")

if __name__ == "__main__":
    main()
