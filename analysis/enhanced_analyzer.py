#!/usr/bin/env python
"""
Enhanced analyzer with proper cooperation metrics calculation
Fixes the cooperation rate issue and generates informative visualizations
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, tukey_hsd
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('ggplot')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class EnhancedAnalyzer:
    def __init__(self, data_file: str = "analysis/unified_consolidated_data.jsonl"):
        """Initialize with unified consolidated data"""
        self.data_file = Path(data_file)
        self.records = []
        self.curriculum_data = []
        self.pilot_data = []
        self.load_data()
        self.calculate_true_cooperation_rates()
        
    def load_data(self):
        """Load and separate pilot and curriculum data"""
        with open(self.data_file, 'r') as f:
            for line in f:
                record = json.loads(line)
                self.records.append(record)
                
                if record.get('experiment_type') == 'curriculum':
                    self.curriculum_data.append(record)
                elif record.get('experiment_type') == 'pilot':
                    self.pilot_data.append(record)
        
        print(f"Loaded {len(self.records)} total records")
        print(f"  - Curriculum: {len(self.curriculum_data)}")
        print(f"  - Pilot: {len(self.pilot_data)}")
    
    def calculate_true_cooperation_rates(self):
        """Calculate actual cooperation rates from contributions/actions"""
        
        for record in self.records:
            game_id = record.get('game_id', '').lower()
            
            # For Public Goods Games: cooperation = contribution / endowment
            if 'public' in game_id or 'pgg' in game_id:
                all_contributions = []
                for round_data in record.get('rounds_data', []):
                    contributions = round_data.get('contributions', {})
                    if contributions:
                        all_contributions.extend(list(contributions.values()))
                
                if all_contributions:
                    avg_contribution = np.mean(all_contributions)
                    # Cooperation rate as fraction of endowment (20)
                    record['true_cooperation_rate'] = avg_contribution / 20.0
                    record['avg_contribution'] = avg_contribution
                else:
                    record['true_cooperation_rate'] = 0.0
                    record['avg_contribution'] = 0.0
            
            # For Prisoner's Dilemma: cooperation = fraction of "cooperate" choices
            elif 'prisoner' in game_id or 'ipd' in game_id:
                all_choices = []
                for round_data in record.get('rounds_data', []):
                    choices = round_data.get('choices', {})
                    if not choices:
                        choices = round_data.get('actions', {})
                    
                    for choice in choices.values():
                        if isinstance(choice, str):
                            all_choices.append(choice.lower())
                
                if all_choices:
                    coop_count = sum(1 for c in all_choices if 'cooperate' in c or c == 'c')
                    record['true_cooperation_rate'] = coop_count / len(all_choices)
                else:
                    record['true_cooperation_rate'] = 0.0
            
            # For Stag Hunt: cooperation = fraction choosing "stag"
            elif 'stag' in game_id:
                all_choices = []
                for round_data in record.get('rounds_data', []):
                    choices = round_data.get('choices', {})
                    if choices:
                        all_choices.extend([c.lower() for c in choices.values()])
                
                if all_choices:
                    stag_count = sum(1 for c in all_choices if 'stag' in c)
                    record['true_cooperation_rate'] = stag_count / len(all_choices)
                else:
                    record['true_cooperation_rate'] = 0.0
            
            else:
                # Default to stored cooperation rate
                record['true_cooperation_rate'] = record.get('cooperation_rate', 0.0)
    
    def analyze_curriculum_trajectories(self) -> Dict[str, Any]:
        """Analyze contribution trajectories for curriculum experiments"""
        trajectories = {}
        
        conditions = ['full_curriculum', 'scrambled_curriculum', 'direct_precursor', 'control_group']
        
        for condition in conditions:
            # Get final stage IPGG data for this condition
            final_stage_records = [
                r for r in self.curriculum_data
                if r.get('curriculum_condition') == condition
                and 'public' in r.get('game_id', '').lower()
                and ('punishment' in r.get('stage_name', '').lower() or 
                     'norm' in r.get('stage_name', '').lower() or
                     'target' in r.get('stage_name', '').lower() or
                     r.get('stage_num', 0) == max([rec.get('stage_num', 0) 
                                                    for rec in self.curriculum_data 
                                                    if rec.get('curriculum_condition') == condition]))
            ]
            
            # Calculate round-by-round averages
            round_contributions = defaultdict(list)
            
            for record in final_stage_records:
                for round_data in record.get('rounds_data', []):
                    round_num = round_data.get('round', 0)
                    contributions = round_data.get('contributions', {})
                    
                    if contributions:
                        avg_contrib = np.mean(list(contributions.values()))
                        round_contributions[round_num].append(avg_contrib)
            
            # Calculate statistics per round
            trajectory = {}
            for round_num in sorted(round_contributions.keys()):
                contribs = round_contributions[round_num]
                if contribs:
                    trajectory[round_num] = {
                        'mean': np.mean(contribs),
                        'std': np.std(contribs),
                        'sem': stats.sem(contribs) if len(contribs) > 1 else 0,
                        'n': len(contribs)
                    }
            
            trajectories[condition] = trajectory
        
        return trajectories
    
    def compare_final_performance(self) -> Dict[str, Any]:
        """Compare performance metrics across conditions in final stage"""
        comparison = {}
        
        for condition in ['full_curriculum', 'scrambled_curriculum', 'direct_precursor', 'control_group']:
            # Get final stage records
            condition_records = [
                r for r in self.curriculum_data
                if r.get('curriculum_condition') == condition
            ]
            
            # Group by trial and get highest stage
            trials = defaultdict(list)
            for record in condition_records:
                trials[record['trial_id']].append(record)
            
            final_stage_data = []
            for trial_id, trial_records in trials.items():
                if trial_records:
                    final_record = max(trial_records, key=lambda x: x.get('stage_num', 0))
                    final_stage_data.append(final_record)
            
            # Calculate metrics
            coop_rates = [r.get('true_cooperation_rate', 0) for r in final_stage_data]
            contributions = [r.get('avg_contribution', 0) for r in final_stage_data]
            payoffs = [r.get('average_payoff', 0) for r in final_stage_data]
            
            # Get last 5 rounds data
            last_5_contribs = []
            for record in final_stage_data:
                rounds_data = record.get('rounds_data', [])
                if len(rounds_data) >= 5:
                    for round_data in rounds_data[-5:]:
                        contribs = round_data.get('contributions', {})
                        if contribs:
                            last_5_contribs.append(np.mean(list(contribs.values())))
            
            comparison[condition] = {
                'cooperation_rate_mean': np.mean(coop_rates) if coop_rates else 0,
                'cooperation_rate_std': np.std(coop_rates) if coop_rates else 0,
                'contribution_mean': np.mean(contributions) if contributions else 0,
                'contribution_std': np.std(contributions) if contributions else 0,
                'payoff_mean': np.mean(payoffs) if payoffs else 0,
                'payoff_std': np.std(payoffs) if payoffs else 0,
                'last_5_rounds_contribution': np.mean(last_5_contribs) if last_5_contribs else 0,
                'n_trials': len(final_stage_data)
            }
        
        return comparison
    
    def analyze_pilot_games(self) -> Dict[str, Any]:
        """Analyze pilot study results with correct metrics"""
        analysis = {
            'public_goods': {},
            'stag_hunt': {},
            'stag_hunt_communication': {},
            'stag_hunt_by_setting': {}  # Add setting-specific analysis
        }
        
        # Public Goods Game analysis
        pgg_records = [r for r in self.pilot_data if 'public_goods' in r.get('game_id', '')]
        
        # By setting
        for setting in set(r.get('setting', '') for r in pgg_records):
            setting_records = [r for r in pgg_records if r.get('setting') == setting]
            
            all_contribs = []
            for record in setting_records:
                for round_data in record.get('rounds_data', []):
                    contribs = round_data.get('contributions', {})
                    if contribs:
                        all_contribs.extend(list(contribs.values()))
            
            if all_contribs:
                analysis['public_goods'][setting] = {
                    'avg_contribution': np.mean(all_contribs),
                    'cooperation_rate': np.mean(all_contribs) / 20.0,
                    'n_trials': len(setting_records)
                }
        
        # Stag Hunt analysis - use round-level cooperation rate from records
        sh_records = [r for r in self.pilot_data if 'stag_hunt' in r.get('game_id', '') 
                      and 'communication' not in r.get('game_id', '')]
        
        # Calculate both metrics
        sh_round_cooperation = []  # Round-level (all agents cooperate)
        sh_individual_choices = []  # Individual choices
        
        for record in sh_records:
            # Use the cooperation_rate from the record (round-level)
            sh_round_cooperation.append(record.get('cooperation_rate', 0))
            
            # Also calculate individual choices for comparison
            for round_data in record.get('rounds_data', []):
                choices = round_data.get('choices', {})
                if choices:
                    sh_individual_choices.extend([1 if 'stag' in c.lower() else 0 for c in choices.values()])
        
        if sh_round_cooperation:
            analysis['stag_hunt'] = {
                'cooperation_rate': np.mean(sh_round_cooperation),  # Use round-level
                'individual_choice_rate': np.mean(sh_individual_choices) if sh_individual_choices else 0,
                'n_trials': len(sh_records)
            }
        
        # Stag Hunt with Communication - use round-level cooperation
        shc_records = [r for r in self.pilot_data if 'stag_hunt_communication' in r.get('game_id', '')]
        
        shc_round_cooperation = []
        shc_individual_choices = []
        
        for record in shc_records:
            # Use the cooperation_rate from the record (round-level)
            shc_round_cooperation.append(record.get('cooperation_rate', 0))
            
            # Also calculate individual choices
            for round_data in record.get('rounds_data', []):
                choices = round_data.get('choices', {})
                if choices:
                    shc_individual_choices.extend([1 if 'stag' in c.lower() else 0 for c in choices.values()])
        
        if shc_round_cooperation:
            analysis['stag_hunt_communication'] = {
                'cooperation_rate': np.mean(shc_round_cooperation),  # Use round-level
                'individual_choice_rate': np.mean(shc_individual_choices) if shc_individual_choices else 0,
                'n_trials': len(shc_records)
            }
        
        # Add setting-specific analysis for Stag Hunt
        for game_type in ['stag_hunt', 'stag_hunt_communication']:
            records = [r for r in self.pilot_data if game_type in r.get('game_id', '')]
            if game_type == 'stag_hunt':
                records = [r for r in records if 'communication' not in r.get('game_id', '')]
            
            by_setting = {}
            for record in records:
                setting = record.get('setting', 'unknown')
                if setting not in by_setting:
                    by_setting[setting] = []
                by_setting[setting].append(record.get('cooperation_rate', 0))
            
            analysis['stag_hunt_by_setting'][game_type] = {
                setting: {'mean': np.mean(rates), 'n': len(rates)}
                for setting, rates in by_setting.items()
            }
        
        return analysis
    
    def generate_enhanced_visualizations(self, output_dir: str = "analysis/figures") -> List[str]:
        """Generate improved, informative visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        generated = []
        
        # 1. Curriculum IPGG Trajectories (Primary Figure)
        fig = self._plot_curriculum_trajectories()
        path = output_path / "curriculum_ipgg_trajectories.png"
        fig.savefig(path, bbox_inches='tight')
        generated.append(str(path))
        plt.close(fig)
        
        # 2. Final Stage Comparison
        fig = self._plot_final_comparison()
        path = output_path / "final_stage_comparison.png"
        fig.savefig(path, bbox_inches='tight')
        generated.append(str(path))
        plt.close(fig)
        
        # 3. Pilot Study Results
        fig = self._plot_pilot_results()
        path = output_path / "pilot_study_overview.png"
        fig.savefig(path, bbox_inches='tight')
        generated.append(str(path))
        plt.close(fig)
        
        # 4. Comprehensive Comparison
        fig = self._plot_comprehensive_comparison()
        path = output_path / "comprehensive_comparison.png"
        fig.savefig(path, bbox_inches='tight')
        generated.append(str(path))
        plt.close(fig)
        
        # 5. Learning Progression
        fig = self._plot_learning_progression()
        path = output_path / "learning_progression.png"
        fig.savefig(path, bbox_inches='tight')
        generated.append(str(path))
        plt.close(fig)
        
        print(f"Generated {len(generated)} enhanced visualizations")
        return generated
    
    def _plot_curriculum_trajectories(self) -> plt.Figure:
        """Plot IPGG contribution trajectories over rounds"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        trajectories = self.analyze_curriculum_trajectories()
        
        colors = {
            'full_curriculum': '#2E7D32',  # Dark green
            'scrambled_curriculum': '#C62828',  # Dark red  
            'direct_precursor': '#1565C0',  # Dark blue
            'control_group': '#F57C00'  # Dark orange
        }
        
        labels = {
            'full_curriculum': 'Full Curriculum',
            'scrambled_curriculum': 'Scrambled Order',
            'direct_precursor': 'Direct Precursor',
            'control_group': 'Control (No Curriculum)'
        }
        
        # Plot 1: Contribution trajectories
        for condition, trajectory in trajectories.items():
            if trajectory:
                rounds = sorted(trajectory.keys())
                means = [trajectory[r]['mean'] for r in rounds]
                sems = [trajectory[r]['sem'] for r in rounds]
                
                ax1.plot(rounds, means, label=labels[condition], 
                        color=colors[condition], linewidth=2.5, marker='o', markersize=6)
                
                # Add error bands
                lower = [means[i] - 1.96*sems[i] for i in range(len(means))]
                upper = [means[i] + 1.96*sems[i] for i in range(len(means))]
                ax1.fill_between(rounds, lower, upper, alpha=0.2, color=colors[condition])
        
        ax1.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Half Endowment')
        ax1.axhline(y=15, color='green', linestyle=':', alpha=0.5, label='Socially Optimal')
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Average Contribution (out of 20)', fontsize=12)
        ax1.set_title('IPGG Contribution Trajectories by Curriculum Condition', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', frameon=True, fancybox=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 20])
        ax1.set_xlim([0.5, 10.5])
        
        # Plot 2: Cooperation rate (as % of endowment)
        for condition, trajectory in trajectories.items():
            if trajectory:
                rounds = sorted(trajectory.keys())
                coop_rates = [trajectory[r]['mean'] / 20.0 * 100 for r in rounds]
                
                ax2.plot(rounds, coop_rates, label=labels[condition],
                        color=colors[condition], linewidth=2.5, marker='s', markersize=5)
        
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=75, color='green', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Cooperation Rate (%)', fontsize=12)
        ax2.set_title('Cooperation Rates Over Time', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', frameon=True, fancybox=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        ax2.set_xlim([0.5, 10.5])
        
        fig.suptitle('Curriculum Learning Impact on IPGG Cooperation', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def _plot_final_comparison(self) -> plt.Figure:
        """Plot final stage comparison across conditions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        comparison = self.compare_final_performance()
        conditions = list(comparison.keys())
        display_names = ['Full\nCurriculum', 'Scrambled\nOrder', 'Direct\nPrecursor', 'Control\n(No Curriculum)']
        colors = ['#2E7D32', '#C62828', '#1565C0', '#F57C00']
        
        # 1. Average Contributions
        ax = axes[0, 0]
        means = [comparison[c]['contribution_mean'] for c in conditions]
        stds = [comparison[c]['contribution_std'] for c in conditions]
        bars = ax.bar(display_names, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Half Endowment')
        ax.set_ylabel('Average Contribution', fontsize=12)
        ax.set_title('Final Stage: Average Contributions', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 20])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.5,
                   f'{mean:.1f}', ha='center', fontsize=10)
        
        # 2. Cooperation Rates
        ax = axes[0, 1]
        coop_means = [comparison[c]['cooperation_rate_mean'] * 100 for c in conditions]
        coop_stds = [comparison[c]['cooperation_rate_std'] * 100 for c in conditions]
        bars = ax.bar(display_names, coop_means, yerr=coop_stds, capsize=5, color=colors, alpha=0.8)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Cooperation Rate (%)', fontsize=12)
        ax.set_title('Final Stage: Cooperation Rates', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars, coop_means, coop_stds):
            ax.text(bar.get_x() + bar.get_width()/2, mean + std + 2,
                   f'{mean:.1f}%', ha='center', fontsize=10)
        
        # 3. Average Payoffs
        ax = axes[1, 0]
        payoff_means = [comparison[c]['payoff_mean'] for c in conditions]
        payoff_stds = [comparison[c]['payoff_std'] for c in conditions]
        bars = ax.bar(display_names, payoff_means, yerr=payoff_stds, capsize=5, color=colors, alpha=0.8)
        ax.set_ylabel('Average Payoff', fontsize=12)
        ax.set_title('Final Stage: Average Payoffs', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, mean in zip(bars, payoff_means):
            ax.text(bar.get_x() + bar.get_width()/2, mean + 10,
                   f'{mean:.0f}', ha='center', fontsize=10)
        
        # 4. Last 5 Rounds Performance
        ax = axes[1, 1]
        last5_means = [comparison[c]['last_5_rounds_contribution'] for c in conditions]
        bars = ax.bar(display_names, last5_means, color=colors, alpha=0.8)
        ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Average Contribution (Last 5 Rounds)', fontsize=12)
        ax.set_title('Final Stage: Endgame Behavior', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 20])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, mean in zip(bars, last5_means):
            ax.text(bar.get_x() + bar.get_width()/2, mean + 0.5,
                   f'{mean:.1f}', ha='center', fontsize=10)
        
        fig.suptitle('Curriculum Learning: Final Stage Performance Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _plot_pilot_results(self) -> plt.Figure:
        """Plot pilot study results overview"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        pilot_analysis = self.analyze_pilot_games()
        
        # 1. Public Goods by Setting
        ax = axes[0, 0]
        if pilot_analysis['public_goods']:
            settings = list(pilot_analysis['public_goods'].keys())
            contributions = [pilot_analysis['public_goods'][s]['avg_contribution'] for s in settings]
            display_names = [s.replace('_', '\n').replace('setting ', '') for s in settings]
            
            bars = ax.bar(display_names, contributions, color=['#1976D2', '#388E3C'], alpha=0.8)
            ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Half Endowment')
            ax.set_ylabel('Average Contribution', fontsize=12)
            ax.set_title('Public Goods Game by Setting', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 20])
            ax.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, contributions):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                       f'{val:.1f}', ha='center', fontsize=10)
        
        # 2. Stag Hunt: Communication Effect
        ax = axes[0, 1]
        sh_no_comm = pilot_analysis.get('stag_hunt', {}).get('cooperation_rate', 0) * 100
        sh_comm = pilot_analysis.get('stag_hunt_communication', {}).get('cooperation_rate', 0) * 100
        
        bars = ax.bar(['No Communication', 'With\nCommunication'], [sh_no_comm, sh_comm],
                      color=['#E64A19', '#689F38'], alpha=0.8)
        ax.set_ylabel('Cooperation Rate (%)', fontsize=12)
        ax.set_title('Stag Hunt: Effect of Communication', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        # Add improvement arrow
        if sh_comm > sh_no_comm:
            improvement = sh_comm - sh_no_comm
            ax.annotate(f'+{improvement:.1f}%', 
                       xy=(1, sh_comm), xytext=(1, sh_comm + 10),
                       ha='center', fontsize=12, fontweight='bold', color='green',
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        for bar, val in zip(bars, [sh_no_comm, sh_comm]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 2,
                   f'{val:.1f}%', ha='center', fontsize=10)
        
        # 3. Model Family Performance
        ax = axes[0, 2]
        model_coop = defaultdict(list)
        
        for record in self.pilot_data:
            for agent in record.get('agents', []):
                model = agent.get('model', '').lower()
                if 'claude' in model:
                    family = 'Claude'
                elif 'llama' in model:
                    family = 'Llama'
                elif 'gpt' in model:
                    family = 'GPT'
                elif 'mixtral' in model:
                    family = 'Mixtral'
                else:
                    continue
                
                model_coop[family].append(record.get('true_cooperation_rate', 0))
        
        if model_coop:
            families = list(model_coop.keys())
            means = [np.mean(model_coop[f]) * 100 for f in families]
            
            bars = ax.bar(families, means, color=['#7B1FA2', '#00796B', '#F57C00', '#455A64'], alpha=0.8)
            ax.set_ylabel('Average Cooperation Rate (%)', fontsize=12)
            ax.set_title('Performance by Model Family', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 100])
            ax.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                       f'{val:.1f}%', ha='center', fontsize=10)
        
        # 4. PGG Contribution Distribution
        ax = axes[1, 0]
        all_pgg_contribs = []
        for record in [r for r in self.pilot_data if 'public_goods' in r.get('game_id', '')]:
            for round_data in record.get('rounds_data', []):
                contribs = round_data.get('contributions', {})
                if contribs:
                    all_pgg_contribs.extend(list(contribs.values()))
        
        if all_pgg_contribs:
            ax.hist(all_pgg_contribs, bins=20, edgecolor='black', alpha=0.7, color='#1565C0')
            ax.axvline(x=np.mean(all_pgg_contribs), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(all_pgg_contribs):.1f}')
            ax.axvline(x=10, color='gray', linestyle='--', alpha=0.5, label='Half Endowment')
            ax.set_xlabel('Contribution Amount', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of PGG Contributions', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # 5. Stag Hunt Choice Distribution
        ax = axes[1, 1]
        sh_all = pilot_analysis.get('stag_hunt', {}).get('cooperation_rate', 0)
        sh_comm_all = pilot_analysis.get('stag_hunt_communication', {}).get('cooperation_rate', 0)
        
        categories = ['Stag Hunt\n(No Comm)', 'Stag Hunt\n(With Comm)']
        stag_rates = [sh_all * 100, sh_comm_all * 100]
        hare_rates = [(1-sh_all) * 100, (1-sh_comm_all) * 100]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, stag_rates, width, label='Stag', color='#2E7D32', alpha=0.8)
        bars2 = ax.bar(x + width/2, hare_rates, width, label='Hare', color='#C62828', alpha=0.8)
        
        ax.set_ylabel('Choice Rate (%)', fontsize=12)
        ax.set_title('Stag Hunt Choice Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                       f'{height:.1f}%', ha='center', fontsize=9)
        
        # 6. Summary Statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = "PILOT STUDY SUMMARY\n" + "="*30 + "\n\n"
        
        # PGG stats
        pgg_overall = np.mean([r.get('true_cooperation_rate', 0) 
                              for r in self.pilot_data if 'public_goods' in r.get('game_id', '')])
        summary_text += f"Public Goods Game:\n"
        summary_text += f"  Avg Cooperation: {pgg_overall*100:.1f}%\n"
        summary_text += f"  Avg Contribution: {np.mean(all_pgg_contribs):.1f}/20\n\n"
        
        # SH stats
        summary_text += f"Stag Hunt:\n"
        summary_text += f"  Without Comm: {sh_all*100:.1f}%\n"
        summary_text += f"  With Comm: {sh_comm_all*100:.1f}%\n"
        summary_text += f"  Improvement: +{(sh_comm_all-sh_all)*100:.1f}%\n\n"
        
        # Overall
        summary_text += f"Total Trials: {len(self.pilot_data)}\n"
        summary_text += f"Games: PGG, SH, SH+Comm"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('Pilot Study: Comprehensive Results Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _plot_comprehensive_comparison(self) -> plt.Figure:
        """Plot comprehensive comparison across all experiments"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Prepare data
        pilot_pgg_coop = np.mean([r.get('true_cooperation_rate', 0) 
                                 for r in self.pilot_data if 'public_goods' in r.get('game_id', '')])
        
        comparison = self.compare_final_performance()
        
        # 1. Pilot vs Curriculum Comparison
        ax = axes[0, 0]
        
        pilot_val = pilot_pgg_coop * 100
        curriculum_vals = [comparison[c]['cooperation_rate_mean'] * 100 
                          for c in ['full_curriculum', 'scrambled_curriculum', 
                                   'direct_precursor', 'control_group']]
        
        x_pos = [0]
        ax.bar(x_pos, [pilot_val], width=0.4, label='Pilot (One-shot PGG)', 
              color='#607D8B', alpha=0.8)
        
        x_pos2 = np.arange(1.5, 5.5, 1)
        colors = ['#2E7D32', '#C62828', '#1565C0', '#F57C00']
        labels = ['Full', 'Scrambled', 'Direct', 'Control']
        bars = ax.bar(x_pos2, curriculum_vals, width=0.4, color=colors, alpha=0.8)
        
        ax.set_ylabel('Cooperation Rate (%)', fontsize=12)
        ax.set_title('Pilot vs Curriculum: Cooperation Rates', fontsize=14, fontweight='bold')
        ax.set_xticks([0] + list(x_pos2))
        ax.set_xticklabels(['Pilot\n(One-shot)'] + labels, fontsize=10)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])
        
        # 2. Learning Effect Size
        ax = axes[0, 1]
        
        # Calculate effect sizes (difference from control)
        control_coop = comparison['control_group']['cooperation_rate_mean']
        effect_sizes = []
        conditions_display = []
        
        for condition in ['full_curriculum', 'scrambled_curriculum', 'direct_precursor']:
            effect = (comparison[condition]['cooperation_rate_mean'] - control_coop) * 100
            effect_sizes.append(effect)
            conditions_display.append(condition.replace('_', '\n').title())
        
        colors = ['#2E7D32' if e > 0 else '#C62828' for e in effect_sizes]
        bars = ax.bar(conditions_display, effect_sizes, color=colors, alpha=0.8)
        
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_ylabel('Effect Size (% vs Control)', fontsize=12)
        ax.set_title('Curriculum Learning Effects', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, effect_sizes):
            y_pos = val + 0.5 if val > 0 else val - 0.5
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{val:+.1f}%', ha='center', fontsize=10)
        
        # 3. Stage Progression (Full Curriculum)
        ax = axes[1, 0]
        
        # Get stage-by-stage data for full curriculum
        full_stages = [r for r in self.curriculum_data 
                      if r.get('curriculum_condition') == 'full_curriculum']
        
        stage_names = ['IPD\n(2-player)', 'N-IPD\n(4-player)', 'IPGG\n(No Punish)', 'IPGG\n(Punish)']
        stage_coop = []
        
        for stage_num in range(1, 5):
            stage_records = [r for r in full_stages if r.get('stage_num') == stage_num]
            if stage_records:
                coop = np.mean([r.get('true_cooperation_rate', 0) for r in stage_records])
                stage_coop.append(coop * 100)
            else:
                stage_coop.append(0)
        
        bars = ax.bar(stage_names, stage_coop, color=['#1565C0', '#2196F3', '#64B5F6', '#90CAF9'], alpha=0.8)
        ax.set_ylabel('Cooperation Rate (%)', fontsize=12)
        ax.set_title('Full Curriculum: Stage Progression', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        # Add trend line
        x_numeric = np.arange(len(stage_names))
        z = np.polyfit(x_numeric, stage_coop, 1)
        p = np.poly1d(z)
        ax.plot(x_numeric, p(x_numeric), "r--", alpha=0.5, linewidth=2, label='Trend')
        ax.legend()
        
        # 4. Key Metrics Summary Table
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create summary table data
        table_data = []
        table_data.append(['Metric', 'Pilot', 'Control', 'Full Curr.', 'Best'])
        table_data.append(['='*15, '='*10, '='*10, '='*10, '='*10])
        
        # Cooperation rate
        pilot_coop = pilot_pgg_coop * 100
        control_coop = comparison['control_group']['cooperation_rate_mean'] * 100
        full_coop = comparison['full_curriculum']['cooperation_rate_mean'] * 100
        best_coop = max(comparison[c]['cooperation_rate_mean'] * 100 for c in comparison.keys())
        best_name = max(comparison.keys(), key=lambda c: comparison[c]['cooperation_rate_mean'])
        
        table_data.append(['Coop. Rate', f'{pilot_coop:.1f}%', f'{control_coop:.1f}%', 
                          f'{full_coop:.1f}%', f'{best_coop:.1f}%'])
        
        # Average contribution
        pilot_contrib = np.mean([r.get('avg_contribution', 0) for r in self.pilot_data 
                                if 'public_goods' in r.get('game_id', '')])
        control_contrib = comparison['control_group']['contribution_mean']
        full_contrib = comparison['full_curriculum']['contribution_mean']
        best_contrib = max(comparison[c]['contribution_mean'] for c in comparison.keys())
        
        table_data.append(['Avg. Contrib.', f'{pilot_contrib:.1f}', f'{control_contrib:.1f}',
                          f'{full_contrib:.1f}', f'{best_contrib:.1f}'])
        
        # Sample size
        pilot_n = len([r for r in self.pilot_data if 'public_goods' in r.get('game_id', '')])
        table_data.append(['N Trials', str(pilot_n), '30', '30', best_name[:8]])
        
        # Create table
        table_text = '\n'.join([' '.join([f'{cell:^15}' for cell in row]) for row in table_data])
        
        ax.text(0.5, 0.5, table_text, transform=ax.transAxes, fontsize=10,
               ha='center', va='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        fig.suptitle('Comprehensive Analysis: Pilot vs Curriculum Learning', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _plot_learning_progression(self) -> plt.Figure:
        """Plot detailed learning progression analysis"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        trajectories = self.analyze_curriculum_trajectories()
        
        # 1. Round-by-round improvement (Full vs Control)
        ax = axes[0]
        
        if 'full_curriculum' in trajectories and 'control_group' in trajectories:
            full_traj = trajectories['full_curriculum']
            control_traj = trajectories['control_group']
            
            rounds = sorted(set(full_traj.keys()) & set(control_traj.keys()))
            
            if rounds:
                full_means = [full_traj[r]['mean'] for r in rounds]
                control_means = [control_traj[r]['mean'] for r in rounds]
                improvements = [f - c for f, c in zip(full_means, control_means)]
                
                ax.plot(rounds, improvements, marker='o', linewidth=2.5, color='#2E7D32', markersize=8)
                ax.axhline(y=0, color='black', linewidth=1)
                ax.fill_between(rounds, 0, improvements, where=[i > 0 for i in improvements],
                               interpolate=True, alpha=0.3, color='green')
                ax.fill_between(rounds, 0, improvements, where=[i <= 0 for i in improvements],
                               interpolate=True, alpha=0.3, color='red')
                
                ax.set_xlabel('Round', fontsize=12)
                ax.set_ylabel('Contribution Difference\n(Full Curriculum - Control)', fontsize=12)
                ax.set_title('Learning Advantage Over Control', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
        
        # 2. Variance reduction over rounds
        ax = axes[1]
        
        for condition, trajectory in trajectories.items():
            if trajectory:
                rounds = sorted(trajectory.keys())
                stds = [trajectory[r]['std'] for r in rounds if trajectory[r]['std'] > 0]
                
                if len(stds) > 1:
                    ax.plot(rounds[:len(stds)], stds, marker='o', linewidth=2,
                           label=condition.replace('_', ' ').title())
        
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Standard Deviation', fontsize=12)
        ax.set_title('Behavioral Consistency Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 3. First vs Last Round Comparison
        ax = axes[2]
        
        conditions = ['full_curriculum', 'scrambled_curriculum', 'direct_precursor', 'control_group']
        first_round = []
        last_round = []
        
        for condition in conditions:
            traj = trajectories.get(condition, {})
            if traj:
                rounds = sorted(traj.keys())
                if rounds:
                    first_round.append(traj[rounds[0]]['mean'])
                    last_round.append(traj[rounds[-1]]['mean'])
                else:
                    first_round.append(0)
                    last_round.append(0)
            else:
                first_round.append(0)
                last_round.append(0)
        
        x = np.arange(len(conditions))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, first_round, width, label='Round 1', color='#FFA726', alpha=0.8)
        bars2 = ax.bar(x + width/2, last_round, width, label='Round 10', color='#26A69A', alpha=0.8)
        
        ax.set_ylabel('Average Contribution', fontsize=12)
        ax.set_title('Learning Effect: First vs Last Round', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', '\n').title() for c in conditions], fontsize=10)
        ax.legend()
        ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 20])
        
        fig.suptitle('Learning Progression Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        comparison = self.compare_final_performance()
        pilot_analysis = self.analyze_pilot_games()
        trajectories = self.analyze_curriculum_trajectories()
        
        summary = {
            'pilot_study': {
                'pgg_cooperation_rate': np.mean([r.get('true_cooperation_rate', 0) 
                                                for r in self.pilot_data 
                                                if 'public_goods' in r.get('game_id', '')]) * 100,
                'sh_no_comm': pilot_analysis.get('stag_hunt', {}).get('cooperation_rate', 0) * 100,
                'sh_with_comm': pilot_analysis.get('stag_hunt_communication', {}).get('cooperation_rate', 0) * 100,
                'n_trials': len(self.pilot_data)
            },
            'curriculum_study': {
                'final_stage_comparison': comparison,
                'trajectories_summary': {
                    condition: {
                        'first_round': traj[min(traj.keys())]['mean'] if traj else 0,
                        'last_round': traj[max(traj.keys())]['mean'] if traj else 0,
                        'improvement': (traj[max(traj.keys())]['mean'] - traj[min(traj.keys())]['mean']) if traj else 0
                    }
                    for condition, traj in trajectories.items()
                },
                'best_performer': max(comparison.keys(), 
                                     key=lambda c: comparison[c]['cooperation_rate_mean']),
                'n_trials': 120
            }
        }
        
        return summary


def main():
    """Main entry point for enhanced analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced analyzer with corrected cooperation metrics")
    parser.add_argument("--data", default="analysis/unified_consolidated_data.jsonl",
                       help="Path to unified consolidated data")
    parser.add_argument("--output-dir", default="analysis/figures",
                       help="Directory for output figures")
    parser.add_argument("--summary", action="store_true",
                       help="Print summary statistics")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = EnhancedAnalyzer(args.data)
    
    print("\n" + "="*60)
    print("ENHANCED ANALYSIS WITH CORRECTED METRICS")
    print("="*60)
    
    # Generate visualizations
    print("\nGenerating enhanced visualizations...")
    generated = analyzer.generate_enhanced_visualizations(args.output_dir)
    for path in generated:
        print(f"  ✓ {path}")
    
    # Print summary if requested
    if args.summary:
        summary = analyzer.get_summary_statistics()
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        print("\nPilot Study:")
        print(f"  PGG Cooperation Rate: {summary['pilot_study']['pgg_cooperation_rate']:.1f}%")
        print(f"  Stag Hunt (no comm): {summary['pilot_study']['sh_no_comm']:.1f}%")
        print(f"  Stag Hunt (with comm): {summary['pilot_study']['sh_with_comm']:.1f}%")
        print(f"  Communication benefit: +{summary['pilot_study']['sh_with_comm'] - summary['pilot_study']['sh_no_comm']:.1f}%")
        
        print("\nCurriculum Study:")
        final = summary['curriculum_study']['final_stage_comparison']
        for condition in final.keys():
            print(f"  {condition}:")
            print(f"    Cooperation: {final[condition]['cooperation_rate_mean']*100:.1f}%")
            print(f"    Contribution: {final[condition]['contribution_mean']:.1f}/20")
        
        print(f"\nBest Performer: {summary['curriculum_study']['best_performer']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()