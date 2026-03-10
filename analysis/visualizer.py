#!/usr/bin/env python
"""
Visualization tools for game-theoretic experiment results
Creates charts and graphs for analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Any
import logging

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

logging.basicConfig(level=logging.INFO)


class ExperimentVisualizer:
    def __init__(self, df: pd.DataFrame = None, metrics: Dict = None):
        """Initialize with data and/or metrics"""
        self.df = df
        self.metrics = metrics
        self.figures = []
        
    def load_data(self, csv_path: str):
        """Load data from CSV"""
        self.df = pd.read_csv(csv_path)
        
    def load_metrics(self, json_path: str):
        """Load metrics from JSON"""
        with open(json_path) as f:
            self.metrics = json.load(f)
    
    def plot_cooperation_rates(self, save_path: str = None) -> plt.Figure:
        """Plot cooperation rates across different dimensions"""
        if not self.metrics or "cooperation_rate" not in self.metrics:
            logging.warning("No cooperation rate metrics available")
            return None
        
        coop_data = self.metrics["cooperation_rate"]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Cooperation Rates Analysis", fontsize=16, fontweight='bold')
        
        # 1. By Game
        ax = axes[0, 0]
        if coop_data.get("by_game"):
            games = list(coop_data["by_game"].keys())
            rates = list(coop_data["by_game"].values())
            bars = ax.bar(games, rates, color='steelblue', alpha=0.7)
            ax.set_title("Cooperation Rate by Game")
            ax.set_ylabel("Cooperation Rate")
            ax.set_ylim(0, 1)
            ax.axhline(y=coop_data.get("overall", 0), color='red', linestyle='--', 
                      label=f'Overall: {coop_data.get("overall", 0):.2%}')
            ax.legend()
            
            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1%}', ha='center', va='bottom')
        
        # 2. By Setting
        ax = axes[0, 1]
        if coop_data.get("by_setting"):
            settings = list(coop_data["by_setting"].keys())
            rates = list(coop_data["by_setting"].values())
            
            # Use different colors for different settings
            colors = ['coral' if 'coalition' in s else 'skyblue' for s in settings]
            bars = ax.bar(range(len(settings)), rates, color=colors, alpha=0.7)
            ax.set_title("Cooperation Rate by Experimental Setting")
            ax.set_ylabel("Cooperation Rate")
            ax.set_xticks(range(len(settings)))
            ax.set_xticklabels([s.replace('_', '\n') for s in settings], rotation=0)
            ax.set_ylim(0, 1)
            
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1%}', ha='center', va='bottom')
        
        # 3. Evolution over Rounds
        ax = axes[1, 0]
        if coop_data.get("by_round"):
            rounds = sorted(coop_data["by_round"].keys())
            rates = [coop_data["by_round"][r] for r in rounds]
            ax.plot(rounds, rates, marker='o', linewidth=2, markersize=8, color='green')
            ax.set_title("Cooperation Rate Evolution Over Rounds")
            ax.set_xlabel("Round")
            ax.set_ylabel("Cooperation Rate")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        
        # 4. By Group (Coalition Analysis)
        ax = axes[1, 1]
        if coop_data.get("by_group") and coop_data["by_group"]:
            groups = list(coop_data["by_group"].keys())
            rates = list(coop_data["by_group"].values())
            bars = ax.bar(groups, rates, color=['purple', 'orange'], alpha=0.7)
            ax.set_title("In-Group Cooperation Rates")
            ax.set_ylabel("Cooperation Rate")
            ax.set_ylim(0, 1)
            
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1%}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, "No group data available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved cooperation rates plot to {save_path}")
        
        self.figures.append(fig)
        return fig
    
    def plot_punishment_analysis(self, save_path: str = None) -> plt.Figure:
        """Plot punishment patterns analysis"""
        if not self.metrics or "punishment_patterns" not in self.metrics:
            logging.warning("No punishment metrics available")
            return None
        
        punishment_data = self.metrics["punishment_patterns"]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Punishment Patterns Analysis", fontsize=16, fontweight='bold')
        
        # 1. In-group vs Out-group Punishment
        ax = axes[0]
        in_group = punishment_data.get("in_group_punishment", {})
        out_group = punishment_data.get("out_group_punishment", {})
        
        if in_group and out_group:
            categories = ['In-Group', 'Out-Group']
            counts = [in_group.get("count", 0), out_group.get("count", 0)]
            means = [in_group.get("mean", 0), out_group.get("mean", 0)]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, counts, width, label='Count', color='steelblue', alpha=0.7)
            bars2 = ax.bar(x + width/2, means, width, label='Mean Intensity', color='coral', alpha=0.7)
            
            ax.set_title("In-Group vs Out-Group Punishment")
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            
            # Add significance indicator if available
            if punishment_data.get("statistical_tests", {}).get("in_vs_out_group", {}).get("significant"):
                ax.text(0.5, max(counts + means) * 0.9, "*** p < 0.05", 
                       ha='center', fontweight='bold', fontsize=14)
        else:
            ax.text(0.5, 0.5, "No coalition punishment data", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # 2. Punishment Rate Over Time
        ax = axes[1]
        if self.df is not None and "game" in self.df.columns:
            pg_df = self.df[self.df["game"].str.contains("public_goods", na=False)]
            if not pg_df.empty and "round" in pg_df.columns:
                punishment_by_round = pg_df[pg_df["action"] == "punish"].groupby("round").size()
                total_by_round = pg_df.groupby("round").size()
                punishment_rate = (punishment_by_round / total_by_round).fillna(0)
                
                ax.plot(punishment_rate.index, punishment_rate.values, 
                       marker='o', linewidth=2, markersize=8, color='red')
                ax.set_title("Punishment Rate Evolution")
                ax.set_xlabel("Round")
                ax.set_ylabel("Punishment Rate")
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No temporal data available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved punishment analysis plot to {save_path}")
        
        self.figures.append(fig)
        return fig
    
    def plot_coordination_success(self, save_path: str = None) -> plt.Figure:
        """Plot coordination success rates"""
        if not self.metrics or "coordination_success" not in self.metrics:
            logging.warning("No coordination metrics available")
            return None
        
        coord_data = self.metrics["coordination_success"]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Stag Hunt Coordination Analysis", fontsize=16, fontweight='bold')
        
        # 1. Overall Success by Setting
        ax = axes[0, 0]
        if coord_data.get("by_setting"):
            settings = list(coord_data["by_setting"].keys())
            rates = list(coord_data["by_setting"].values())
            
            colors = ['coral' if 'coalition' in s else 'skyblue' for s in settings]
            bars = ax.bar(range(len(settings)), rates, color=colors, alpha=0.7)
            ax.set_title("Coordination Success by Setting")
            ax.set_ylabel("Success Rate")
            ax.set_xticks(range(len(settings)))
            ax.set_xticklabels([s.replace('_', '\n') for s in settings], rotation=0)
            ax.set_ylim(0, 1)
            ax.axhline(y=coord_data.get("overall_success_rate", 0), color='red', 
                      linestyle='--', alpha=0.5,
                      label=f'Overall: {coord_data.get("overall_success_rate", 0):.1%}')
            ax.legend()
            
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1%}', ha='center', va='bottom')
        
        # 2. With vs Without Communication
        ax = axes[0, 1]
        if coord_data.get("by_communication"):
            comm_types = list(coord_data["by_communication"].keys())
            rates = list(coord_data["by_communication"].values())
            
            colors = ['green' if 'with' in c else 'gray' for c in comm_types]
            bars = ax.bar(comm_types, rates, color=colors, alpha=0.7)
            ax.set_title("Impact of Communication on Coordination")
            ax.set_ylabel("Success Rate")
            ax.set_ylim(0, 1)
            
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1%}', ha='center', va='bottom')
            
            # Add improvement percentage if both available
            if len(rates) == 2 and rates[0] > 0:
                improvement = (rates[1] / rates[0] - 1) * 100 if rates[0] > 0 else 0
                ax.text(0.5, 0.9, f'Improvement: {improvement:+.1f}%', 
                       transform=ax.transAxes, ha='center', fontweight='bold')
        
        # 3. Success Evolution Over Rounds
        ax = axes[1, 0]
        if coord_data.get("success_evolution"):
            rounds = sorted(coord_data["success_evolution"].keys())
            rates = [coord_data["success_evolution"][r] for r in rounds]
            ax.plot(rounds, rates, marker='o', linewidth=2, markersize=8, color='purple')
            ax.set_title("Coordination Success Evolution")
            ax.set_xlabel("Round")
            ax.set_ylabel("Success Rate")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        
        # 4. Coalition Performance Comparison
        ax = axes[1, 1]
        if self.df is not None and "setting" in self.df.columns:
            coalition_df = self.df[self.df["setting"].str.contains("coalition", na=False)]
            if not coalition_df.empty and "agent_group" in coalition_df.columns:
                group_success = coalition_df.groupby("agent_group").apply(
                    lambda x: (x["action"].str.contains("STAG", na=False).sum() / len(x))
                )
                
                if not group_success.empty:
                    bars = ax.bar(group_success.index, group_success.values, 
                                 color=['purple', 'orange'], alpha=0.7)
                    ax.set_title("Coalition Group Performance")
                    ax.set_ylabel("STAG Choice Rate")
                    ax.set_ylim(0, 1)
                    
                    for bar, rate in zip(bars, group_success.values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{rate:.1%}', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, "No coalition data available", 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No data available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved coordination analysis plot to {save_path}")
        
        self.figures.append(fig)
        return fig
    
    def plot_communication_patterns(self, save_path: str = None) -> plt.Figure:
        """Plot communication patterns analysis"""
        if not self.metrics or "communication_patterns" not in self.metrics:
            logging.warning("No communication metrics available")
            return None
        
        comm_data = self.metrics["communication_patterns"]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Communication Patterns Analysis", fontsize=16, fontweight='bold')
        
        # 1. Word Frequency Distribution
        ax = axes[0, 0]
        if comm_data.get("word_frequency"):
            words = list(comm_data["word_frequency"].keys())[:10]
            frequencies = list(comm_data["word_frequency"].values())[:10]
            
            bars = ax.barh(range(len(words)), frequencies, color='teal', alpha=0.7)
            ax.set_title("Top 10 Communication Words")
            ax.set_xlabel("Frequency")
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            
            for bar, freq in zip(bars, frequencies):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{freq}', ha='left', va='center')
        
        # 2. Word-Action Correlation
        ax = axes[0, 1]
        if comm_data.get("word_action_correlation"):
            words = list(comm_data["word_action_correlation"].keys())
            stag_rates = [v["stag_rate"] for v in comm_data["word_action_correlation"].values()]
            
            bars = ax.bar(range(len(words)), stag_rates, color='green', alpha=0.7)
            ax.set_title("Word → STAG Action Correlation")
            ax.set_ylabel("P(STAG | Word)")
            ax.set_xticks(range(len(words)))
            ax.set_xticklabels(words, rotation=45, ha='right')
            ax.set_ylim(0, 1)
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
            ax.legend()
            
            for bar, rate in zip(bars, stag_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1%}', ha='center', va='bottom')
        
        # 3. Group Convergence
        ax = axes[1, 0]
        if comm_data.get("group_convergence") and comm_data["group_convergence"]:
            groups = list(comm_data["group_convergence"].keys())
            convergence_rates = [v["convergence_rate"] for v in comm_data["group_convergence"].values()]
            vocab_sizes = [v["vocabulary_size"] for v in comm_data["group_convergence"].values()]
            
            x = np.arange(len(groups))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, convergence_rates, width, 
                          label='Convergence Rate', color='purple', alpha=0.7)
            bars2 = ax.bar(x + width/2, np.array(vocab_sizes)/10, width, 
                          label='Vocab Size (÷10)', color='orange', alpha=0.7)
            
            ax.set_title("Group Communication Convergence")
            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            ax.legend()
            ax.set_ylim(0, 1)
            
            # Add dominant words as text
            for i, group in enumerate(groups):
                dominant_word = comm_data["group_convergence"][group].get("dominant_word", "")
                ax.text(i, 0.05, dominant_word, ha='center', fontweight='bold')
        else:
            ax.text(0.5, 0.5, "No group convergence data", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # 4. Emergent Conventions
        ax = axes[1, 1]
        if comm_data.get("emergent_conventions"):
            conventions = comm_data["emergent_conventions"][:5]  # Top 5
            
            if conventions:
                words = [c["word"] for c in conventions]
                coop_rates = [c["cooperation_rate"] for c in conventions]
                
                bars = ax.bar(words, coop_rates, color='darkgreen', alpha=0.7)
                ax.set_title("Emergent Communication Conventions")
                ax.set_ylabel("Cooperation Rate")
                ax.set_ylim(0, 1)
                ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5,
                          label='High Cooperation Threshold')
                ax.legend()
                
                for bar, rate in zip(bars, coop_rates):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{rate:.1%}', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, "No emergent conventions detected", 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No convention data available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved communication patterns plot to {save_path}")
        
        self.figures.append(fig)
        return fig
    
    def create_summary_figure(self, save_path: str = None) -> plt.Figure:
        """Create a summary figure with key findings"""
        if not self.metrics:
            logging.warning("No metrics available for summary")
            return None
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("Game-Theoretic LLM Experiments: Summary Results", 
                    fontsize=18, fontweight='bold')
        
        # Create a grid for summary statistics
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Key metrics boxes
        metrics_to_show = [
            ("Overall Cooperation", self.metrics.get("cooperation_rate", {}).get("overall", 0), "%"),
            ("Coordination Success", self.metrics.get("coordination_success", {}).get("overall_success_rate", 0), "%"),
            ("Punishment Rate", self.metrics.get("punishment_patterns", {}).get("punishment_rate", 0), "%"),
            ("Communication Words", len(self.metrics.get("communication_patterns", {}).get("unique_words", [])), ""),
            ("Emergent Conventions", len(self.metrics.get("communication_patterns", {}).get("emergent_conventions", [])), ""),
            ("In-Group Bias", "Detected" if self.metrics.get("punishment_patterns", {}).get("statistical_tests", {}).get("in_vs_out_group", {}).get("significant") else "Not Detected", "")
        ]
        
        for i, (label, value, unit) in enumerate(metrics_to_show):
            row = i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            
            # Format value
            if unit == "%":
                display_value = f"{value:.1%}"
            elif isinstance(value, (int, float)):
                display_value = f"{value:.0f}"
            else:
                display_value = str(value)
            
            # Create metric box
            ax.text(0.5, 0.7, display_value, ha='center', va='center',
                   fontsize=24, fontweight='bold',
                   color='darkgreen' if "Detected" in str(value) or value > 0.5 else 'darkblue')
            ax.text(0.5, 0.3, label, ha='center', va='center',
                   fontsize=14, color='gray')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Add border
            for spine in ['top', 'right', 'bottom', 'left']:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_linewidth(2)
                ax.spines[spine].set_edgecolor('lightgray')
        
        # Key findings text box
        ax = fig.add_subplot(gs[2, :])
        ax.axis('off')
        
        findings_text = "Key Findings:\n"
        if self.metrics.get("summary", {}).get("key_findings"):
            for finding in self.metrics["summary"]["key_findings"]:
                findings_text += f"• {finding}\n"
        else:
            findings_text += "• Analysis complete - see detailed plots for insights"
        
        ax.text(0.5, 0.5, findings_text, ha='center', va='center',
               fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved summary figure to {save_path}")
        
        self.figures.append(fig)
        return fig
    
    def generate_all_plots(self, output_dir: str = "plots"):
        """Generate all visualization plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate each plot type
        self.plot_cooperation_rates(output_path / "cooperation_rates.png")
        self.plot_punishment_analysis(output_path / "punishment_patterns.png")
        self.plot_coordination_success(output_path / "coordination_success.png")
        self.plot_communication_patterns(output_path / "communication_patterns.png")
        self.create_summary_figure(output_path / "summary.png")
        
        logging.info(f"Generated {len(self.figures)} plots in {output_dir}")
        
        return self.figures


def main():
    """Main entry point for visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize experimental results")
    parser.add_argument("--data", default="parsed_results.csv", help="Parsed data CSV")
    parser.add_argument("--metrics", default="metrics_results.json", help="Metrics JSON")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = ExperimentVisualizer()
    
    # Load data if available
    if Path(args.data).exists():
        visualizer.load_data(args.data)
    
    # Load metrics if available
    if Path(args.metrics).exists():
        visualizer.load_metrics(args.metrics)
    else:
        print(f"Metrics file {args.metrics} not found. Some plots may be unavailable.")
    
    # Generate plots
    visualizer.generate_all_plots(args.output_dir)
    
    if args.show:
        plt.show()
    
    print(f"\nVisualization complete! Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()