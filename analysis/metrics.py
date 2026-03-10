#!/usr/bin/env python
"""
Metrics calculation for game-theoretic experiments
Implements key metrics from CLAUDE.md research objectives
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
import json
import logging

logging.basicConfig(level=logging.INFO)


class MetricsCalculator:
    def __init__(self, df: pd.DataFrame):
        """Initialize with parsed results DataFrame"""
        self.df = df
        
    def calculate_cooperation_rate(self, df: pd.DataFrame = None) -> Dict:
        """
        Calculate cooperation rates across different dimensions
        Returns rates by game, setting, agent, and round
        """
        if df is None:
            df = self.df
        
        results = {
            "overall": 0,
            "by_game": {},
            "by_setting": {},
            "by_agent": {},
            "by_round": {},
            "by_group": {}
        }
        
        if df.empty or "cooperated" not in df.columns:
            return results
        
        # Overall cooperation rate
        results["overall"] = df["cooperated"].mean()
        
        # By game
        if "game" in df.columns:
            for game in df["game"].unique():
                game_df = df[df["game"] == game]
                results["by_game"][game] = game_df["cooperated"].mean()
        
        # By setting
        if "setting" in df.columns:
            for setting in df["setting"].unique():
                setting_df = df[df["setting"] == setting]
                results["by_setting"][setting] = setting_df["cooperated"].mean()
        
        # By agent
        if "agent_id" in df.columns:
            for agent in df["agent_id"].unique():
                agent_df = df[df["agent_id"] == agent]
                if not agent_df.empty:
                    results["by_agent"][agent] = agent_df["cooperated"].mean()
        
        # By round (evolution over time)
        if "round" in df.columns:
            round_cooperation = df.groupby("round")["cooperated"].mean()
            results["by_round"] = round_cooperation.to_dict()
        
        # By group (for coalition analysis)
        if "agent_group" in df.columns:
            for group in df["agent_group"].unique():
                if group != "none":
                    group_df = df[df["agent_group"] == group]
                    results["by_group"][group] = group_df["cooperated"].mean()
        
        return results
    
    def analyze_punishment_patterns(self, df: pd.DataFrame = None) -> Dict:
        """
        Analyze punishment patterns in Public Goods game
        Focus on in-group vs out-group punishment (key research question)
        """
        if df is None:
            df = self.df
        
        results = {
            "total_punishments": 0,
            "punishment_rate": 0,
            "in_group_punishment": {},
            "out_group_punishment": {},
            "punishment_by_contribution": {},
            "statistical_tests": {}
        }
        
        # Filter for public goods game and punishment actions
        pg_df = df[(df["game"].str.contains("public_goods")) & 
                   (df["action"] == "punish")]
        
        if pg_df.empty:
            return results
        
        results["total_punishments"] = len(pg_df)
        
        # Punishment rate (agents who punished / total agents in punishment rounds)
        punishment_rounds = df[(df["game"].str.contains("public_goods")) & 
                              (df["action"].isin(["punish", "no_punish"]))]
        if not punishment_rounds.empty:
            results["punishment_rate"] = len(pg_df) / len(punishment_rounds)
        
        # Analyze in-group vs out-group punishment for coalition setting
        coalition_df = pg_df[pg_df["setting"].str.contains("coalition")]
        
        if not coalition_df.empty and "punishment_targets" in coalition_df.columns:
            in_group_punishments = []
            out_group_punishments = []
            
            for _, row in coalition_df.iterrows():
                punisher_group = row.get("agent_group", "unknown")
                targets = row.get("punishment_targets", "{}")
                
                try:
                    targets_dict = json.loads(targets) if isinstance(targets, str) else targets
                    
                    for target, amount in targets_dict.items():
                        # Determine if target is in same group
                        target_group = self._get_agent_group_from_name(target)
                        
                        if punisher_group == target_group:
                            in_group_punishments.append(amount)
                        else:
                            out_group_punishments.append(amount)
                except:
                    continue
            
            results["in_group_punishment"] = {
                "count": len(in_group_punishments),
                "mean": np.mean(in_group_punishments) if in_group_punishments else 0,
                "total": sum(in_group_punishments)
            }
            
            results["out_group_punishment"] = {
                "count": len(out_group_punishments),
                "mean": np.mean(out_group_punishments) if out_group_punishments else 0,
                "total": sum(out_group_punishments)
            }
            
            # Statistical test for difference
            if in_group_punishments and out_group_punishments:
                t_stat, p_value = stats.ttest_ind(in_group_punishments, out_group_punishments)
                results["statistical_tests"]["in_vs_out_group"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
        
        # Analyze punishment based on contribution levels
        if "contribution" in df.columns:
            # Get contribution data for punished agents
            contribution_df = df[df["game"].str.contains("public_goods") & 
                                df["action"] == "contribute"]
            
            if not contribution_df.empty:
                # Categorize contributions
                median_contribution = contribution_df["contribution"].median()
                low_contributors = contribution_df[contribution_df["contribution"] < median_contribution]["agent_id"].unique()
                high_contributors = contribution_df[contribution_df["contribution"] >= median_contribution]["agent_id"].unique()
                
                # Count punishments received
                results["punishment_by_contribution"] = {
                    "low_contributors_punished": 0,
                    "high_contributors_punished": 0
                }
                
                # This would require more complex parsing of punishment targets
                # Simplified version here
        
        return results
    
    def calculate_coordination_success(self, df: pd.DataFrame = None) -> Dict:
        """
        Calculate coordination success rate for Stag Hunt games
        Success = all players choosing STAG
        """
        if df is None:
            df = self.df
        
        results = {
            "overall_success_rate": 0,
            "by_setting": {},
            "by_communication": {},
            "success_evolution": {},
            "communication_effectiveness": {}
        }
        
        # Filter for stag hunt games
        sh_df = df[df["game"].str.contains("stag_hunt")]
        
        if sh_df.empty:
            return results
        
        # Group by game, setting, trial, and round to check for unanimous STAG
        grouped = sh_df.groupby(["game", "setting", "trial", "round"])
        
        successful_rounds = 0
        total_rounds = 0
        
        for name, group in grouped:
            if "stag" in str(group["action"].values).lower():
                # Check if all agents chose STAG
                stag_count = group["action"].str.lower().str.contains("stag").sum()
                total_agents = len(group)
                
                if stag_count == total_agents:
                    successful_rounds += 1
                total_rounds += 1
        
        results["overall_success_rate"] = successful_rounds / total_rounds if total_rounds > 0 else 0
        
        # Success rate by setting
        for setting in sh_df["setting"].unique():
            setting_df = sh_df[sh_df["setting"] == setting]
            setting_grouped = setting_df.groupby(["trial", "round"])
            
            setting_success = 0
            setting_total = 0
            
            for name, group in setting_grouped:
                stag_count = group["action"].str.lower().str.contains("stag").sum()
                if stag_count == len(group):
                    setting_success += 1
                setting_total += 1
            
            results["by_setting"][setting] = setting_success / setting_total if setting_total > 0 else 0
        
        # Compare with and without communication
        no_comm_df = sh_df[~sh_df["game"].str.contains("communication")]
        with_comm_df = sh_df[sh_df["game"].str.contains("communication")]
        
        if not no_comm_df.empty:
            results["by_communication"]["without"] = self._calculate_game_success_rate(no_comm_df)
        
        if not with_comm_df.empty:
            results["by_communication"]["with"] = self._calculate_game_success_rate(with_comm_df)
        
        # Success evolution over rounds
        if "round" in sh_df.columns:
            for round_num in sh_df["round"].unique():
                round_df = sh_df[sh_df["round"] == round_num]
                results["success_evolution"][round_num] = self._calculate_game_success_rate(round_df)
        
        return results
    
    def analyze_communication_patterns(self, df: pd.DataFrame = None) -> Dict:
        """
        Analyze communication patterns in Stag Hunt with communication
        Look for emergent conventions and in-group signaling
        """
        if df is None:
            df = self.df
        
        results = {
            "total_communications": 0,
            "unique_words": [],
            "word_frequency": {},
            "word_action_correlation": {},
            "group_convergence": {},
            "emergent_conventions": []
        }
        
        # Filter for communication actions
        comm_df = df[(df["game"].str.contains("communication")) & 
                     (df["action"] == "communicate")]
        
        if comm_df.empty or "communication" not in comm_df.columns:
            return results
        
        results["total_communications"] = len(comm_df)
        
        # Analyze word usage
        all_words = []
        for word in comm_df["communication"].dropna():
            if isinstance(word, str):
                all_words.append(word.upper())
        
        results["unique_words"] = list(set(all_words))
        
        # Word frequency
        from collections import Counter
        word_counts = Counter(all_words)
        results["word_frequency"] = dict(word_counts.most_common(10))
        
        # Correlation between words and subsequent actions
        for word in results["unique_words"][:10]:  # Top 10 words
            word_users = comm_df[comm_df["communication"].str.upper() == word]["agent_id"].values
            
            # Find their next actions
            next_actions = []
            for agent in word_users:
                agent_actions = df[(df["agent_id"] == agent) & 
                                  (df["action"].isin(["STAG", "HARE"]))]
                if not agent_actions.empty:
                    next_actions.extend(agent_actions["action"].values)
            
            if next_actions:
                stag_rate = sum(1 for a in next_actions if "STAG" in str(a).upper()) / len(next_actions)
                results["word_action_correlation"][word] = {
                    "stag_rate": stag_rate,
                    "usage_count": word_counts[word]
                }
        
        # Analyze group convergence in coalition setting
        coalition_comm = comm_df[comm_df["setting"].str.contains("coalition")]
        
        if not coalition_comm.empty and "agent_group" in coalition_comm.columns:
            for group in coalition_comm["agent_group"].unique():
                if group != "none":
                    group_df = coalition_comm[coalition_comm["agent_group"] == group]
                    group_words = [w.upper() for w in group_df["communication"].dropna() if isinstance(w, str)]
                    
                    if group_words:
                        group_word_counts = Counter(group_words)
                        # Check for convergence (dominant word usage)
                        most_common_word, count = group_word_counts.most_common(1)[0]
                        convergence_rate = count / len(group_words)
                        
                        results["group_convergence"][group] = {
                            "dominant_word": most_common_word,
                            "convergence_rate": convergence_rate,
                            "vocabulary_size": len(set(group_words))
                        }
        
        # Identify emergent conventions (words consistently associated with cooperation)
        for word, correlation in results["word_action_correlation"].items():
            if correlation["stag_rate"] > 0.8 and correlation["usage_count"] > 5:
                results["emergent_conventions"].append({
                    "word": word,
                    "cooperation_rate": correlation["stag_rate"],
                    "frequency": correlation["usage_count"]
                })
        
        return results
    
    def _get_agent_group_from_name(self, agent_name: str) -> str:
        """Helper to determine agent group from name"""
        if "Claude" in agent_name:
            return "claude"
        elif "Llama" in agent_name:
            return "llama"
        else:
            return "unknown"
    
    def _calculate_game_success_rate(self, game_df: pd.DataFrame) -> float:
        """Helper to calculate success rate for a game subset"""
        if game_df.empty:
            return 0.0
        
        grouped = game_df.groupby(["trial", "round"])
        successful = 0
        total = 0
        
        for name, group in grouped:
            if len(group) > 0:
                stag_count = group["action"].str.lower().str.contains("stag").sum()
                if stag_count == len(group):
                    successful += 1
                total += 1
        
        return successful / total if total > 0 else 0.0
    
    def calculate_all_metrics(self) -> Dict:
        """Calculate all metrics and return comprehensive results"""
        results = {
            "cooperation_rate": self.calculate_cooperation_rate(),
            "punishment_patterns": self.analyze_punishment_patterns(),
            "coordination_success": self.calculate_coordination_success(),
            "communication_patterns": self.analyze_communication_patterns()
        }
        
        # Add statistical summary
        results["summary"] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, metrics: Dict) -> Dict:
        """Generate a summary of key findings"""
        summary = {
            "key_findings": [],
            "statistical_significance": []
        }
        
        # Check for in-group bias in punishment
        punishment = metrics.get("punishment_patterns", {})
        if punishment.get("statistical_tests", {}).get("in_vs_out_group", {}).get("significant"):
            summary["key_findings"].append("Significant in-group bias detected in punishment behavior")
            summary["statistical_significance"].append({
                "test": "in-group vs out-group punishment",
                "p_value": punishment["statistical_tests"]["in_vs_out_group"]["p_value"]
            })
        
        # Check for communication effectiveness
        coordination = metrics.get("coordination_success", {})
        if coordination.get("by_communication"):
            with_comm = coordination["by_communication"].get("with", 0)
            without_comm = coordination["by_communication"].get("without", 0)
            
            if with_comm > without_comm * 1.2:  # 20% improvement
                summary["key_findings"].append(
                    f"Communication improved coordination by {(with_comm/without_comm - 1)*100:.1f}%"
                )
        
        # Check for emergent conventions
        comm_patterns = metrics.get("communication_patterns", {})
        if comm_patterns.get("emergent_conventions"):
            summary["key_findings"].append(
                f"Found {len(comm_patterns['emergent_conventions'])} emergent communication conventions"
            )
        
        return summary
    
    def save_metrics(self, output_file: str = "metrics_results.json"):
        """Save calculated metrics to JSON file"""
        metrics = self.calculate_all_metrics()
        
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        metrics = convert_types(metrics)
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logging.info(f"Saved metrics to {output_file}")
        return metrics


def main():
    """Main entry point for metrics calculation"""
    import argparse
    from pathlib import Path
    from parser import ResultsParser
    
    parser = argparse.ArgumentParser(description="Calculate metrics from experimental results")
    parser.add_argument("--input", default="parsed_results.csv", help="Input CSV file")
    parser.add_argument("--output", default="metrics_results.json", help="Output JSON file")
    parser.add_argument("--results-dir", default="results", help="Results directory (if no input CSV)")
    
    args = parser.parse_args()
    
    # Load data
    if args.input and Path(args.input).exists():
        df = pd.read_csv(args.input)
    else:
        # Parse from results directory
        results_parser = ResultsParser(args.results_dir)
        df = results_parser.parse_all_results()
    
    if df.empty:
        print("No data to analyze")
        return
    
    # Calculate metrics
    calculator = MetricsCalculator(df)
    metrics = calculator.save_metrics(args.output)
    
    # Print summary
    print("\n=== Key Metrics ===")
    print(f"Overall cooperation rate: {metrics['cooperation_rate']['overall']:.2%}")
    print(f"Coordination success rate: {metrics['coordination_success']['overall_success_rate']:.2%}")
    print(f"Total punishments: {metrics['punishment_patterns']['total_punishments']}")
    
    if metrics["summary"]["key_findings"]:
        print("\n=== Key Findings ===")
        for finding in metrics["summary"]["key_findings"]:
            print(f"  • {finding}")


if __name__ == "__main__":
    main()