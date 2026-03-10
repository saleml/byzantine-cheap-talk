#!/usr/bin/env python
"""
Qualitative analysis of experimental results
Focuses on PGG failure modes and Stag Hunt communication patterns
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import re
import logging

logging.basicConfig(level=logging.INFO)


class QualitativeAnalyzer:
    """Analyzer for qualitative aspects of game results"""
    
    # PGG Failure Taxonomy based on game theory literature
    PGG_FAILURE_TAXONOMY = {
        "FEAR": "Fear of exploitation or being the only contributor",
        "GREED": "Explicit desire to free-ride on others' contributions",
        "RATIONAL_CHOICE": "Citing the one-shot nature or Nash equilibrium",
        "CONFUSION": "Misunderstanding the game rules or payoff structure",
        "LACK_OF_TRUST": "Explicit distrust of other players",
        "DEFECTION_DEFAULT": "Default to non-cooperation without clear reasoning"
    }
    
    def __init__(self, data_file: str = "analysis/consolidated_data.jsonl"):
        self.data_file = Path(data_file)
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """Load consolidated JSONL data"""
        data = []
        if self.data_file.exists():
            with open(self.data_file) as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logging.error(f"Error parsing line: {e}")
        return data
    
    def analyze_pgg_contributions(self) -> Dict:
        """Analyze Public Goods Game contribution patterns"""
        pgg_analysis = {
            "total_trials": 0,
            "cooperation_rate": 0.0,
            "avg_contribution_by_model": {},
            "contribution_distribution": defaultdict(int),
            "defection_patterns": defaultdict(list),
            "round_dynamics": defaultdict(list)
        }
        
        pgg_trials = [d for d in self.data if "public_goods" in d.get("game_id", "")]
        pgg_analysis["total_trials"] = len(pgg_trials)
        
        model_contributions = defaultdict(list)
        
        for trial in pgg_trials:
            pgg_analysis["cooperation_rate"] += trial.get("cooperation_rate", 0)
            
            # Analyze round-by-round contributions
            for round_data in trial.get("rounds_data", []):
                if "contributions" in round_data:
                    contributions = round_data["contributions"]
                    for agent, amount in contributions.items():
                        # Categorize contribution levels
                        if amount <= 10:
                            category = "minimal"
                        elif amount <= 15:
                            category = "moderate"
                        else:
                            category = "high"
                        pgg_analysis["contribution_distribution"][category] += 1
                        
                        # Track by round
                        round_num = round_data.get("round", 0)
                        pgg_analysis["round_dynamics"][round_num].append(amount)
                        
                        # Map to model if available
                        model_avgs = trial.get("model_family_averages", {})
                        for model in model_avgs:
                            if agent in str(model_avgs):
                                model_contributions[model].append(amount)
                                break
        
        # Calculate averages
        if pgg_trials:
            pgg_analysis["cooperation_rate"] /= len(pgg_trials)
        
        for model, contribs in model_contributions.items():
            if contribs:
                pgg_analysis["avg_contribution_by_model"][model] = sum(contribs) / len(contribs)
        
        # Calculate round averages
        for round_num, values in pgg_analysis["round_dynamics"].items():
            pgg_analysis["round_dynamics"][round_num] = sum(values) / len(values) if values else 0
        
        return pgg_analysis
    
    def classify_pgg_failure_modes(self, sample_size: int = 50) -> Dict:
        """
        Classify PGG failure modes based on actual rationales
        """
        failure_analysis = {
            "taxonomy_counts": {key: 0 for key in self.PGG_FAILURE_TAXONOMY},
            "model_failure_patterns": defaultdict(lambda: defaultdict(int)),
            "coded_rationales": [],
            "sample_rationales": defaultdict(list)
        }
        
        pgg_trials = [d for d in self.data if "public_goods" in d.get("game_id", "")]
        
        rationale_count = 0
        for trial in pgg_trials[:sample_size]:
            for round_data in trial.get("rounds_data", []):
                contributions = round_data.get("contributions", {})
                rationales = round_data.get("rationales", {})
                
                for agent, amount in contributions.items():
                    if amount <= 15:  # Focus on low contributions
                        rationale = rationales.get(agent, "")
                        
                        # Classify based on rationale content
                        failure_mode = self._classify_rationale(rationale, amount)
                        
                        failure_analysis["taxonomy_counts"][failure_mode] += 1
                        
                        # Track by model family
                        model_avgs = trial.get("model_family_averages", {})
                        for model in model_avgs:
                            failure_analysis["model_failure_patterns"][model][failure_mode] += 1
                        
                        # Store coded rationale
                        coded_entry = {
                            "agent": agent,
                            "contribution": amount,
                            "failure_mode": failure_mode,
                            "rationale": rationale[:500],  # First 500 chars
                            "trial": trial.get("trial_id"),
                            "round": round_data.get("round")
                        }
                        failure_analysis["coded_rationales"].append(coded_entry)
                        
                        # Keep sample rationales for each category
                        if len(failure_analysis["sample_rationales"][failure_mode]) < 3:
                            failure_analysis["sample_rationales"][failure_mode].append(rationale)
                        
                        rationale_count += 1
        
        failure_analysis["total_rationales_coded"] = rationale_count
        return failure_analysis
    
    def _classify_rationale(self, rationale: str, contribution: int) -> str:
        """
        Classify a single rationale into failure taxonomy
        """
        if not rationale:
            return "DEFECTION_DEFAULT"
        
        rationale_lower = rationale.lower()
        
        # Check for specific patterns
        if any(phrase in rationale_lower for phrase in 
               ["nash equilibrium", "dominant strategy", "rational choice", 
                "maximize my personal", "optimal strategy", "game theory"]):
            return "RATIONAL_CHOICE"
        
        if any(phrase in rationale_lower for phrase in 
               ["fear", "exploit", "sucker", "taken advantage", "risk"]):
            return "FEAR"
        
        if any(phrase in rationale_lower for phrase in 
               ["free ride", "free-ride", "others will contribute", 
                "benefit from others", "let others"]):
            return "GREED"
        
        if any(phrase in rationale_lower for phrase in 
               ["don't trust", "lack of trust", "distrust", "cannot trust", 
                "others won't", "others will defect"]):
            return "LACK_OF_TRUST"
        
        if any(phrase in rationale_lower for phrase in 
               ["confused", "unclear", "don't understand", "not sure"]):
            return "CONFUSION"
        
        # Default classification based on contribution level
        if contribution == 0:
            return "GREED"
        elif contribution <= 10:
            return "RATIONAL_CHOICE"
        else:
            return "LACK_OF_TRUST"
    
    def analyze_stag_hunt_communication(self) -> Dict:
        """Analyze Stag Hunt communication patterns and effectiveness"""
        comm_analysis = {
            "total_trials": 0,
            "cooperation_with_comm": 0.0,
            "cooperation_without_comm": 0.0,
            "top_signals": Counter(),
            "signal_effectiveness": {},
            "communication_sequences": [],
            "initiator_success": defaultdict(int),
            "emergent_lexicon": set()
        }
        
        # Separate trials with and without communication
        sh_comm_trials = [d for d in self.data if "stag_hunt_communication" in d.get("game_id", "")]
        sh_no_comm_trials = [d for d in self.data if d.get("game_id", "") == "stag_hunt"]
        
        comm_analysis["total_trials"] = len(sh_comm_trials)
        
        # Calculate cooperation rates
        if sh_comm_trials:
            comm_analysis["cooperation_with_comm"] = sum(
                t.get("cooperation_rate", 0) for t in sh_comm_trials
            ) / len(sh_comm_trials)
        
        if sh_no_comm_trials:
            comm_analysis["cooperation_without_comm"] = sum(
                t.get("cooperation_rate", 0) for t in sh_no_comm_trials
            ) / len(sh_no_comm_trials)
        
        # Analyze communication content
        all_signals = []
        signal_outcomes = defaultdict(lambda: {"success": 0, "total": 0})
        
        for trial in sh_comm_trials:
            # Use communication_analysis if available
            if "communication_analysis" in trial:
                word_freq = trial["communication_analysis"].get("word_frequency", {})
                for word, count in word_freq.items():
                    comm_analysis["top_signals"][word.lower()] += count
                    all_signals.extend([word.lower()] * count)
            
            # Analyze round-by-round communications
            for round_data in trial.get("rounds_data", []):
                if "communications" in round_data:
                    comms = round_data["communications"]
                    choices = round_data.get("choices", {})
                    cooperated = round_data.get("all_cooperated", False)
                    
                    # Track communication sequences
                    comm_sequence = list(comms.values())
                    comm_analysis["communication_sequences"].append({
                        "signals": comm_sequence,
                        "cooperated": cooperated,
                        "round": round_data.get("round", 0)
                    })
                    
                    # Track signal effectiveness
                    for signal in comm_sequence:
                        if signal:
                            normalized_signal = signal.lower().strip()
                            signal_outcomes[normalized_signal]["total"] += 1
                            if cooperated:
                                signal_outcomes[normalized_signal]["success"] += 1
                    
                    # Identify initiator (first non-empty signal)
                    for agent, signal in comms.items():
                        if signal:
                            comm_analysis["initiator_success"][agent] += 1 if cooperated else 0
                            break
        
        # Calculate signal effectiveness
        for signal, outcomes in signal_outcomes.items():
            if outcomes["total"] > 0:
                effectiveness = outcomes["success"] / outcomes["total"]
                comm_analysis["signal_effectiveness"][signal] = {
                    "effectiveness": effectiveness,
                    "occurrences": outcomes["total"],
                    "successes": outcomes["success"]
                }
        
        # Identify emergent lexicon (frequently used successful signals)
        for signal, data in comm_analysis["signal_effectiveness"].items():
            if data["effectiveness"] >= 0.7 and data["occurrences"] >= 5:
                comm_analysis["emergent_lexicon"].add(signal)
        
        comm_analysis["emergent_lexicon"] = list(comm_analysis["emergent_lexicon"])
        comm_analysis["top_signals"] = dict(comm_analysis["top_signals"].most_common(10))
        
        return comm_analysis
    
    def analyze_communication_evolution(self) -> Dict:
        """Analyze how communication strategies evolved over rounds"""
        evolution_analysis = {
            "signal_convergence": {},
            "round_by_round_diversity": [],
            "model_specific_signals": defaultdict(Counter),
            "cross_model_adoption": []
        }
        
        sh_comm_trials = [d for d in self.data if "stag_hunt_communication" in d.get("game_id", "")]
        
        round_signals = defaultdict(list)
        model_signals = defaultdict(lambda: defaultdict(list))
        
        for trial in sh_comm_trials:
            model_avgs = trial.get("model_family_averages", {})
            
            for round_data in trial.get("rounds_data", []):
                if "communications" in round_data:
                    round_num = round_data.get("round", 0)
                    comms = round_data["communications"]
                    
                    # Track signals by round
                    for agent, signal in comms.items():
                        if signal:
                            normalized = signal.lower().strip()
                            round_signals[round_num].append(normalized)
                            
                            # Try to map to model
                            for model in model_avgs:
                                if agent in str(model_avgs):
                                    model_signals[model][round_num].append(normalized)
                                    evolution_analysis["model_specific_signals"][model][normalized] += 1
                                    break
        
        # Calculate diversity metrics per round
        for round_num in sorted(round_signals.keys()):
            signals = round_signals[round_num]
            if signals:
                unique_signals = len(set(signals))
                total_signals = len(signals)
                diversity = unique_signals / total_signals if total_signals > 0 else 0
                
                evolution_analysis["round_by_round_diversity"].append({
                    "round": round_num,
                    "diversity": diversity,
                    "unique_signals": unique_signals,
                    "total_signals": total_signals
                })
        
        # Check for signal convergence
        if len(round_signals) > 1:
            early_rounds = [s for r in range(1, 3) for s in round_signals.get(r, [])]
            late_rounds = [s for r in range(4, 7) for s in round_signals.get(r, [])]
            
            if early_rounds and late_rounds:
                early_top = Counter(early_rounds).most_common(3)
                late_top = Counter(late_rounds).most_common(3)
                
                evolution_analysis["signal_convergence"] = {
                    "early_top_signals": dict(early_top),
                    "late_top_signals": dict(late_top),
                    "convergence_observed": early_top[0][0] == late_top[0][0] if early_top and late_top else False
                }
        
        # Convert model signals to regular dict
        evolution_analysis["model_specific_signals"] = {
            model: dict(counter.most_common(5))
            for model, counter in evolution_analysis["model_specific_signals"].items()
        }
        
        return evolution_analysis
    
    def generate_summary_report(self) -> Dict:
        """Generate comprehensive summary report"""
        report = {
            "pgg_analysis": self.analyze_pgg_contributions(),
            "pgg_failure_modes": self.classify_pgg_failure_modes(),
            "stag_hunt_communication": self.analyze_stag_hunt_communication(),
            "communication_evolution": self.analyze_communication_evolution()
        }
        
        # Add key insights
        report["key_insights"] = {
            "pgg_cooperation_rate": report["pgg_analysis"]["cooperation_rate"],
            "stag_hunt_comm_benefit": (
                report["stag_hunt_communication"]["cooperation_with_comm"] - 
                report["stag_hunt_communication"]["cooperation_without_comm"]
            ),
            "dominant_failure_mode": max(
                report["pgg_failure_modes"]["taxonomy_counts"].items(),
                key=lambda x: x[1]
            )[0] if report["pgg_failure_modes"]["taxonomy_counts"] else "UNKNOWN",
            "emergent_lexicon_size": len(report["stag_hunt_communication"]["emergent_lexicon"]),
            "top_effective_signal": max(
                report["stag_hunt_communication"]["signal_effectiveness"].items(),
                key=lambda x: x[1]["effectiveness"]
            )[0] if report["stag_hunt_communication"]["signal_effectiveness"] else None
        }
        
        return report
    
    def save_report(self, report: Dict, output_file: str = "analysis/qualitative_report.json"):
        """Save analysis report to JSON"""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logging.info(f"Saved qualitative analysis report to {output_path}")
        return output_path
    
    def print_summary(self, report: Dict):
        """Print human-readable summary"""
        print("\n" + "="*60)
        print("QUALITATIVE ANALYSIS SUMMARY")
        print("="*60)
        
        print("\n1. PUBLIC GOODS GAME ANALYSIS")
        print("-" * 40)
        pgg = report["pgg_analysis"]
        print(f"Total trials analyzed: {pgg['total_trials']}")
        print(f"Overall cooperation rate: {pgg['cooperation_rate']:.2%}")
        print(f"Contribution distribution:")
        for level, count in pgg["contribution_distribution"].items():
            print(f"  {level}: {count}")
        
        print("\n2. PGG FAILURE MODE TAXONOMY")
        print("-" * 40)
        failure = report["pgg_failure_modes"]
        total_failures = sum(failure["taxonomy_counts"].values())
        for mode, count in failure["taxonomy_counts"].items():
            pct = (count / total_failures * 100) if total_failures > 0 else 0
            print(f"  {mode}: {count} ({pct:.1f}%)")
            print(f"    - {self.PGG_FAILURE_TAXONOMY[mode]}")
        
        print("\n3. STAG HUNT COMMUNICATION ANALYSIS")
        print("-" * 40)
        sh = report["stag_hunt_communication"]
        print(f"Cooperation WITHOUT communication: {sh['cooperation_without_comm']:.2%}")
        print(f"Cooperation WITH communication: {sh['cooperation_with_comm']:.2%}")
        print(f"Communication benefit: +{sh['cooperation_with_comm'] - sh['cooperation_without_comm']:.2%}")
        
        print(f"\nTop 5 communication signals:")
        for signal, count in list(sh["top_signals"].items())[:5]:
            effectiveness = sh["signal_effectiveness"].get(signal, {}).get("effectiveness", 0)
            print(f"  '{signal}': {count} uses, {effectiveness:.2%} success rate")
        
        print(f"\nEmergent lexicon: {sh['emergent_lexicon']}")
        
        print("\n4. COMMUNICATION EVOLUTION")
        print("-" * 40)
        evo = report["communication_evolution"]
        if evo["signal_convergence"]:
            print(f"Convergence observed: {evo['signal_convergence'].get('convergence_observed', False)}")
            if "early_top_signals" in evo["signal_convergence"]:
                print(f"Early rounds top signal: {list(evo['signal_convergence']['early_top_signals'].keys())[0] if evo['signal_convergence']['early_top_signals'] else 'N/A'}")
                print(f"Late rounds top signal: {list(evo['signal_convergence']['late_top_signals'].keys())[0] if evo['signal_convergence']['late_top_signals'] else 'N/A'}")
        
        print("\n5. KEY INSIGHTS")
        print("-" * 40)
        insights = report["key_insights"]
        print(f"PGG Cooperation Rate: {insights['pgg_cooperation_rate']:.2%}")
        print(f"Communication Benefit: +{insights['stag_hunt_comm_benefit']:.2%}")
        print(f"Dominant Failure Mode: {insights['dominant_failure_mode']}")
        print(f"Emergent Lexicon Size: {insights['emergent_lexicon_size']} signals")
        if insights["top_effective_signal"]:
            print(f"Most Effective Signal: '{insights['top_effective_signal']}'")
        
        print("\n" + "="*60)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qualitative analysis of game results")
    parser.add_argument("--data", default="analysis/consolidated_data.jsonl",
                       help="Path to consolidated data file")
    parser.add_argument("--output", default="analysis/qualitative_report.json",
                       help="Output report file")
    parser.add_argument("--sample-size", type=int, default=50,
                       help="Sample size for manual coding")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = QualitativeAnalyzer(args.data)
    report = analyzer.generate_summary_report()
    
    # Save report
    analyzer.save_report(report, args.output)
    
    # Print summary
    analyzer.print_summary(report)


if __name__ == "__main__":
    main()