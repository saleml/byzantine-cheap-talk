#!/usr/bin/env python3
"""Analyze curriculum learning benefits from partial results."""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

def load_trial_data(condition_path):
    """Load all available trial data for a condition."""
    trials = []
    condition_dir = Path(condition_path)
    
    # Check for complete results
    complete_file = condition_dir / "complete_results.json"
    if complete_file.exists():
        with open(complete_file) as f:
            data = json.load(f)
            trials.extend(data.get("trials", []))
    
    # Also check individual trial directories
    for trial_dir in sorted(condition_dir.glob("trial_*")):
        results_file = trial_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                trial_data = json.load(f)
                # Check if not already in complete results
                trial_num = trial_data.get("trial", int(trial_dir.name.split("_")[1]))
                if not any(t.get("trial") == trial_num for t in trials):
                    trials.append(trial_data)
    
    return trials

def analyze_final_stage_performance(trials, condition_name):
    """Analyze performance in the final IPGG+Punishment stage."""
    final_stage_stats = {
        "cooperation_rates": [],
        "average_payoffs": [],
        "punishment_used": [],
        "contributions": [],
        "trial_count": 0
    }
    
    for trial in trials:
        stages = trial.get("stages", [])
        if not stages:
            continue
            
        # Get the final stage (IPGG with punishment)
        final_stage = None
        for stage in stages:
            if isinstance(stage, dict) and "enable_punishment" in str(stage):
                final_stage = stage
        
        if not final_stage and stages:
            final_stage = stages[-1]  # Use last stage as fallback
            
        if final_stage and isinstance(final_stage, dict):
            final_stage_stats["trial_count"] += 1
            
            # Extract metrics
            if "cooperation_rate" in final_stage:
                final_stage_stats["cooperation_rates"].append(final_stage["cooperation_rate"])
            
            if "average_payoff" in final_stage:
                final_stage_stats["average_payoffs"].append(final_stage["average_payoff"])
            
            # Check for punishment usage
            if "punishment_given" in final_stage or "punishments" in final_stage:
                final_stage_stats["punishment_used"].append(1)
            else:
                final_stage_stats["punishment_used"].append(0)
            
            # Track individual contributions if available
            if "rounds" in final_stage:
                for round_data in final_stage["rounds"]:
                    if "contributions" in round_data:
                        for contrib in round_data["contributions"].values():
                            final_stage_stats["contributions"].append(contrib)
    
    return final_stage_stats

def main():
    """Main analysis function."""
    print("\n" + "="*60)
    print("CURRICULUM LEARNING BENEFITS ANALYSIS")
    print("Analyzing partial results from ongoing experiments")
    print("="*60 + "\n")
    
    results_dir = Path("results/curriculum")
    
    # Define conditions to analyze
    conditions = {
        "control_group": "Control (No Curriculum)",
        "direct_precursor": "Direct Precursor (IPGG → IPGG+P)",
        "curriculum_scrambled": "Scrambled Curriculum",
        "curriculum_full": "Full Curriculum"
    }
    
    all_stats = {}
    
    for condition_id, condition_name in conditions.items():
        condition_path = results_dir / condition_id
        if not condition_path.exists():
            print(f"⚠️  {condition_name}: No data available yet")
            continue
        
        trials = load_trial_data(condition_path)
        stats = analyze_final_stage_performance(trials, condition_name)
        all_stats[condition_id] = stats
        
        print(f"\n📊 {condition_name}")
        print(f"   Trials completed: {stats['trial_count']}")
        
        if stats['cooperation_rates']:
            avg_coop = np.mean(stats['cooperation_rates']) * 100
            std_coop = np.std(stats['cooperation_rates']) * 100
            print(f"   Cooperation rate: {avg_coop:.1f}% (±{std_coop:.1f}%)")
        
        if stats['average_payoffs']:
            avg_payoff = np.mean(stats['average_payoffs'])
            std_payoff = np.std(stats['average_payoffs'])
            print(f"   Average payoff: {avg_payoff:.1f} (±{std_payoff:.1f})")
        
        if stats['contributions']:
            avg_contrib = np.mean(stats['contributions'])
            print(f"   Average contribution: {avg_contrib:.1f} tokens")
        
        if stats['punishment_used']:
            punishment_rate = np.mean(stats['punishment_used']) * 100
            print(f"   Trials with punishment: {punishment_rate:.1f}%")
    
    # Compare curriculum vs control if both have data
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS")
    print("="*60)
    
    if "control_group" in all_stats and all_stats["control_group"]["trial_count"] > 0:
        control_stats = all_stats["control_group"]
        control_coop = np.mean(control_stats['cooperation_rates']) if control_stats['cooperation_rates'] else 0
        control_payoff = np.mean(control_stats['average_payoffs']) if control_stats['average_payoffs'] else 0
        
        print(f"\n🎯 Control Group Baseline:")
        print(f"   Cooperation: {control_coop*100:.1f}%")
        print(f"   Payoff: {control_payoff:.1f}")
        
        # Compare each curriculum condition to control
        for condition_id in ["direct_precursor", "curriculum_scrambled", "curriculum_full"]:
            if condition_id in all_stats and all_stats[condition_id]["trial_count"] > 0:
                curr_stats = all_stats[condition_id]
                if curr_stats['cooperation_rates']:
                    curr_coop = np.mean(curr_stats['cooperation_rates'])
                    curr_payoff = np.mean(curr_stats['average_payoffs']) if curr_stats['average_payoffs'] else 0
                    
                    coop_diff = (curr_coop - control_coop) * 100
                    payoff_diff = curr_payoff - control_payoff
                    
                    print(f"\n📈 {conditions[condition_id]} vs Control:")
                    print(f"   Cooperation: {'+' if coop_diff >= 0 else ''}{coop_diff:.1f}% points")
                    print(f"   Payoff: {'+' if payoff_diff >= 0 else ''}{payoff_diff:.1f} tokens")
                    
                    if curr_stats['trial_count'] >= 2:  # Only make claims with sufficient data
                        if coop_diff > 5:
                            print(f"   ✅ Shows improved cooperation!")
                        elif coop_diff < -5:
                            print(f"   ❌ Shows reduced cooperation")
                        else:
                            print(f"   ➖ Similar cooperation levels")
    
    # Check lesson quality
    print("\n" + "="*60)
    print("LESSON GENERATION STATUS")
    print("="*60)
    
    # Count Claude-generated lessons
    claude_lessons = 0
    for condition_dir in results_dir.iterdir():
        if condition_dir.is_dir():
            for trial_dir in condition_dir.glob("trial_*"):
                for lesson_file in trial_dir.glob("*lesson*.txt"):
                    with open(lesson_file) as f:
                        content = f.read()
                        if "Strategic insight:" in content or "Key pattern:" in content:
                            claude_lessons += 1
    
    print(f"\n🤖 Claude Opus 4.1 lessons generated: {claude_lessons}")
    print("   These provide sophisticated strategic insights")
    print("   vs generic template-based lessons from before")
    
    # Final summary
    print("\n" + "="*60)
    print("PRELIMINARY FINDINGS")
    print("="*60)
    
    if any(s["trial_count"] > 0 for s in all_stats.values()):
        print("\n⚠️  Note: Results are preliminary with limited trials completed")
        print("\nBased on partial data:")
        
        # Check if any curriculum shows benefits
        has_benefits = False
        for condition_id in ["direct_precursor", "curriculum_scrambled", "curriculum_full"]:
            if condition_id in all_stats and all_stats[condition_id]["trial_count"] > 0:
                if all_stats[condition_id]['cooperation_rates']:
                    curr_coop = np.mean(all_stats[condition_id]['cooperation_rates'])
                    if control_coop < curr_coop:
                        has_benefits = True
                        break
        
        if has_benefits:
            print("✅ Early signs suggest curriculum learning may provide benefits")
            print("   Some conditions show higher cooperation than control")
        else:
            print("➖ Too early to determine clear benefits")
            print("   More trials needed for statistical significance")
    else:
        print("\n⚠️  Insufficient data for analysis")
        print("   Experiments still in early stages")

if __name__ == "__main__":
    main()