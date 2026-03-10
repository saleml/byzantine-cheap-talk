#!/usr/bin/env python
"""
Unified Parser for both pilot study and curriculum experimental results
Handles different data structures and outputs a single consolidated dataset
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class UnifiedResultsParser:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise ValueError(f"Results directory {results_dir} does not exist")
        
        # Game rules for validation
        self.game_rules = {
            "public_goods": {
                "endowment": 20,
                "multiplier": 1.5,
                "min_contribution": 0,
                "max_contribution": 20
            },
            "stag_hunt": {
                "payoffs": {
                    ("stag", "stag"): (4, 4),
                    ("stag", "hare"): (0, 3),
                    ("hare", "stag"): (3, 0),
                    ("hare", "hare"): (1, 1)
                }
            }
        }
        
        self.validation_errors = []
    
    def parse_all_data(self, include_pilot: bool = True, include_curriculum: bool = True) -> Tuple[List[Dict], List[str]]:
        """
        Parse all experimental data (pilot and/or curriculum)
        Returns: (consolidated_records, validation_errors)
        """
        all_records = []
        
        if include_pilot:
            logging.info("Parsing pilot study data...")
            pilot_records = self._parse_pilot_data()
            all_records.extend(pilot_records)
            logging.info(f"Parsed {len(pilot_records)} pilot study trials")
        
        if include_curriculum:
            logging.info("Parsing curriculum experiment data...")
            curriculum_records = self._parse_curriculum_data()
            all_records.extend(curriculum_records)
            logging.info(f"Parsed {len(curriculum_records)} curriculum experiment trials")
        
        # Validate all records
        logging.info("Validating data integrity...")
        self._validate_all_records(all_records)
        
        return all_records, self.validation_errors
    
    def _parse_pilot_data(self) -> List[Dict]:
        """Parse pilot study data from results/{game}/{setting}/trial_XX/"""
        records = []
        
        # Games in pilot study
        pilot_games = ["public_goods", "stag_hunt", "stag_hunt_communication"]
        
        for game_name in pilot_games:
            game_dir = self.results_dir / game_name
            if not game_dir.exists():
                logging.warning(f"Game directory {game_dir} not found")
                continue
            
            for setting_dir in game_dir.iterdir():
                if not setting_dir.is_dir():
                    continue
                
                setting = setting_dir.name
                
                for trial_dir in setting_dir.iterdir():
                    if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                        continue
                    
                    trial_num = int(trial_dir.name.split("_")[1])
                    data_file = trial_dir / "experiment_data.json"
                    
                    if data_file.exists():
                        try:
                            with open(data_file) as f:
                                exp_data = json.load(f)
                            
                            record = self._create_pilot_record(
                                exp_data, game_name, setting, trial_num
                            )
                            records.append(record)
                        except Exception as e:
                            logging.error(f"Error parsing {data_file}: {e}")
                            self.validation_errors.append(f"Parse error in {data_file}: {e}")
        
        return records
    
    def _parse_curriculum_data(self) -> List[Dict]:
        """Parse curriculum experiment data from results/curriculum/{condition}/trial_XX/"""
        records = []
        
        curriculum_dir = self.results_dir / "curriculum"
        if not curriculum_dir.exists():
            logging.warning("No curriculum directory found")
            return records
        
        # Curriculum conditions
        conditions = ["full_curriculum", "scrambled_curriculum", "direct_precursor", "control_group"]
        
        for condition in conditions:
            condition_dir = curriculum_dir / condition
            if not condition_dir.exists():
                logging.warning(f"Condition directory {condition_dir} not found")
                continue
            
            for trial_dir in condition_dir.iterdir():
                if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                    continue
                
                trial_num = int(trial_dir.name.split("_")[1])
                results_file = trial_dir / "results.json"
                
                if results_file.exists():
                    try:
                        with open(results_file) as f:
                            curriculum_data = json.load(f)
                        
                        # Create records for each stage in the curriculum
                        stage_records = self._create_curriculum_records(
                            curriculum_data, condition, trial_num
                        )
                        records.extend(stage_records)
                    except Exception as e:
                        logging.error(f"Error parsing {results_file}: {e}")
                        self.validation_errors.append(f"Parse error in {results_file}: {e}")
        
        return records
    
    def _create_pilot_record(self, exp_data: Dict, game_id: str, setting: str, trial_id: int) -> Dict:
        """Create a consolidated record from pilot study data"""
        record = {
            "experiment_type": "pilot",
            "game_id": game_id,
            "trial_id": trial_id,
            "setting": setting,
            "curriculum_condition": None,
            "stage_num": None,
            "stage_name": None,
            "timestamp": exp_data.get("metadata", {}).get("timestamp"),
            "duration": exp_data.get("duration", 0),
            "cooperation_rate": exp_data.get("cooperation_rate", 0),
            "average_payoff": exp_data.get("average_payoff", 0),
            "total_rounds": exp_data.get("total_rounds", 1),
            "model_family_averages": exp_data.get("model_family_averages", {}),
            "rounds_data": [],
            "agents": []
        }
        
        # Extract agent information
        for agent_config in exp_data.get("config", {}).get("agents", []):
            record["agents"].append({
                "name": agent_config["name"],
                "model": agent_config["llm"]["model"]
            })
        
        # Extract round-by-round data
        for round_data in exp_data.get("rounds_data", []):
            round_record = {
                "round": round_data.get("round", 1),
                "actions": {},
                "rationales": {},
                "payoffs": {},
                "communications": {}
            }
            
            # Extract rationales from full_decisions if available
            if "full_decisions" in round_data:
                for agent, decision in round_data["full_decisions"].items():
                    round_record["rationales"][agent] = decision.get("reasoning", "")
                    
            # Handle different game types
            if "public_goods" in game_id:
                round_record["contributions"] = round_data.get("contributions", {})
                round_record["punishments"] = round_data.get("punishments", {})
                round_record["payoffs"] = round_data.get("payoffs", {})
                
                # Set actions based on contributions
                for agent, contribution in round_record["contributions"].items():
                    round_record["actions"][agent] = f"contribute_{contribution}"
            
            elif "stag_hunt" in game_id:
                if "communication" in game_id:
                    round_record["communications"] = round_data.get("communications", {})
                round_record["choices"] = round_data.get("choices", {})
                round_record["payoffs"] = round_data.get("payoffs", {})
                round_record["all_cooperated"] = round_data.get("all_cooperated", False)
                
                # Set actions based on choices
                for agent, choice in round_record.get("choices", {}).items():
                    round_record["actions"][agent] = choice
            
            record["rounds_data"].append(round_record)
        
        return record
    
    def _create_curriculum_records(self, curriculum_data: Dict, condition: str, trial_id: int) -> List[Dict]:
        """Create consolidated records from curriculum experiment data"""
        records = []
        
        # Each stage becomes a separate record for analysis
        for stage_idx, stage_data in enumerate(curriculum_data.get("stages", [])):
            record = {
                "experiment_type": "curriculum",
                "game_id": stage_data.get("game", "unknown"),
                "trial_id": trial_id,
                "setting": condition,
                "curriculum_condition": condition,
                "stage_num": stage_data.get("stage", stage_idx + 1),
                "stage_name": stage_data.get("stage_name", f"Stage {stage_idx + 1}"),
                "timestamp": curriculum_data.get("timestamp"),
                "duration": stage_data.get("duration", 0),
                "cooperation_rate": stage_data.get("cooperation_rate", 0),
                "average_payoff": stage_data.get("average_payoff", 0),
                "total_rounds": stage_data.get("rounds_played", 0),
                "model_family_averages": stage_data.get("model_family_averages", {}),
                "cumulative_payoffs": stage_data.get("cumulative_payoffs", {}),
                "cooperation_trajectory": stage_data.get("cooperation_trajectory", []),
                "lessons_learned": curriculum_data.get("lessons_learned", []),
                "rounds_data": [],
                "agents": []
            }
            
            # Extract round-by-round data from rounds_data
            for round_data in stage_data.get("rounds_data", []):
                round_record = {
                    "round": round_data.get("round", 0),
                    "actions": {},
                    "rationales": {},
                    "payoffs": round_data.get("payoffs", {}),
                    "communications": {}
                }
                
                # Handle different game types
                game_type = stage_data.get("game", "").lower()
                
                if "prisoners" in game_type or "ipd" in game_type:
                    # Prisoner's Dilemma games
                    round_record["choices"] = round_data.get("actions", {})
                    for agent, choice in round_record["choices"].items():
                        round_record["actions"][agent] = choice
                        
                elif "public" in game_type or "ipgg" in game_type:
                    # Public Goods games
                    round_record["contributions"] = round_data.get("contributions", {})
                    round_record["punishments"] = round_data.get("punishments", {})
                    
                    for agent, contribution in round_record.get("contributions", {}).items():
                        round_record["actions"][agent] = f"contribute_{contribution}"
                
                # Extract rationales if available
                if "rationales" in round_data:
                    round_record["rationales"] = round_data["rationales"]
                elif "full_decisions" in round_data:
                    for agent, decision in round_data["full_decisions"].items():
                        round_record["rationales"][agent] = decision.get("reasoning", "")
                
                record["rounds_data"].append(round_record)
            
            records.append(record)
        
        return records
    
    def _validate_all_records(self, records: List[Dict]) -> None:
        """Validate data integrity for all records"""
        for idx, record in enumerate(records):
            errors = self._validate_record(record)
            if errors:
                error_msg = f"Record {idx} ({record.get('experiment_type')}, {record.get('game_id')}, trial {record.get('trial_id')}): {'; '.join(errors)}"
                self.validation_errors.append(error_msg)
    
    def _validate_record(self, record: Dict) -> List[str]:
        """Validate a single record for data integrity"""
        errors = []
        
        # Check required fields
        required_fields = ["experiment_type", "game_id", "trial_id", "setting"]
        for field in required_fields:
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate game-specific rules
        game_id = record.get("game_id", "").lower()
        
        if "public_goods" in game_id:
            errors.extend(self._validate_public_goods_record(record))
        elif "stag_hunt" in game_id:
            errors.extend(self._validate_stag_hunt_record(record))
        elif "prisoners" in game_id or "ipd" in game_id:
            errors.extend(self._validate_prisoners_dilemma_record(record))
        
        # Check for missing rationales in newer experiments
        if record.get("experiment_type") == "curriculum" or record.get("trial_id", 0) > 10:
            has_rationales = any(
                bool(round_data.get("rationales"))
                for round_data in record.get("rounds_data", [])
            )
            if not has_rationales:
                errors.append("No rationales found (expected for newer experiments)")
        
        return errors
    
    def _validate_public_goods_record(self, record: Dict) -> List[str]:
        """Validate Public Goods Game data"""
        errors = []
        rules = self.game_rules["public_goods"]
        
        for round_data in record.get("rounds_data", []):
            contributions = round_data.get("contributions", {})
            payoffs = round_data.get("payoffs", {})
            
            # Check contribution bounds
            for agent, contribution in contributions.items():
                if not (rules["min_contribution"] <= contribution <= rules["max_contribution"]):
                    errors.append(f"Invalid contribution {contribution} for {agent} in round {round_data.get('round')}")
            
            # Verify payoff calculation (if we have all data)
            if contributions and payoffs and len(contributions) == len(payoffs):
                total_contribution = sum(contributions.values())
                pot_share = (total_contribution * rules["multiplier"]) / len(contributions)
                
                for agent, payoff in payoffs.items():
                    contribution = contributions.get(agent, 0)
                    expected_payoff = rules["endowment"] - contribution + pot_share
                    
                    # Allow small floating point differences
                    if abs(payoff - expected_payoff) > 0.1:
                        errors.append(f"Payoff mismatch for {agent}: got {payoff}, expected {expected_payoff:.1f}")
        
        return errors
    
    def _validate_stag_hunt_record(self, record: Dict) -> List[str]:
        """Validate Stag Hunt Game data"""
        errors = []
        payoff_matrix = self.game_rules["stag_hunt"]["payoffs"]
        
        for round_data in record.get("rounds_data", []):
            choices = round_data.get("choices", {})
            payoffs = round_data.get("payoffs", {})
            
            # Check valid choices
            for agent, choice in choices.items():
                if choice.lower() not in ["stag", "hare"]:
                    errors.append(f"Invalid choice '{choice}' for {agent} in round {round_data.get('round')}")
            
            # Verify payoffs for 2-player games
            if len(choices) == 2 and len(payoffs) == 2:
                agents = list(choices.keys())
                if len(agents) == 2:
                    choice1 = choices[agents[0]].lower()
                    choice2 = choices[agents[1]].lower()
                    
                    if (choice1, choice2) in payoff_matrix:
                        expected_payoffs = payoff_matrix[(choice1, choice2)]
                        actual_p1 = payoffs.get(agents[0], 0)
                        actual_p2 = payoffs.get(agents[1], 0)
                        
                        if abs(actual_p1 - expected_payoffs[0]) > 0.1:
                            errors.append(f"Payoff mismatch for {agents[0]}: got {actual_p1}, expected {expected_payoffs[0]}")
                        if abs(actual_p2 - expected_payoffs[1]) > 0.1:
                            errors.append(f"Payoff mismatch for {agents[1]}: got {actual_p2}, expected {expected_payoffs[1]}")
        
        return errors
    
    def _validate_prisoners_dilemma_record(self, record: Dict) -> List[str]:
        """Validate Prisoner's Dilemma data"""
        errors = []
        
        # Basic PD payoff matrix (can be adjusted based on actual game parameters)
        pd_payoffs = {
            ("cooperate", "cooperate"): (3, 3),
            ("cooperate", "defect"): (0, 5),
            ("defect", "cooperate"): (5, 0),
            ("defect", "defect"): (1, 1)
        }
        
        for round_data in record.get("rounds_data", []):
            choices = round_data.get("choices", {})
            
            # Check valid choices
            for agent, choice in choices.items():
                if choice.lower() not in ["cooperate", "defect", "c", "d"]:
                    errors.append(f"Invalid PD choice '{choice}' for {agent} in round {round_data.get('round')}")
        
        return errors
    
    def save_consolidated_data(self, records: List[Dict], output_file: str = "analysis/unified_consolidated_data.jsonl") -> Path:
        """Save consolidated data to JSONL file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        
        logging.info(f"Saved {len(records)} records to {output_path}")
        return output_path
    
    def save_validation_report(self, errors: List[str], output_file: str = "analysis/validation_report.txt") -> Path:
        """Save validation errors to a report file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(f"Data Validation Report\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*50}\n\n")
            
            if not errors:
                f.write("No validation errors found! Data integrity check passed.\n")
            else:
                f.write(f"Found {len(errors)} validation errors:\n\n")
                for idx, error in enumerate(errors, 1):
                    f.write(f"{idx}. {error}\n")
        
        logging.info(f"Saved validation report to {output_path}")
        return output_path
    
    def get_summary_statistics(self, records: List[Dict]) -> Dict:
        """Generate summary statistics from consolidated data"""
        summary = {
            "total_records": len(records),
            "pilot_records": sum(1 for r in records if r.get("experiment_type") == "pilot"),
            "curriculum_records": sum(1 for r in records if r.get("experiment_type") == "curriculum"),
            "games": {},
            "curriculum_conditions": {},
            "records_with_rationales": 0,
            "total_rounds": 0,
            "validation_errors": len(self.validation_errors)
        }
        
        for record in records:
            # Count by game
            game = record.get("game_id", "unknown")
            summary["games"][game] = summary["games"].get(game, 0) + 1
            
            # Count by curriculum condition
            if record.get("experiment_type") == "curriculum":
                condition = record.get("curriculum_condition", "unknown")
                summary["curriculum_conditions"][condition] = summary["curriculum_conditions"].get(condition, 0) + 1
            
            # Check for rationales
            has_rationales = any(
                bool(round_data.get("rationales"))
                for round_data in record.get("rounds_data", [])
            )
            if has_rationales:
                summary["records_with_rationales"] += 1
            
            # Count total rounds
            summary["total_rounds"] += len(record.get("rounds_data", []))
        
        return summary


def main():
    """Main entry point for unified parser"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified parser for all experimental results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output", default="analysis/unified_consolidated_data.jsonl", 
                       help="Output JSONL file")
    parser.add_argument("--validation-report", default="analysis/validation_report.txt",
                       help="Validation report output file")
    parser.add_argument("--pilot-only", action="store_true", help="Parse only pilot data")
    parser.add_argument("--curriculum-only", action="store_true", help="Parse only curriculum data")
    parser.add_argument("--summary", action="store_true", help="Print summary statistics")
    
    args = parser.parse_args()
    
    # Determine what to parse
    include_pilot = not args.curriculum_only
    include_curriculum = not args.pilot_only
    
    # Initialize parser
    unified_parser = UnifiedResultsParser(args.results_dir)
    
    # Parse all data
    logging.info("Starting unified data parsing...")
    records, validation_errors = unified_parser.parse_all_data(
        include_pilot=include_pilot,
        include_curriculum=include_curriculum
    )
    
    # Save consolidated data
    unified_parser.save_consolidated_data(records, args.output)
    
    # Save validation report
    unified_parser.save_validation_report(validation_errors, args.validation_report)
    
    # Print summary if requested
    if args.summary:
        summary = unified_parser.get_summary_statistics(records)
        print("\n" + "="*60)
        print("UNIFIED DATA PARSING SUMMARY")
        print("="*60)
        print(f"Total records parsed: {summary['total_records']}")
        print(f"  - Pilot study: {summary['pilot_records']}")
        print(f"  - Curriculum experiments: {summary['curriculum_records']}")
        print(f"\nRecords with rationales: {summary['records_with_rationales']} ({summary['records_with_rationales']/summary['total_records']*100:.1f}%)")
        print(f"Total rounds of data: {summary['total_rounds']}")
        print(f"Validation errors found: {summary['validation_errors']}")
        
        print("\nGames parsed:")
        for game, count in summary['games'].items():
            print(f"  - {game}: {count}")
        
        if summary['curriculum_conditions']:
            print("\nCurriculum conditions:")
            for condition, count in summary['curriculum_conditions'].items():
                print(f"  - {condition}: {count}")
        
        print("\n" + "="*60)
        
        if validation_errors:
            print(f"\n⚠️  Found {len(validation_errors)} validation errors.")
            print(f"See {args.validation_report} for details.")
        else:
            print("\n✅ All data validation checks passed!")


if __name__ == "__main__":
    main()