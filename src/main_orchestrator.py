#!/usr/bin/env python3
"""
Main orchestration script for LLM Game Theory experiments
Manages game configurations, experimental settings, and execution flow
"""

import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Ensure API key is set
if not os.environ.get("DEEPINFRA_API_KEY"):
    os.environ["DEEPINFRA_API_KEY"] = "iy1F0S1F3hO8igg1mZJI4XpjqsmYRFg5"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import engine and games
from engine import GameEngine
from games import (
    StagHuntGame, PublicGoodsGame, StagHuntWithCommunication,
    MinimumEffortGame, MinimumEffortGameWithCommunication,
    VolunteersDilemmaGame, VolunteersDilemmaWithCommunication,
    BattleOfTheSexesGame, BattleOfTheSexesWithCommunication,
    IteratedPublicGoodsGame, IteratedPublicGoodsGameWithCommunication
)

# Experimental settings from CLAUDE.md
EXPERIMENTAL_SETTINGS = {
    "setting_1_heterogeneous_battle": {
        "name": "Heterogeneous Battle",
        "description": "Strategic interactions in a fully heterogeneous group of top-tier instruction-tuned models",
        "agents": [
            {
                "name": "Agent_1",
                "model": "mistralai/Mixtral-8x22B-Instruct-v0.1",
                "model_family": "Mixtral"
            },
            {
                "name": "Agent_2", 
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "model_family": "Qwen"
            },
            {
                "name": "Agent_3",
                "model": "meta-llama/Llama-3.3-70B-Instruct",
                "model_family": "Llama"
            },
            {
                "name": "Agent_4",
                "model": "deepseek-ai/DeepSeek-V3",
                "model_family": "DeepSeek"
            }
        ]
    },
    "setting_2_model_family_coalition": {
        "name": "Model Family Coalition Test",
        "description": "Testing for in-group bias and collusion between agents of the same model family",
        "agents": [
            {
                "name": "Agent_1",
                "model": "meta-llama/Llama-3.3-70B-Instruct",
                "model_family": "Llama",
                "group": "Team_Llama"
            },
            {
                "name": "Agent_2",
                "model": "meta-llama/Llama-3.3-70B-Instruct",
                "model_family": "Llama",
                "group": "Team_Llama"
            },
            {
                "name": "Agent_3",
                "model": "mistralai/Mixtral-8x22B-Instruct-v0.1",
                "model_family": "Mixtral",
                "group": "Team_Mixtral"
            },
            {
                "name": "Agent_4",
                "model": "mistralai/Mixtral-8x22B-Instruct-v0.1",
                "model_family": "Mixtral",
                "group": "Team_Mixtral"
            }
        ]
    }
}

# Game configurations - PHASE 2 EXPANDED
GAMES = {
    # Existing games
    "stag_hunt": {
        "class": StagHuntGame,
        "name": "N-Player Stag Hunt",
        "rounds": 3,
        "description": "Test of coordination and trust"
    },
    "public_goods": {
        "class": PublicGoodsGame,
        "name": "N-Player Public Goods Game with Punishment",
        "rounds": 3,
        "description": "Test of cooperation vs. free-riding with punishment mechanism"
    },
    "stag_hunt_communication": {
        "class": StagHuntWithCommunication,
        "name": "N-Player Stag Hunt with One-Word Communication",
        "rounds": 3,
        "description": "Test for emergent communication and collusion"
    },

    # NEW GAMES - Phase 2
    "minimum_effort": {
        "class": MinimumEffortGame,
        "name": "Minimum Effort Game",
        "rounds": 5,
        "description": "Pure coordination / weakest-link game"
    },
    "minimum_effort_comm": {
        "class": MinimumEffortGameWithCommunication,
        "name": "Minimum Effort Game with Communication",
        "rounds": 5,
        "description": "Weakest-link coordination with cheap-talk"
    },
    "volunteers_dilemma": {
        "class": VolunteersDilemmaGame,
        "name": "Volunteer's Dilemma",
        "rounds": 5,
        "description": "Asymmetric coordination / diffusion of responsibility"
    },
    "volunteers_dilemma_comm": {
        "class": VolunteersDilemmaWithCommunication,
        "name": "Volunteer's Dilemma with Communication",
        "rounds": 5,
        "description": "Volunteering coordination with cheap-talk"
    },
    "battle_of_sexes": {
        "class": BattleOfTheSexesGame,
        "name": "Battle of the Sexes",
        "rounds": 5,
        "description": "Multiple equilibria coordination with conflicting preferences"
    },
    "battle_of_sexes_comm": {
        "class": BattleOfTheSexesWithCommunication,
        "name": "Battle of the Sexes with Communication",
        "rounds": 5,
        "description": "Conflicting preferences coordination with cheap-talk"
    },
    "ipgg_communication": {
        "class": IteratedPublicGoodsGameWithCommunication,
        "name": "IPGG+P with Communication (CRITICAL CONTROL)",
        "rounds": 10,
        "description": "Public goods with punishment AND cheap-talk - tests if communication solves social dilemma"
    }
}

class ExperimentOrchestrator:
    """Orchestrates the execution of game theory experiments"""
    
    def __init__(self, trials: int = 30, games_to_run: Optional[List[str]] = None):
        """
        Initialize the orchestrator
        
        Args:
            trials: Number of trials to run per game/setting combination
            games_to_run: List of game IDs to run (None means all)
        """
        self.trials = trials
        self.games_to_run = games_to_run or list(GAMES.keys())
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Validate API key
        if not os.environ.get("DEEPINFRA_API_KEY"):
            raise ValueError("DEEPINFRA_API_KEY environment variable not set")
        
        logger.info(f"Initialized orchestrator for {trials} trials")
        logger.info(f"Games to run: {self.games_to_run}")
    
    def run_single_experiment(
        self, 
        game_id: str, 
        setting_id: str, 
        trial_num: int
    ) -> Dict[str, Any]:
        """
        Run a single experiment (one trial of one game with one setting)
        
        Args:
            game_id: ID of the game to run
            setting_id: ID of the experimental setting
            trial_num: Trial number
            
        Returns:
            Dictionary containing experiment results
        """
        game_config = GAMES[game_id]
        setting = EXPERIMENTAL_SETTINGS[setting_id]
        
        logger.info(f"Running {game_config['name']} - {setting['name']} - Trial {trial_num}")
        
        # Create game instance
        game_class = game_config["class"]
        game = game_class(
            agents=setting["agents"],
            rounds=game_config["rounds"]
        )
        
        # Create engine and run game
        engine = GameEngine(game=game)
        
        start_time = time.time()
        results = engine.run()
        duration = time.time() - start_time
        
        # Add metadata
        results["metadata"] = {
            "game_id": game_id,
            "game_name": game_config["name"],
            "setting_id": setting_id,
            "setting_name": setting["name"],
            "trial_num": trial_num,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        
        return results
    
    def save_results(
        self, 
        results: Dict[str, Any], 
        game_id: str, 
        setting_id: str, 
        trial_num: int
    ):
        """Save experiment results to file"""
        # Create directory structure
        game_dir = self.results_dir / game_id / setting_id
        game_dir.mkdir(parents=True, exist_ok=True)
        
        trial_dir = game_dir / f"trial_{trial_num:02d}"
        trial_dir.mkdir(exist_ok=True)
        
        # Save experiment data
        with open(trial_dir / "experiment_data.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {trial_dir}")
    
    def run_all_experiments(self):
        """Run all configured experiments"""
        logger.info("="*60)
        logger.info("STARTING FULL EXPERIMENT RUN")
        logger.info(f"Trials per configuration: {self.trials}")

        # Check if we should use only one setting (for quick tests)
        single_setting_mode = os.environ.get("GAMETH_SINGLE_SETTING") == "1"
        if single_setting_mode:
            settings_to_use = {"setting_1_heterogeneous_battle": EXPERIMENTAL_SETTINGS["setting_1_heterogeneous_battle"]}
            logger.info("SINGLE SETTING MODE: Using only setting_1_heterogeneous_battle")
        else:
            settings_to_use = EXPERIMENTAL_SETTINGS

        # Calculate total experiments
        total_experiments = len(self.games_to_run) * len(settings_to_use) * self.trials
        logger.info(f"Total experiments: {total_experiments}")
        logger.info("="*60)

        overall_results = {
            "start_time": datetime.now().isoformat(),
            "trials": self.trials,
            "games": {},
            "summary_statistics": {}
        }

        # Create main progress bar
        start_time = time.time()
        pbar = tqdm(
            total=total_experiments,
            desc="Overall Progress",
            unit="trial",
            position=0,
            leave=True
        )

        for game_id in self.games_to_run:
            overall_results["games"][game_id] = {}
            game_name = GAMES[game_id]["name"]

            for setting_id in settings_to_use.keys():
                setting_name = settings_to_use[setting_id]["name"]
                overall_results["games"][game_id][setting_id] = {
                    "trials": [],
                    "summary": {}
                }

                # Update description for current game/setting
                pbar.set_description(f"{game_name} - {setting_name}")

                # Run trials
                for trial_num in range(1, self.trials + 1):
                    try:
                        # Run experiment
                        results = self.run_single_experiment(
                            game_id=game_id,
                            setting_id=setting_id,
                            trial_num=trial_num
                        )

                        # Save results
                        self.save_results(results, game_id, setting_id, trial_num)

                        # Add to overall results
                        overall_results["games"][game_id][setting_id]["trials"].append({
                            "trial_num": trial_num,
                            "success": True,
                            "total_payoffs": results.get("total_payoffs", {}),
                            "cooperation_rate": results.get("cooperation_rate", 0)
                        })

                        # Update progress bar
                        pbar.update(1)

                        # Calculate and display ETA
                        elapsed = time.time() - start_time
                        completed = pbar.n
                        if completed > 0:
                            avg_time_per_trial = elapsed / completed
                            remaining = total_experiments - completed
                            eta_seconds = remaining * avg_time_per_trial
                            eta_minutes = eta_seconds / 60
                            pbar.set_postfix({
                                'ETA': f'{eta_minutes:.1f}min',
                                'Avg': f'{avg_time_per_trial:.1f}s/trial'
                            })

                        # Rate limiting
                        time.sleep(2)

                    except Exception as e:
                        logger.error(f"Error in trial {trial_num}: {e}")
                        overall_results["games"][game_id][setting_id]["trials"].append({
                            "trial_num": trial_num,
                            "success": False,
                            "error": str(e)
                        })
                        pbar.update(1)

                # Calculate summary statistics for this game/setting
                trials_data = overall_results["games"][game_id][setting_id]["trials"]
                successful_trials = [t for t in trials_data if t.get("success", False)]

                if successful_trials:
                    avg_cooperation = sum(t.get("cooperation_rate", 0) for t in successful_trials) / len(successful_trials)
                    overall_results["games"][game_id][setting_id]["summary"] = {
                        "successful_trials": len(successful_trials),
                        "failed_trials": len(trials_data) - len(successful_trials),
                        "average_cooperation_rate": avg_cooperation
                    }

        pbar.close()

        # Save overall results
        overall_results["end_time"] = datetime.now().isoformat()
        total_duration = time.time() - start_time
        overall_results["total_duration_seconds"] = total_duration

        with open(self.results_dir / "overall_summary.json", "w") as f:
            json.dump(overall_results, f, indent=2)

        logger.info("="*60)
        logger.info("EXPERIMENT RUN COMPLETE")
        logger.info(f"Total duration: {total_duration/60:.1f} minutes")
        logger.info(f"Results saved to {self.results_dir}")
        logger.info("="*60)

        return overall_results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run LLM game theory experiments")
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials per game/setting combination (default: 10)"
    )
    parser.add_argument(
        "--games",
        nargs="+",
        choices=list(GAMES.keys()),
        help="Specific games to run (default: all)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with 2 trials only"
    )
    parser.add_argument(
        "--experiment",
        choices=["standard", "curriculum"],
        default="standard",
        help="Experiment type: standard or curriculum learning"
    )
    parser.add_argument(
        "--curriculum-config",
        type=str,
        help="Path to curriculum configuration JSON file"
    )
    parser.add_argument(
        "--curriculum-only",
        action="store_true",
        help="Run only curriculum experiments (cooperation-first and punishment-focused)"
    )
    parser.add_argument(
        "--communication-only",
        action="store_true",
        help="Run only communication experiments (all new games with/without communication)"
    )
    parser.add_argument(
        "--condition",
        type=str,
        help="Run specific experimental condition (e.g., 'minimum_effort_comm')"
    )
    
    args = parser.parse_args()

    # Override trials for quick test
    if args.quick_test:
        args.trials = 2
        logger.info("Quick test mode: Running 2 trials only")

    # Handle special experiment modes
    if args.curriculum_only:
        # Run both new curricula
        from engine import CurriculumEngine

        curricula = [
            "config/curriculum_cooperation_first.json",
            "config/curriculum_punishment_focused.json"
        ]

        for curriculum_path in curricula:
            with open(curriculum_path, 'r') as f:
                curriculum_config = json.load(f)

            logger.info(f"Running curriculum: {curriculum_config['name']}")

            curriculum_engine = CurriculumEngine(
                curriculum_config=curriculum_config,
                trials=args.trials
            )

            results = curriculum_engine.run_all_trials()

        return

    if args.communication_only:
        # Run only communication experiments
        communication_games = [
            "minimum_effort", "minimum_effort_comm",
            "volunteers_dilemma", "volunteers_dilemma_comm",
            "battle_of_sexes", "battle_of_sexes_comm",
            "ipgg_communication"
        ]
        args.games = communication_games
        logger.info(f"Communication-only mode: Running {len(communication_games)} game variants")

    if args.condition:
        # Run single condition
        if args.condition not in GAMES:
            logger.error(f"Unknown condition: {args.condition}")
            sys.exit(1)
        args.games = [args.condition]
        logger.info(f"Running single condition: {args.condition}")

    if args.experiment == "curriculum":
        # Run curriculum learning experiment
        if not args.curriculum_config:
            logger.error("Curriculum config file required for curriculum experiments")
            sys.exit(1)
        
        from engine import CurriculumEngine
        
        # Load curriculum config
        with open(args.curriculum_config, 'r') as f:
            curriculum_config = json.load(f)
        
        logger.info(f"Running curriculum experiment: {curriculum_config['name']}")
        
        # Create curriculum engine
        curriculum_engine = CurriculumEngine(
            curriculum_config=curriculum_config,
            trials=args.trials
        )
        
        # Run curriculum
        results = curriculum_engine.run_all_trials()
        
    else:
        # Standard experiment
        # Create orchestrator
        orchestrator = ExperimentOrchestrator(
            trials=args.trials,
            games_to_run=args.games
        )
        
        # Run experiments
        results = orchestrator.run_all_experiments()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if args.experiment == "curriculum":
        # Print curriculum summary
        if "summary" in results:
            summary = results["summary"]
            print(f"\nCurriculum: {results.get('name', 'Unknown')}")
            print(f"  Successful trials: {summary.get('successful_trials', 0)}/{args.trials}")
            print(f"  Final cooperation rate: {summary.get('avg_final_cooperation_rate', 0):.2%}")
            print(f"  Final average payoff: {summary.get('avg_final_payoff', 0):.1f}")
    else:
        # Standard experiment summary
        for game_id, game_data in results["games"].items():
            print(f"\n{GAMES[game_id]['name']}:")
            for setting_id, setting_data in game_data.items():
                summary = setting_data.get("summary", {})
                print(f"  {EXPERIMENTAL_SETTINGS[setting_id]['name']}:")
                print(f"    Successful trials: {summary.get('successful_trials', 0)}/{args.trials}")
                if summary.get('average_cooperation_rate') is not None:
                    print(f"    Average cooperation: {summary['average_cooperation_rate']:.2%}")

if __name__ == "__main__":
    main()