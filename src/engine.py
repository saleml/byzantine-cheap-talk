#!/usr/bin/env python3
"""
Game Engine for running LLM game theory experiments
Handles API calls, history tracking, and payoff calculation
"""

import os
import json
import time
import logging
import re
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
import re

# Import Claude lesson generator (REQUIRED)
from engine_claude_lessons import ClaudeLessonGenerator

logger = logging.getLogger(__name__)

# Configure API client (DeepInfra by default, override with OPENAI_BASE_URL)
_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepinfra.com/v1/openai")
# Choose the right key for the chosen endpoint. Previously we always preferred
# OPENAI_API_KEY which sent an OpenAI key to the DeepInfra endpoint and caused
# 401/invalid model errors. Now we pick the key that matches the base URL.
if "deepinfra" in _base_url:
    _api_key = os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("OPENAI_API_KEY")
else:
    _api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPINFRA_API_KEY")

client = OpenAI(api_key=_api_key, base_url=_base_url)

# Optional formatter client (for post-processing malformed outputs)
_fmt_api_key = os.environ.get("OPENAI_API_KEY")
_fmt_base = os.environ.get("OPENAI_BASE_URL_FORMATTER", "https://api.openai.com/v1")
formatter_client = OpenAI(api_key=_fmt_api_key, base_url=_fmt_base) if _fmt_api_key else None

class GameEngine:
    """
    Core engine for running game theory experiments with LLMs
    """
    
    def __init__(self, game: 'Game', rate_limit_delay: float = 1.0):
        """
        Initialize the game engine
        
        Args:
            game: Game instance to run
            rate_limit_delay: Delay between API calls in seconds
        """
        self.game = game
        self.rate_limit_delay = rate_limit_delay
        self.history = []
        
        logger.info(f"Initialized GameEngine for {game.__class__.__name__}")
    
    def call_agent(
        self, 
        agent_name: str, 
        model: str, 
        prompt: str,
        agent_config: Dict[str, Any] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Call a single agent for their decision
        
        Args:
            agent_name: Name of the agent
            model: Model ID to use
            prompt: Prompt to send to the model
            agent_config: Agent configuration (may contain system_prompt_suffix)
            max_retries: Maximum number of retries on failure
            
        Returns:
            Parsed response from the agent
        """
        # Build system prompt with optional suffix for curriculum learning
        system_prompt = (
            "You are a rational game-playing agent. Return ONLY a single valid JSON object matching the"
            " requested schema; no extra text, code fences, markdown, or explanations."
        )
        if agent_config and "system_prompt_suffix" in agent_config:
            system_prompt += agent_config["system_prompt_suffix"]
        
        # Debug logging to verify curriculum injection
        if agent_config and "system_prompt_suffix" in agent_config:
            logger.debug(f"Curriculum context injected for {agent_name} (first 500 chars): {agent_config['system_prompt_suffix'][:500]}")
        
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "timeout": 60,
                    "response_format": {"type": "json_object"},
                }
                # New OpenAI models (o3, gpt-4.1 family) require max_completion_tokens and do not accept temperature
                if "openai.com" in _base_url and (model.startswith("o3") or model.startswith("gpt-4.1")):
                    kwargs["max_completion_tokens"] = 300
                else:
                    kwargs["max_tokens"] = 300
                    kwargs["temperature"] = 0.0

                response = client.chat.completions.create(**kwargs)

                content = response.choices[0].message.content or ""

                def extract_json(text: str):
                    # Strip code fences if present
                    if text.strip().startswith("```"):
                        parts = text.split('```')
                        if len(parts) >= 3:
                            text = parts[1] if parts[1].strip().startswith('json') else parts[1]
                        else:
                            text = parts[0]
                    start = text.find('{')
                    end = text.rfind('}') + 1
                    if start >= 0 and end > start:
                        try:
                            return json.loads(text[start:end])
                        except Exception as e:
                            logger.warning(f"JSON load failed: {e}; raw snippet: {text[start:end][:200]}")
                    return None

                parsed = extract_json(content)
                if parsed is not None:
                    logger.debug(f"{agent_name} response: {parsed}")
                    return parsed
                else:
                    logger.error(f"No JSON found for {agent_name}. Raw content: {content}")
                    # Heuristic fallback parsing
                    heuristic = self._heuristic_parse_response(content, prompt)
                    if heuristic is not None:
                        logger.warning(f"Using heuristic parse for {agent_name}: {heuristic}")
                        return heuristic
                    # Formatter LLM fallback
                    formatted = self._llm_format_response(content, prompt)
                    if formatted is not None:
                        logger.warning(f"Using formatter parse for {agent_name}: {formatted}")
                        return formatted
                    raise ValueError("No JSON found in response")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error for {agent_name} (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    return self.game.get_default_response(agent_name)
                    
            except Exception as e:
                logger.error(f"API error for {agent_name} (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    return self.game.get_default_response(agent_name)
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return self.game.get_default_response(agent_name)

    def _heuristic_parse_response(self, text: str, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Best-effort extraction for non-JSON responses from some endpoints.
        Handles our game action schemas heuristically.
        """
        lower = text.lower()
        # Communication phase
        if "communicate" in prompt.lower() or "communication phase" in prompt.lower():
            word = re.findall(r"[a-zA-Z]+", text)
            word = word[0] if word else "cooperate"
            return {"reasoning": text[:200], "action": {"type": "communicate", "word": word}}
        # Contribution stage
        if "contribute" in prompt.lower() or "public goods" in prompt.lower():
            nums = re.findall(r"\d+", text)
            amount = int(nums[0]) if nums else 10
            amount = max(0, min(20, amount))
            return {"reasoning": text[:200], "action": {"type": "contribute", "amount": amount}}
        # Punishment stage
        if "punish" in prompt.lower():
            return {"reasoning": text[:200], "action": {"type": "punish", "targets": []}}
        # Stag Hunt / action choice
        if "hunt stag" in prompt.lower() or "hunt hare" in prompt.lower():
            choice = "Hunt Stag" if "stag" in lower else "Hunt Hare"
            return {"reasoning": text[:200], "action": {"choice": choice}}
        # Default fallback
        return None

    def _llm_format_response(self, text: str, prompt: str) -> Optional[Dict[str, Any]]:
        """Use a secondary LLM to coerce malformed output into JSON."""
        if formatter_client is None:
            return None
        try:
            sys_instr = (
                "You are a strict JSON reformatter. Given a MODEL OUTPUT and the ORIGINAL PROMPT, "
                "return ONLY a single valid JSON object that best fits the requested schema. "
                "If unsure about numbers, choose a reasonable integer within allowed range."
            )
            msg = f"ORIGINAL PROMPT:\n{prompt}\n\nMODEL OUTPUT:\n{text}\n\nReturn only JSON."
            resp = formatter_client.chat.completions.create(
                model=os.environ.get("FORMATTER_MODEL", "gpt-4o-mini"),
                messages=[{"role": "system", "content": sys_instr}, {"role": "user", "content": msg}],
                temperature=0,
                max_tokens=200
            )
            content = resp.choices[0].message.content or ""
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except Exception as e:
            logger.error(f"Formatter fallback failed: {e}")
        return None
    
    def run_round(self, round_num: int) -> Dict[str, Any]:
        """
        Run a single round of the game
        
        Args:
            round_num: Current round number
            
        Returns:
            Results of the round
        """
        logger.info(f"Running round {round_num}")
        
        # Get decisions from all agents
        decisions = {}
        
        for agent_config in self.game.agents:
            agent_name = agent_config["name"]
            model = agent_config["model"]
            
            # Get prompt for this agent
            prompt = self.game.get_agent_prompt(
                agent_name=agent_name,
                agent_config=agent_config,
                round_num=round_num,
                history=self.history
            )
            
            # Get agent's decision
            logger.debug(f"Calling {agent_name} with model {model}")
            decision = self.call_agent(agent_name, model, prompt, agent_config)
            if not isinstance(decision, dict):
                decision = self.game.get_default_response(agent_name)
            decisions[agent_name] = decision
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
        
        # Process round results
        round_results = self.game.process_round(decisions, round_num)
        
        # IMPORTANT: Store full decisions (including reasoning) in results
        round_results["full_decisions"] = decisions
        
        # Add to history
        self.history.append(round_results)
        
        logger.info(f"Round {round_num} complete: {round_results.get('summary', 'No summary')}")
        
        return round_results
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete game
        
        Returns:
            Complete game results including all rounds and final statistics
        """
        logger.info(f"Starting game: {self.game.__class__.__name__}")
        logger.info(f"Agents: {[a['name'] for a in self.game.agents]}")
        logger.info(f"Rounds: {self.game.rounds}")
        
        start_time = time.time()
        
        # Initialize game
        self.game.initialize()
        
        # Check if this is a two-stage game
        is_two_stage = hasattr(self.game, 'stage')
        
        # Run all rounds
        for round_num in range(1, self.game.rounds + 1):
            if is_two_stage:
                # For two-stage games, run both stages
                stage1_results = self.run_round(round_num)
                stage2_results = self.run_round(round_num)
                # Only the second stage results go into history
                # (it contains the complete round data)
            else:
                round_results = self.run_round(round_num)
        
        # Calculate final results
        final_results = self.game.calculate_final_results(self.history)
        
        # Add timing
        duration = time.time() - start_time
        final_results["duration"] = duration
        final_results["rounds_data"] = self.history
        
        logger.info(f"Game complete in {duration:.2f} seconds")
        logger.info(f"Final results: {final_results.get('summary', 'No summary')}")
        
        return final_results


class Game(ABC):
    """
    Abstract base class for all games
    """
    
    def __init__(self, agents: List[Dict[str, Any]], rounds: int = 3):
        """
        Initialize the game
        
        Args:
            agents: List of agent configurations
            rounds: Number of rounds to play
        """
        self.agents = agents
        self.rounds = rounds
        self.total_payoffs = {agent["name"]: 0 for agent in agents}
    
    def initialize(self):
        """Initialize the game (can be overridden by subclasses)"""
        pass
    
    @abstractmethod
    def get_agent_prompt(
        self, 
        agent_name: str, 
        agent_config: Dict[str, Any],
        round_num: int, 
        history: List[Dict[str, Any]]
    ) -> str:
        """
        Get the prompt for an agent
        
        Args:
            agent_name: Name of the agent
            agent_config: Agent configuration
            round_num: Current round number
            history: Game history
            
        Returns:
            Prompt string for the agent
        """
        pass
    
    @abstractmethod
    def process_round(
        self, 
        decisions: Dict[str, Dict[str, Any]], 
        round_num: int
    ) -> Dict[str, Any]:
        """
        Process the results of a round
        
        Args:
            decisions: Decisions from all agents
            round_num: Current round number
            
        Returns:
            Round results including payoffs
        """
        pass
    
    @abstractmethod
    def get_default_response(self, agent_name: str) -> Dict[str, Any]:
        """
        Get a default response for an agent (used on API failure)
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Default response dictionary
        """
        pass
    
    def calculate_final_results(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate final results from game history
        
        Args:
            history: Complete game history
            
        Returns:
            Final results including statistics
        """
        # Calculate total payoffs
        total_payoffs = {agent["name"]: 0 for agent in self.agents}
        cooperation_rounds = 0
        
        for round_data in history:
            payoffs = round_data.get("payoffs", {})
            for agent_name, payoff in payoffs.items():
                total_payoffs[agent_name] += payoff
            
            if round_data.get("all_cooperated", False):
                cooperation_rounds += 1
        
        # Calculate statistics
        avg_payoff = sum(total_payoffs.values()) / len(total_payoffs) if total_payoffs else 0
        cooperation_rate = cooperation_rounds / len(history) if history else 0
        
        # Model family analysis (if applicable)
        model_family_payoffs = {}
        for agent in self.agents:
            family = agent.get("model_family", "Unknown")
            if family not in model_family_payoffs:
                model_family_payoffs[family] = []
            model_family_payoffs[family].append(total_payoffs[agent["name"]])
        
        model_family_avg = {
            family: sum(payoffs) / len(payoffs) if payoffs else 0
            for family, payoffs in model_family_payoffs.items()
        }
        
        return {
            "total_payoffs": total_payoffs,
            "average_payoff": avg_payoff,
            "cooperation_rate": cooperation_rate,
            "cooperation_rounds": cooperation_rounds,
            "total_rounds": len(history),
            "model_family_averages": model_family_avg,
            "summary": f"Cooperation rate: {cooperation_rate:.2%}, Avg payoff: {avg_payoff:.1f}"
        }


class CurriculumEngine:
    """
    Engine for running curriculum learning experiments
    """
    
    def __init__(self, curriculum_config: Dict[str, Any], trials: int = 30):
        """
        Initialize the curriculum engine
        
        Args:
            curriculum_config: Configuration for the curriculum
            trials: Number of trials to run
        """
        self.config = curriculum_config
        self.trials = trials
        self.name = curriculum_config.get("name", "curriculum")
        self.stages = curriculum_config.get("stages", [])
        self.agents_config = curriculum_config.get("agents", [])
        
        # Initialize Claude lesson generator (REQUIRED)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for curriculum learning")
        
        self.claude_generator = ClaudeLessonGenerator()
        logger.info("Claude Opus 4.1 lesson generator initialized")
        
        # Results storage
        self.results_dir = Path("results") / "curriculum" / self.name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized CurriculumEngine: {self.name}")
        logger.info(f"Stages: {len(self.stages)}")
    
    def generate_lesson_summary(self, stage_results: Dict[str, Any], stage_config: Dict[str, Any], 
                               stage_num: int = 1, previous_lessons: List[str] = None) -> str:
        """
        Generate a natural language summary of lessons learned from a stage
        
        Args:
            stage_results: Results from the completed stage
            stage_config: Configuration of the completed stage
            stage_num: Current stage number
            previous_lessons: Lessons from previous stages
            
        Returns:
            Lesson summary string
        """
        # Use Claude (REQUIRED - no fallback)
        return self.claude_generator.generate_lesson(
            stage_results, stage_config, stage_num, previous_lessons
        )
    
    def inject_curriculum_context(self, agents: List[Dict[str, Any]], stage_num: int, 
                                 lessons: List[str]) -> List[Dict[str, Any]]:
        """
        Inject curriculum context and lessons into agent configurations
        
        Args:
            agents: Base agent configurations
            stage_num: Current stage number
            lessons: List of lesson summaries from previous stages
            
        Returns:
            Modified agent configurations with curriculum context
        """
        modified_agents = []
        
        for agent in agents:
            modified_agent = agent.copy()
            
            # Create curriculum-aware system prompt
            curriculum_context = "\n\n### CURRICULUM CONTEXT\n"
            curriculum_context += f"You are in Stage {stage_num} of a learning curriculum.\n"
            
            if lessons:
                curriculum_context += "\nLessons from previous stages:\n"
                for i, lesson in enumerate(lessons, 1):
                    curriculum_context += f"{i}. {lesson}\n"
                curriculum_context += "\nApply these lessons to improve your performance in this stage."
            else:
                curriculum_context += "This is your first stage. Learn from this experience."
            
            # Add context to agent config
            if "system_prompt_suffix" not in modified_agent:
                modified_agent["system_prompt_suffix"] = ""
            modified_agent["system_prompt_suffix"] += curriculum_context
            
            modified_agents.append(modified_agent)
        
        return modified_agents
    
    def run_stage(self, stage_config: Dict[str, Any], stage_num: int, 
                  lessons: List[str], trial_num: int) -> Dict[str, Any]:
        """
        Run a single curriculum stage
        
        Args:
            stage_config: Configuration for this stage
            stage_num: Stage number (1-indexed)
            lessons: Lessons from previous stages
            trial_num: Current trial number
            
        Returns:
            Stage results
        """
        game_class_name = stage_config.get("game")
        game_params = stage_config.get("params", {})
        
        logger.info(f"Running Stage {stage_num}: {game_class_name}")
        
        # Import the game class dynamically
        if game_class_name == "IteratedPrisonersDilemma":
            from games import IteratedPrisonersDilemma
            game_class = IteratedPrisonersDilemma
        elif game_class_name == "NPlayerIteratedPrisonersDilemma":
            from games import NPlayerIteratedPrisonersDilemma
            game_class = NPlayerIteratedPrisonersDilemma
        elif game_class_name == "IteratedPublicGoodsGame":
            from games import IteratedPublicGoodsGame
            game_class = IteratedPublicGoodsGame
        elif game_class_name == "MinimumEffortGame":
            from games import MinimumEffortGame
            game_class = MinimumEffortGame
        elif game_class_name == "BattleOfTheSexesGame":
            from games import BattleOfTheSexesGame
            game_class = BattleOfTheSexesGame
        elif game_class_name == "StagHuntWithCommunication":
            from games import StagHuntWithCommunication
            game_class = StagHuntWithCommunication
        elif game_class_name == "VolunteersDilemmaGame":
            from games import VolunteersDilemmaGame
            game_class = VolunteersDilemmaGame
        elif game_class_name == "IteratedPublicGoodsGameWithCommunication":
            from games import IteratedPublicGoodsGameWithCommunication
            game_class = IteratedPublicGoodsGameWithCommunication
        else:
            raise ValueError(f"Unknown game class: {game_class_name}")
        
        # Prepare agents with curriculum context
        agents_with_context = self.inject_curriculum_context(
            self.agents_config, stage_num, lessons
        )

        # Adjust agents for variable player counts
        num_players = stage_config.get("num_players", None)
        if num_players is not None:
            # Use specified number of players
            agents_with_context = agents_with_context[:num_players]
        elif game_class_name == "IteratedPrisonersDilemma":
            # 2-player IPD
            agents_with_context = agents_with_context[:2]

        # Create game instance
        game = game_class(agents=agents_with_context, **game_params)
        
        # Create engine and run game
        engine = GameEngine(game)
        results = engine.run()
        
        # Add stage metadata
        results["stage"] = stage_num
        results["stage_name"] = stage_config.get("name", f"Stage {stage_num}")
        results["game"] = game_class_name
        results["trial"] = trial_num
        
        return results
    
    def run_single_trial(self, trial_num: int) -> Dict[str, Any]:
        """
        Run a complete curriculum (all stages) for a single trial
        
        Args:
            trial_num: Trial number
            
        Returns:
            Complete trial results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Curriculum Trial {trial_num}")
        logger.info(f"{'='*60}")
        
        trial_results = {
            "trial": trial_num,
            "curriculum": self.name,
            "stages": [],
            "lessons_learned": [],
            "timestamp": datetime.now().isoformat()
        }
        
        lessons = []
        
        # Run each stage sequentially
        for stage_num, stage_config in enumerate(self.stages, 1):
            try:
                # Run the stage
                stage_results = self.run_stage(
                    stage_config, stage_num, lessons, trial_num
                )
                
                # Generate lesson from this stage
                lesson = self.generate_lesson_summary(stage_results, stage_config, stage_num, lessons)
                lessons.append(lesson)
                
                # Store results
                trial_results["stages"].append(stage_results)
                trial_results["lessons_learned"].append(lesson)
                
                logger.info(f"Stage {stage_num} complete: {lesson}")
                
                # Brief pause between stages
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in stage {stage_num}: {e}")
                trial_results["stages"].append({
                    "stage": stage_num,
                    "error": str(e),
                    "success": False
                })
        
        # Calculate overall metrics
        if trial_results["stages"]:
            final_stage = trial_results["stages"][-1]
            trial_results["final_cooperation_rate"] = final_stage.get("cooperation_rate", 0)
            trial_results["final_avg_payoff"] = final_stage.get("average_payoff", 0)
        
        return trial_results
    
    def run_all_trials(self) -> Dict[str, Any]:
        """
        Run all trials of the curriculum experiment

        Returns:
            Complete experiment results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING CURRICULUM EXPERIMENT: {self.name}")
        logger.info(f"Trials: {self.trials}")
        logger.info(f"Stages: {len(self.stages)}")
        logger.info(f"{'='*60}")

        all_results = {
            "experiment": "curriculum",
            "name": self.name,
            "config": self.config,
            "trials": [],
            "summary": {},
            "timestamp": datetime.now().isoformat()
        }

        # Create progress bar
        start_time = time.time()
        pbar = tqdm(
            total=self.trials,
            desc=f"Curriculum: {self.name}",
            unit="trial",
            position=0,
            leave=True
        )

        # Run each trial
        for trial_num in range(1, self.trials + 1):
            try:
                trial_results = self.run_single_trial(trial_num)
                all_results["trials"].append(trial_results)

                # Save trial results
                trial_dir = self.results_dir / f"trial_{trial_num:02d}"
                trial_dir.mkdir(exist_ok=True)

                with open(trial_dir / "results.json", "w") as f:
                    json.dump(trial_results, f, indent=2)

                # Update progress bar with ETA
                pbar.update(1)
                elapsed = time.time() - start_time
                completed = pbar.n
                if completed > 0:
                    avg_time_per_trial = elapsed / completed
                    remaining = self.trials - completed
                    eta_seconds = remaining * avg_time_per_trial
                    eta_minutes = eta_seconds / 60
                    pbar.set_postfix({
                        'ETA': f'{eta_minutes:.1f}min',
                        'Avg': f'{avg_time_per_trial:.1f}s/trial'
                    })

                logger.info(f"Trial {trial_num} complete")

            except Exception as e:
                logger.error(f"Error in trial {trial_num}: {e}")
                all_results["trials"].append({
                    "trial": trial_num,
                    "error": str(e),
                    "success": False
                })
                pbar.update(1)

        pbar.close()

        # Calculate summary statistics
        successful_trials = [t for t in all_results["trials"] if "error" not in t]

        total_duration = time.time() - start_time

        if successful_trials:
            avg_final_cooperation = sum(
                t.get("final_cooperation_rate", 0) for t in successful_trials
            ) / len(successful_trials)

            avg_final_payoff = sum(
                t.get("final_avg_payoff", 0) for t in successful_trials
            ) / len(successful_trials)

            all_results["summary"] = {
                "successful_trials": len(successful_trials),
                "failed_trials": len(all_results["trials"]) - len(successful_trials),
                "avg_final_cooperation_rate": avg_final_cooperation,
                "avg_final_payoff": avg_final_payoff,
                "total_duration_seconds": total_duration
            }

        # Save complete results
        with open(self.results_dir / "complete_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info("CURRICULUM EXPERIMENT COMPLETE")
        logger.info(f"Total duration: {total_duration/60:.1f} minutes")
        logger.info(f"Results saved to {self.results_dir}")
        if "summary" in all_results:
            logger.info(f"Final cooperation rate: {all_results['summary']['avg_final_cooperation_rate']:.2%}")
            logger.info(f"Final average payoff: {all_results['summary']['avg_final_payoff']:.1f}")
        logger.info(f"{'='*60}")

        return all_results
