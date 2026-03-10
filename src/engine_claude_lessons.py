#!/usr/bin/env python3
"""
Enhanced CurriculumEngine with Claude Opus 4.1 lesson generation
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List
import logging
import anthropic
from datetime import datetime

logger = logging.getLogger(__name__)

class ClaudeLessonGenerator:
    """
    Generate sophisticated lessons using Claude Opus 4.1
    """
    
    def __init__(self):
        """Initialize Claude client"""
        self.client = anthropic.Client(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        
    def generate_lesson(self, 
                       stage_results: Dict[str, Any], 
                       stage_config: Dict[str, Any],
                       stage_num: int,
                       previous_lessons: List[str] = None) -> str:
        """
        Generate a strategic lesson using Claude Opus 4.1
        
        Args:
            stage_results: Results from the completed stage
            stage_config: Configuration of the completed stage
            stage_num: Current stage number
            previous_lessons: Lessons from previous stages
            
        Returns:
            A sophisticated, actionable lesson string
        """
        
        # Extract key metrics
        game_name = stage_config.get("game", "Unknown")
        cooperation_rate = stage_results.get("cooperation_rate", 0)
        avg_payoff = stage_results.get("average_payoff", 0)
        rounds_played = stage_results.get("rounds_played", 3)
        
        # Extract behavioral patterns
        patterns = self._extract_behavioral_patterns(stage_results)
        
        # Build the prompt for Claude
        prompt = f"""You are analyzing results from a game theory experiment to extract strategic lessons for AI agents.

GAME DETAILS:
- Game Type: {game_name}
- Stage {stage_num} of a curriculum learning experiment
- Number of rounds played: {rounds_played}
- Number of players: {len(stage_results.get("total_payoffs", {}))}

PERFORMANCE METRICS:
- Overall cooperation rate: {cooperation_rate:.1%}
- Average payoff per player: {avg_payoff:.1f}
- Cooperation trajectory over rounds: {stage_results.get("cooperation_trajectory", [])}

KEY BEHAVIORAL PATTERNS:
{patterns}

{"PREVIOUS LESSONS LEARNED:" if previous_lessons else ""}
{chr(10).join(f"- {lesson}" for lesson in (previous_lessons or []))}

TASK:
Generate a concise (2-3 sentence) lesson that captures the KEY STRATEGIC INSIGHTS from this game.

The lesson should:
1. Identify SPECIFIC strategies that worked or failed and explain WHY
2. Extract concrete, actionable principles for future games
3. Be specific enough to guide behavior (e.g., "cooperate for first 2 rounds then match opponent" not just "cooperation is good")
4. Consider how this builds on or contradicts previous lessons if applicable

Focus on ACTIONABLE strategic guidance. For example:
- "Initiating cooperation in rounds 1-2 establishes trust, but requires immediate retaliation against defection to prevent exploitation"
- "Punishment should target only the lowest contributor, as indiscriminate punishment creates destructive cycles"
- "Groups with 3+ players require explicit coordination signals; silent cooperation strategies that work in 2-player games fail at scale"

Format: Start with "Lesson from {game_name}:" then provide your strategic insight.
"""
        
        # Retry logic for overloaded API
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                # Call Claude Opus 4.1
                response = self.client.messages.create(
                    model="claude-opus-4-1-20250805",  # Claude Opus 4.1
                    max_tokens=250,
                    temperature=0.7,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                break  # Success, exit retry loop
                
            except (anthropic._exceptions.OverloadedError, anthropic.InternalServerError) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"API error (likely overloaded): {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(f"Claude API persistently unavailable after {max_retries} attempts: {e}")
            except Exception as e:
                raise RuntimeError(f"Claude lesson generation failed: {e}")
        
        # Process successful response
        lesson = response.content[0].text.strip()
        
        # Ensure it starts with the expected format
        if not lesson.startswith(f"Lesson from {game_name}:"):
            lesson = f"Lesson from {game_name}: {lesson}"
            
        logger.info(f"Generated lesson with Claude: {lesson[:100]}...")
        return lesson
    
    def _extract_behavioral_patterns(self, stage_results: Dict[str, Any]) -> str:
        """
        Extract key behavioral patterns from gameplay
        
        Args:
            stage_results: Results from the stage
            
        Returns:
            String describing key patterns
        """
        patterns = []
        
        # Analyze cooperation trajectory
        trajectory = stage_results.get("cooperation_trajectory", [])
        if trajectory:
            # Check for cooperation breakdown
            for i in range(1, len(trajectory)):
                if trajectory[i] < trajectory[i-1] - 0.2:
                    patterns.append(f"- Major cooperation breakdown occurred in round {i+1} (from {trajectory[i-1]:.0%} to {trajectory[i]:.0%})")
                    break
            
            # Check for recovery
            if len(trajectory) > 2:
                if trajectory[-1] > trajectory[-2] + 0.2:
                    patterns.append(f"- Cooperation recovered in final round (from {trajectory[-2]:.0%} to {trajectory[-1]:.0%})")
        
        # Analyze player-specific patterns if available
        if "total_payoffs" in stage_results:
            payoffs = stage_results["total_payoffs"]
            if payoffs:
                # Identify outliers
                avg = sum(payoffs.values()) / len(payoffs)
                for player, payoff in payoffs.items():
                    if payoff > avg * 1.3:
                        patterns.append(f"- {player} achieved notably high payoff ({payoff:.0f} vs avg {avg:.0f})")
                    elif payoff < avg * 0.7:
                        patterns.append(f"- {player} achieved notably low payoff ({payoff:.0f} vs avg {avg:.0f})")
        
        # Note punishment patterns if this is a public goods game
        if "punishment_count" in stage_results:
            patterns.append(f"- Punishment was used {stage_results['punishment_count']} times")
            if stage_results.get("punishment_cycles", False):
                patterns.append(f"- Retaliatory punishment cycles detected")
        
        # Add round count context
        rounds = stage_results.get("rounds_played", 0)
        if rounds <= 3:
            patterns.append(f"- Short game ({rounds} rounds) limited opportunity for reputation building")
        
        return "\n".join(patterns) if patterns else "- No clear strategic patterns emerged"
    


def inject_enhanced_curriculum_context(agents: List[Dict[str, Any]], 
                                      stage_num: int,
                                      lessons: List[str]) -> List[Dict[str, Any]]:
    """
    Inject curriculum context with enhanced Claude-generated lessons
    
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
        curriculum_context = "\n\n### CURRICULUM LEARNING CONTEXT\n"
        curriculum_context += f"You are in Stage {stage_num} of a strategic learning curriculum.\n"
        
        if lessons:
            curriculum_context += "\nStrategic lessons from previous stages:\n"
            for i, lesson in enumerate(lessons, 1):
                curriculum_context += f"{i}. {lesson}\n"
            curriculum_context += "\nAPPLY these specific strategic principles to improve your performance in this stage."
            curriculum_context += "\nConsider: What worked before? What failed? How should you adapt your strategy?"
        else:
            curriculum_context += "This is your first stage. Observe carefully and learn from the outcomes."
        
        # Add context to agent config
        if "system_prompt_suffix" not in modified_agent:
            modified_agent["system_prompt_suffix"] = ""
        modified_agent["system_prompt_suffix"] += curriculum_context
        
        modified_agents.append(modified_agent)
    
    return modified_agents


if __name__ == "__main__":
    # Test the Claude lesson generator
    print("Testing Claude Opus 4.1 Lesson Generator...")
    
    # Sample stage results
    test_results = {
        "cooperation_rate": 0.2,
        "average_payoff": 150,
        "rounds_played": 3,
        "total_payoffs": {"Agent_1": 120, "Agent_2": 180, "Agent_3": 140, "Agent_4": 160},
        "cooperation_trajectory": [0.5, 0.25, 0.0]
    }
    
    test_config = {
        "game": "IteratedPublicGoodsGame"
    }
    
    generator = ClaudeLessonGenerator()
    lesson = generator.generate_lesson(test_results, test_config, 1, [])
    print(f"\nGenerated lesson:\n{lesson}")