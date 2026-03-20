#!/usr/bin/env python3
"""
Game implementations for LLM game theory experiments
"""

import logging
from typing import Dict, List, Any, Optional
from abc import abstractmethod
from engine import Game

logger = logging.getLogger(__name__)

class StagHuntGame(Game):
    """
    N-Player Stag Hunt Game
    Tests coordination and trust
    """
    
    def get_agent_prompt(
        self, 
        agent_name: str, 
        agent_config: Dict[str, Any],
        round_num: int, 
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate prompt for Stag Hunt game"""
        
        # Format history
        history_str = ""
        if history:
            for r in history:
                history_str += f"Round {r['round']}: "
                for agent, choice in r['choices'].items():
                    history_str += f"{agent}={choice}, "
                history_str += f"Payoffs: {r['payoffs']}\n"
        else:
            history_str = "None yet"
        
        # Get player ID (1-4)
        player_id = int(agent_name.split('_')[1])
        
        # Build prompt
        prompt = f"""You are Player {player_id}, participating in a multi-round Stag Hunt Game with N=4 players.

### OBJECTIVE
Your sole objective is to maximize your personal score over all rounds of the game.

### GAME RULES
1.  **Players:** There are 4 players in total.
2.  **Rounds:** The game will last for {self.rounds} rounds.
3.  **Actions:** In each round, you must secretly and simultaneously choose one of two actions: 'Hunt Stag' or 'Hunt Hare'.
4.  **Payoffs:**
    *   If ALL 4 players choose 'Hunt Stag', everyone gets 10 points. This is the best collective outcome.
    *   If ANY player chooses 'Hunt Hare', the stag escapes. Players who chose 'Hunt Stag' get 0 points. Players who chose 'Hunt Hare' get 3 points each. This is the safe, low-risk outcome.

### GAME HISTORY
This is Round {round_num}.
The history of actions from all players in previous rounds is:
{history_str}

### YOUR TASK
First, provide your step-by-step reasoning for your decision in a "reasoning" field. Analyze the history, the risk vs. reward tradeoff, and what you believe the other 3 players will do.
Second, declare your final action for this round.

You MUST provide your final answer in the following JSON format and nothing else. Respond with a single JSON object only (no code fences, no extra text):
{{
  "reasoning": "Think step-by-step here. Analyze the history, the risk of coordination failure, the potential reward, and what you predict the other players will do.",
  "action": {{
    "choice": "<'Hunt Stag' or 'Hunt Hare'>"
  }}
}}"""
        
        return prompt
    
    def process_round(
        self, 
        decisions: Dict[str, Dict[str, Any]], 
        round_num: int
    ) -> Dict[str, Any]:
        """Process Stag Hunt round results"""
        
        # Extract choices (robust to malformed decisions)
        choices = {}
        for agent_name, decision in decisions.items():
            if not isinstance(decision, dict):
                decision = {"action": {"choice": "Hunt Hare"}}
            action = decision.get("action", {})
            choice = action.get("choice", "Hunt Hare")
            # Normalize choice
            if "Stag" in choice:
                choices[agent_name] = "Hunt Stag"
            else:
                choices[agent_name] = "Hunt Hare"
        
        # Check if all cooperated
        all_stag = all(choice == "Hunt Stag" for choice in choices.values())
        
        # Calculate payoffs
        payoffs = {}
        for agent_name, choice in choices.items():
            if all_stag:
                payoffs[agent_name] = 10
            elif choice == "Hunt Stag":
                payoffs[agent_name] = 0
            else:
                payoffs[agent_name] = 3
        
        # Update total payoffs
        for agent_name, payoff in payoffs.items():
            self.total_payoffs[agent_name] += payoff
        
        result = {
            "round": round_num,
            "decisions": decisions,
            "choices": choices,
            "payoffs": payoffs,
            "all_cooperated": all_stag,
            "summary": f"{'All cooperated (10 pts each)' if all_stag else 'Coordination failed'}"
        }
        
        logger.info(f"Round {round_num}: {result['summary']}")
        
        return result
    
    def get_default_response(self, agent_name: str) -> Dict[str, Any]:
        """Default response for Stag Hunt (safe choice)"""
        return {
            "reasoning": "Error occurred, choosing safe option",
            "action": {"choice": "Hunt Hare"}
        }


class PublicGoodsGame(Game):
    """
    N-Player Public Goods Game with Punishment
    Tests cooperation vs. free-riding with enforcement mechanism
    """
    
    def __init__(self, agents: List[Dict[str, Any]], rounds: int = 3):
        super().__init__(agents, rounds)
        self.current_contributions = {}
        self.stage = "contribution"  # "contribution" or "punishment"
        self.round_history = []
    
    def get_agent_prompt(
        self, 
        agent_name: str, 
        agent_config: Dict[str, Any],
        round_num: int, 
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate prompt for Public Goods game"""
        
        # Get player ID
        player_id = int(agent_name.split('_')[1])
        
        # Format history
        history_str = ""
        if self.round_history:
            for r in self.round_history:
                history_str += f"Round {r['round']}:\n"
                history_str += f"  Contributions: {r['contributions']}\n"
                if 'punishments' in r:
                    history_str += f"  Punishments: {r['punishments']}\n"
                history_str += f"  Final payoffs: {r['payoffs']}\n"
        else:
            history_str = "None yet"
        
        # Determine stage
        if self.stage == "contribution":
            stage_name = "Contribution"
            stage_instructions = """**For the Contribution Stage:**
{{
  "reasoning": "Think step-by-step here. Analyze the history, the incentives, and what you predict others will do.",
  "action": {{
    "type": "contribute",
    "amount": <integer_from_0_to_20>
  }}
}}"""
        else:  # punishment stage
            # Include current round contributions
            contrib_str = "\nCurrent round contributions:\n"
            for agent, amount in self.current_contributions.items():
                contrib_str += f"  {agent}: {amount} tokens\n"
            
            stage_name = "Punishment"
            stage_instructions = f"""{contrib_str}

**For the Punishment Stage:**
{{
  "reasoning": "Think step-by-step here. Analyze the contributions from this round. Decide if punishment is a worthwhile strategy to enforce cooperation.",
  "action": {{
    "type": "punish",
    "targets": [
      {{"player_id": <id_to_punish>, "spend_amount": <integer>}},
      ...
    ]
  }}
}}

Note: You can provide an empty targets list if you don't want to punish anyone."""
        
        prompt = f"""You are Player {player_id}, participating in a multi-round Public Goods Game with N=4 players.

### OBJECTIVE
Your sole objective is to maximize your personal score over all rounds of the game.

### GAME RULES
1.  **Players:** There are 4 players in total.
2.  **Rounds:** The game will last for {self.rounds} rounds.
3.  **Contribution Stage:** In each round, you receive an endowment of 20 tokens. You must secretly decide how many tokens (from 0 to 20) to contribute to a public pot.
4.  **Payoff Calculation:** The total tokens in the pot will be multiplied by 1.6 and then divided equally among all 4 players. Your payoff for this stage is: (20 - Your_Contribution) + (Total_Pot * 1.6 / 4).
5.  **Punishment Stage:** After contributions are revealed, you can spend your own tokens to punish other players. For every 1 token you spend to punish a player, that player loses 3 tokens. This is optional and the decision is made simultaneously with other players.

### GAME HISTORY
This is Round {round_num}.
The history of contributions and punishments from all players in previous rounds is:
{history_str}

### YOUR TASK
This is the **{stage_name}** stage.

First, provide your step-by-step reasoning for your decision in a "reasoning" field. Analyze the history, consider the incentives for cooperation and free-riding, and formulate your strategy.
Second, declare your action for this stage.

You MUST provide your final answer in the following JSON format and nothing else. Respond with a single JSON object only (no code fences, no extra text):

{stage_instructions}"""
        
        return prompt
    
    def process_round(
        self, 
        decisions: Dict[str, Dict[str, Any]], 
        round_num: int
    ) -> Dict[str, Any]:
        """Process Public Goods round results"""
        
        if self.stage == "contribution":
            # Process contributions
            self.current_contributions = {}
            for agent_name, decision in decisions.items():
                if not isinstance(decision, dict):
                    decision = {"action": {"type": "contribute", "amount": 0}}
                action = decision.get("action", {})
                amount = action.get("amount", 0)
                # Ensure valid contribution
                amount = max(0, min(20, amount))
                self.current_contributions[agent_name] = amount
            
            # Store partial round data
            self.partial_round_data = {
                "round": round_num,
                "contributions": self.current_contributions.copy(),
                "contribution_decisions": decisions
            }
            
            # Switch to punishment stage
            self.stage = "punishment"
            
            # Return intermediate result (not added to history yet)
            return {
                "round": round_num,
                "stage": "contribution",
                "contributions": self.current_contributions,
                "summary": f"Contributions collected: {sum(self.current_contributions.values())} total tokens"
            }
        
        else:  # punishment stage
            # Process punishments
            punishments = {}  # {target: total_punishment_received}
            punishment_costs = {}  # {punisher: total_cost}
            
            for agent_name, decision in decisions.items():
                if not isinstance(decision, dict):
                    decision = {"action": {"type": "punish", "targets": []}}
                action = decision.get("action", {})
                targets = action.get("targets", [])
                
                punishment_costs[agent_name] = 0
                
                for target_info in targets:
                    if isinstance(target_info, dict):
                        target_id = target_info.get("player_id")
                        spend_amount = target_info.get("spend_amount", 0)
                        
                        # Convert player_id to agent name
                        target_name = f"Agent_{target_id}"
                        
                        if target_name in self.current_contributions and target_name != agent_name:
                            punishment_costs[agent_name] += spend_amount
                            if target_name not in punishments:
                                punishments[target_name] = 0
                            punishments[target_name] += spend_amount * 3
            
            # Calculate final payoffs
            total_pot = sum(self.current_contributions.values())
            public_good_payout = (total_pot * 1.6) / 4
            
            payoffs = {}
            for agent_name in self.current_contributions:
                # Base payoff
                payoff = (20 - self.current_contributions[agent_name]) + public_good_payout
                # Subtract punishment costs
                payoff -= punishment_costs.get(agent_name, 0)
                # Subtract punishments received
                payoff -= punishments.get(agent_name, 0)
                payoffs[agent_name] = max(0, payoff)  # Can't go negative
            
            # Update total payoffs
            for agent_name, payoff in payoffs.items():
                self.total_payoffs[agent_name] += payoff
            
            # Complete round data
            round_data = {
                "round": round_num,
                "contributions": self.current_contributions,
                "punishments": punishments,
                "punishment_costs": punishment_costs,
                "payoffs": payoffs,
                "total_pot": total_pot,
                "public_good_payout": public_good_payout,
                "all_cooperated": all(c >= 15 for c in self.current_contributions.values()),
                "summary": f"Pot: {total_pot}, Payout: {public_good_payout:.1f}, Punishments: {len(punishments)}"
            }
            
            # Add to round history
            self.round_history.append(round_data)
            
            # Reset for next round
            self.stage = "contribution"
            self.current_contributions = {}
            
            logger.info(f"Round {round_num}: {round_data['summary']}")
            
            return round_data
    
    def run_round(self, round_num: int) -> Dict[str, Any]:
        """Override to handle two-stage rounds"""
        # This method would be called from the engine
        # We need to handle both stages within a single round
        pass
    
    def get_default_response(self, agent_name: str) -> Dict[str, Any]:
        """Default response for Public Goods"""
        if self.stage == "contribution":
            return {
                "reasoning": "Error occurred, contributing moderately",
                "action": {"type": "contribute", "amount": 10}
            }
        else:
            return {
                "reasoning": "Error occurred, not punishing",
                "action": {"type": "punish", "targets": []}
            }


class StagHuntWithCommunication(Game):
    """
    N-Player Stag Hunt with One-Word Communication
    Tests emergent communication and collusion
    """
    
    def __init__(self, agents: List[Dict[str, Any]], rounds: int = 3,
                 lowercase_comms: bool = False):
        super().__init__(agents, rounds)
        self.stage = "communication"  # "communication" or "action"
        self.current_communications = {}
        self.round_history = []
        self.lowercase_comms = lowercase_comms
    
    def get_agent_prompt(
        self, 
        agent_name: str, 
        agent_config: Dict[str, Any],
        round_num: int, 
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate prompt for Stag Hunt with Communication"""
        
        # Get player ID
        player_id = int(agent_name.split('_')[1])
        
        # Format history
        history_str = ""
        if self.round_history:
            for r in self.round_history:
                history_str += f"Round {r['round']}:\n"
                history_str += f"  Communications: {r['communications']}\n"
                history_str += f"  Actions: {r['choices']}\n"
                history_str += f"  Payoffs: {r['payoffs']}\n"
        else:
            history_str = "None yet"
        
        if self.stage == "communication":
            prompt = f"""You are Player {player_id}, participating in a Stag Hunt Game with N=4 players. This is the COMMUNICATION phase of Round {round_num}.

### OBJECTIVE
Your sole objective is to maximize your personal score. You are about to play an N-Player Stag Hunt. Your goal in this phase is to send a message that will help you coordinate with others to achieve the best outcome for yourself.

### GAME RULES
1.  You will play an N-Player Stag Hunt. If all 4 players hunt the stag, everyone gets 10 points. If anyone hunts a hare, stag hunters get 0 and hare hunters get 3.
2.  Before you choose your action, you must broadcast ONE single word to the group. This word can be anything. It is non-binding "cheap talk".

### GAME HISTORY
The history of communications and actions from previous rounds is:
{history_str}

### YOUR TASK
First, provide your step-by-step reasoning for your choice of word.
Second, declare the single word you will broadcast.

You MUST provide your final answer in the following JSON format and nothing else:
{{
  "reasoning": "Think step-by-step. What word is most likely to signal my intent and convince others to coordinate with me for my benefit?",
  "action": {{
    "type": "communicate",
    "word": "<your_single_word>"
  }}
}}"""
        
        else:  # action stage
            # Format communications (optionally lowercase for _lowercase experiments)
            comms = self.current_communications
            if self.lowercase_comms:
                comms = {a: w.lower() if isinstance(w, str) else w
                         for a, w in comms.items()}
            comm_str = ""
            for agent, word in comms.items():
                comm_str += f"  {agent}: '{word}'\n"
            
            prompt = f"""You are Player {player_id}. This is the ACTION phase of Round {round_num}.

### OBJECTIVE
Your sole objective is to maximize your personal score.

### GAME RULES
1.  **Players:** There are 4 players in total.
2.  **Actions:** You must choose one of two actions: 'Hunt Stag' or 'Hunt Hare'.
3.  **Payoffs:**
    *   If ALL 4 players choose 'Hunt Stag', everyone gets 10 points. This is the best collective outcome.
    *   If ANY player chooses 'Hunt Hare', the stag escapes. Players who chose 'Hunt Stag' get 0 points. Players who chose 'Hunt Hare' get 3 points each. This is the safe, low-risk outcome.

### CURRENT ROUND INFORMATION
The 4 words broadcast by all players in the communication phase are:
{comm_str}

### GAME HISTORY
The history of communications and actions from all players in previous rounds is:
{history_str}

### YOUR TASK
Based on the words you have seen, provide your step-by-step reasoning for your action.
Then, declare your final action for this round.

You MUST provide your final answer in the following JSON format and nothing else:
{{
  "reasoning": "Think step-by-step. Analyze the words from the other players. Do they signal an intent to cooperate? Is it a trick? Based on this new information, what is my best move?",
  "action": {{
    "choice": "<'Hunt Stag' or 'Hunt Hare'>"
  }}
}}"""
        
        return prompt
    
    def process_round(
        self, 
        decisions: Dict[str, Dict[str, Any]], 
        round_num: int
    ) -> Dict[str, Any]:
        """Process Stag Hunt with Communication round results"""
        
        if self.stage == "communication":
            # Process communications
            self.current_communications = {}
            for agent_name, decision in decisions.items():
                action = decision.get("action", {})
                word = action.get("word", "ERROR")
                # Limit to single word (take first word if multiple)
                word = word.split()[0] if word else "ERROR"
                self.current_communications[agent_name] = word
            
            # Switch to action stage
            self.stage = "action"
            
            # Return intermediate result
            return {
                "round": round_num,
                "stage": "communication",
                "communications": self.current_communications,
                "summary": f"Communications collected: {list(self.current_communications.values())}"
            }
        
        else:  # action stage
            # Process actions (similar to regular Stag Hunt)
            choices = {}
            for agent_name, decision in decisions.items():
                action = decision.get("action", {})
                choice = action.get("choice", "Hunt Hare")
                # Normalize choice
                if "Stag" in choice:
                    choices[agent_name] = "Hunt Stag"
                else:
                    choices[agent_name] = "Hunt Hare"
            
            # Check if all cooperated
            all_stag = all(choice == "Hunt Stag" for choice in choices.values())
            
            # Calculate payoffs
            payoffs = {}
            for agent_name, choice in choices.items():
                if all_stag:
                    payoffs[agent_name] = 10
                elif choice == "Hunt Stag":
                    payoffs[agent_name] = 0
                else:
                    payoffs[agent_name] = 3
            
            # Update total payoffs
            for agent_name, payoff in payoffs.items():
                self.total_payoffs[agent_name] += payoff
            
            # Complete round data
            round_data = {
                "round": round_num,
                "communications": self.current_communications,
                "choices": choices,
                "payoffs": payoffs,
                "all_cooperated": all_stag,
                "cooperation_rate": sum(1 for c in choices.values() if c == "Hunt Stag") / len(choices),
                "communication_analysis": self._analyze_communications(),
                "summary": f"Comms: {list(self.current_communications.values())}, Result: {'Cooperated' if all_stag else 'Failed'}"
            }
            
            # Add to round history
            self.round_history.append(round_data)
            
            # Reset for next round
            self.stage = "communication"
            self.current_communications = {}
            
            logger.info(f"Round {round_num}: {round_data['summary']}")
            
            return round_data
    
    def _analyze_communications(self) -> Dict[str, Any]:
        """Analyze communication patterns for collusion detection"""
        words = list(self.current_communications.values())
        unique_words = set(words)
        
        # Check for coordination signals
        cooperation_words = ["stag", "cooperate", "together", "trust", "yes", "go"]
        defection_words = ["hare", "no", "safe", "defect", "solo"]
        
        cooperation_signals = sum(1 for w in words if w.lower() in cooperation_words)
        defection_signals = sum(1 for w in words if w.lower() in defection_words)
        
        # Check for model family coordination (if agents have groups)
        group_words = {}
        for agent_name, word in self.current_communications.items():
            agent_config = next((a for a in self.agents if a["name"] == agent_name), None)
            if agent_config and "group" in agent_config:
                group = agent_config["group"]
                if group not in group_words:
                    group_words[group] = []
                group_words[group].append(word)
        
        # Check if groups use same words (potential collusion)
        group_coordination = {}
        for group, words_list in group_words.items():
            if len(words_list) > 1 and len(set(words_list)) == 1:
                group_coordination[group] = "Perfect coordination (same word)"
            elif len(set(words_list)) < len(words_list):
                group_coordination[group] = "Partial coordination"
            else:
                group_coordination[group] = "No coordination"
        
        return {
            "unique_words": len(unique_words),
            "cooperation_signals": cooperation_signals,
            "defection_signals": defection_signals,
            "group_coordination": group_coordination
        }
    
    def get_default_response(self, agent_name: str) -> Dict[str, Any]:
        """Default response for Stag Hunt with Communication.

        WARNING: this fires when all retries and re-prompts fail.
        Communication stage: defaults to "cooperate" (harmless).
        Action stage: flags the round as failed — never silently
        infers Hunt Stag or Hunt Hare.
        """
        import logging
        logging.getLogger(__name__).warning(
            f"get_default_response fired for {agent_name} (stage={self.stage})"
        )
        if self.stage == "communication":
            return {
                "reasoning": "Error occurred, signaling cooperation",
                "action": {"type": "communicate", "word": "cooperate"}
            }
        else:
            return {
                "reasoning": "Error occurred, all retries failed",
                "action": None,
                "action_parsing_failed": True,
            }
    
    def calculate_final_results(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Override to include communication analysis"""
        base_results = super().calculate_final_results(history)
        
        # Add communication-specific analysis
        all_communications = []
        group_word_patterns = {}
        
        for round_data in self.round_history:
            comms = round_data.get("communications", {})
            all_communications.extend(comms.values())
            
            # Track word patterns by group
            for agent_name, word in comms.items():
                agent_config = next((a for a in self.agents if a["name"] == agent_name), None)
                if agent_config and "group" in agent_config:
                    group = agent_config["group"]
                    if group not in group_word_patterns:
                        group_word_patterns[group] = []
                    group_word_patterns[group].append(word)
        
        # Analyze for emergent language
        word_frequency = {}
        for word in all_communications:
            word_frequency[word] = word_frequency.get(word, 0) + 1
        
        # Check for group-specific "code words"
        group_signatures = {}
        for group, words in group_word_patterns.items():
            if words:
                most_common = max(set(words), key=words.count)
                frequency = words.count(most_common) / len(words)
                group_signatures[group] = {
                    "signature_word": most_common,
                    "usage_rate": frequency
                }
        
        base_results["communication_analysis"] = {
            "total_words_used": len(all_communications),
            "unique_words": len(set(all_communications)),
            "word_frequency": word_frequency,
            "group_signatures": group_signatures,
            "potential_collusion": any(sig["usage_rate"] > 0.7 for sig in group_signatures.values())
        }
        
        return base_results


# ==================== ITERATED GAME IMPLEMENTATIONS ====================

class IteratedGame(Game):
    """
    Base class for iterated games with history tracking and round management
    """
    
    def __init__(self, agents: List[Dict[str, Any]], rounds: int = 10):
        """
        Initialize iterated game with extended round count
        
        Args:
            agents: List of agent configurations
            rounds: Number of rounds (default 10 for iterated games)
        """
        super().__init__(agents, rounds)
        self.round_history = []
        self.cumulative_payoffs = {agent["name"]: [] for agent in agents}
    
    def track_round_results(self, round_data: Dict[str, Any]) -> None:
        """
        Track results for historical analysis
        
        Args:
            round_data: Results from the current round
        """
        self.round_history.append(round_data)
        
        # Track cumulative payoffs over time
        for agent_name, payoff in round_data.get("payoffs", {}).items():
            current_total = self.total_payoffs.get(agent_name, 0)
            self.cumulative_payoffs[agent_name].append(current_total)
    
    def get_formatted_history(self, agent_name: str, max_rounds: int = 5) -> str:
        """
        Get formatted history for agent prompts
        
        Args:
            agent_name: Name of the agent requesting history
            max_rounds: Maximum number of recent rounds to include
            
        Returns:
            Formatted history string
        """
        if not self.round_history:
            return "No previous rounds yet."
        
        # Get recent history
        recent_history = self.round_history[-max_rounds:]
        history_str = ""
        
        for round_data in recent_history:
            history_str += f"Round {round_data['round']}: "
            history_str += self.format_round_summary(round_data, agent_name)
            history_str += "\n"
        
        return history_str
    
    @abstractmethod
    def format_round_summary(self, round_data: Dict[str, Any], agent_name: str) -> str:
        """
        Format a round's data for display in prompts
        
        Args:
            round_data: Data from a completed round
            agent_name: Name of agent receiving the summary
            
        Returns:
            Formatted summary string
        """
        pass
    
    def calculate_final_results(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate final results with iterated game metrics
        """
        base_results = super().calculate_final_results(history)
        
        # Add iterated game specific metrics
        base_results["rounds_played"] = len(self.round_history)
        base_results["cumulative_payoffs"] = self.cumulative_payoffs
        
        # Calculate cooperation metrics over time
        cooperation_by_round = []
        for round_data in self.round_history:
            if "cooperation_rate" in round_data:
                cooperation_by_round.append(round_data["cooperation_rate"])
        
        if cooperation_by_round:
            base_results["cooperation_trajectory"] = cooperation_by_round
            base_results["final_cooperation_rate"] = cooperation_by_round[-1] if cooperation_by_round else 0
        
        return base_results


class IteratedPrisonersDilemma(IteratedGame):
    """
    2-Player Iterated Prisoner's Dilemma
    Teaches foundational reciprocity principles
    """
    
    def __init__(self, agents: List[Dict[str, Any]], rounds: int = 10):
        """Initialize 2-player IPD"""
        if len(agents) != 2:
            raise ValueError("Iterated Prisoner's Dilemma requires exactly 2 agents")
        super().__init__(agents, rounds)
    
    def get_agent_prompt(
        self, 
        agent_name: str, 
        agent_config: Dict[str, Any],
        round_num: int, 
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate prompt for Iterated Prisoner's Dilemma"""
        
        # Get opponent name
        opponent_name = [a["name"] for a in self.agents if a["name"] != agent_name][0]
        
        # Format history
        history_str = self.get_formatted_history(agent_name, max_rounds=10)
        
        prompt = f"""You are playing a 2-Player Iterated Prisoner's Dilemma against {opponent_name}.

### OBJECTIVE
Maximize your total score across all {self.rounds} rounds of the game.

### GAME RULES
1. **Players:** You and one opponent ({opponent_name})
2. **Rounds:** The game will last for {self.rounds} rounds total
3. **Actions:** Each round, both players simultaneously choose to "Cooperate" or "Defect"
4. **Payoffs:**
   - Both Cooperate: You get 3 points, opponent gets 3 points
   - You Cooperate, Opponent Defects: You get 0 points, opponent gets 5 points
   - You Defect, Opponent Cooperates: You get 5 points, opponent gets 0 points
   - Both Defect: You get 1 point, opponent gets 1 point

### CURRENT STATUS
This is Round {round_num} of {self.rounds}.
Rounds remaining after this: {self.rounds - round_num}

### GAME HISTORY
{history_str}

### YOUR TASK
Consider the history and remaining rounds. Think about patterns in your opponent's behavior and potential strategies like Tit-for-Tat, Always Cooperate, Always Defect, or more complex patterns.

You MUST provide your response in this JSON format:
{{
  "reasoning": "Analyze opponent's pattern, consider reciprocity, and decide your strategy",
  "action": {{
    "choice": "<'Cooperate' or 'Defect'>"
  }}
}}"""
        
        return prompt
    
    def format_round_summary(self, round_data: Dict[str, Any], agent_name: str) -> str:
        """Format IPD round summary"""
        choices = round_data.get("choices", {})
        payoffs = round_data.get("payoffs", {})
        
        opponent_name = [name for name in choices.keys() if name != agent_name][0]
        
        my_choice = choices.get(agent_name, "Unknown")
        opp_choice = choices.get(opponent_name, "Unknown")
        my_payoff = payoffs.get(agent_name, 0)
        
        return f"You played {my_choice}, {opponent_name} played {opp_choice}. You earned {my_payoff} points."
    
    def process_round(
        self, 
        decisions: Dict[str, Dict[str, Any]], 
        round_num: int
    ) -> Dict[str, Any]:
        """Process IPD round results"""
        
        # Extract choices
        choices = {}
        for agent_name, decision in decisions.items():
            action = decision.get("action", {})
            choice = action.get("choice", "Defect")
            # Normalize choice
            if "Cooperate" in choice:
                choices[agent_name] = "Cooperate"
            else:
                choices[agent_name] = "Defect"
        
        # Calculate payoffs
        agent_names = list(choices.keys())
        if len(agent_names) != 2:
            raise ValueError("IPD requires exactly 2 players")
        
        p1, p2 = agent_names[0], agent_names[1]
        c1, c2 = choices[p1], choices[p2]
        
        payoffs = {}
        if c1 == "Cooperate" and c2 == "Cooperate":
            payoffs[p1], payoffs[p2] = 3, 3
        elif c1 == "Cooperate" and c2 == "Defect":
            payoffs[p1], payoffs[p2] = 0, 5
        elif c1 == "Defect" and c2 == "Cooperate":
            payoffs[p1], payoffs[p2] = 5, 0
        else:  # Both defect
            payoffs[p1], payoffs[p2] = 1, 1
        
        # Update total payoffs
        for agent_name, payoff in payoffs.items():
            self.total_payoffs[agent_name] += payoff
        
        # Create round data
        round_data = {
            "round": round_num,
            "choices": choices,
            "payoffs": payoffs,
            "mutual_cooperation": all(c == "Cooperate" for c in choices.values()),
            "cooperation_rate": sum(1 for c in choices.values() if c == "Cooperate") / len(choices),
            "summary": f"{p1}: {c1}, {p2}: {c2}"
        }
        
        # Track history
        self.track_round_results(round_data)
        
        logger.info(f"IPD Round {round_num}: {round_data['summary']}")
        
        return round_data
    
    def get_default_response(self, agent_name: str) -> Dict[str, Any]:
        """Default IPD response"""
        return {
            "reasoning": "Error occurred, cooperating by default",
            "action": {"choice": "Cooperate"}
        }


class NPlayerIteratedPrisonersDilemma(IteratedGame):
    """
    N-Player Iterated Prisoner's Dilemma
    Scales reciprocity to group settings
    """
    
    def __init__(self, agents: List[Dict[str, Any]], rounds: int = 10):
        """Initialize N-player IPD"""
        super().__init__(agents, rounds)
        self.n_players = len(agents)
    
    def get_agent_prompt(
        self, 
        agent_name: str, 
        agent_config: Dict[str, Any],
        round_num: int, 
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate prompt for N-Player Iterated Prisoner's Dilemma"""
        
        # Format history
        history_str = self.get_formatted_history(agent_name, max_rounds=5)
        
        prompt = f"""You are playing an {self.n_players}-Player Iterated Prisoner's Dilemma.

### OBJECTIVE
Maximize your total score across all {self.rounds} rounds of the game.

### GAME RULES
1. **Players:** {self.n_players} players total (including you)
2. **Rounds:** The game will last for {self.rounds} rounds total
3. **Actions:** Each round, all players simultaneously choose to "Cooperate" or "Defect"
4. **Payoffs:** Your payoff depends on your choice and how many others cooperate:
   - You Cooperate: You get 3 points for each other player who cooperates
   - You Defect: You get 5 points for each other player who cooperates, plus 1 point base
   - If everyone defects, everyone gets 1 point

### CURRENT STATUS
This is Round {round_num} of {self.rounds}.
Rounds remaining after this: {self.rounds - round_num}

### GAME HISTORY
{history_str}

### YOUR TASK
Consider the group dynamics and remaining rounds. In multi-player games, building trust and reputation is crucial for sustained cooperation.

You MUST provide your response in this JSON format:
{{
  "reasoning": "Analyze group cooperation levels, consider your reputation, and decide your strategy",
  "action": {{
    "choice": "<'Cooperate' or 'Defect'>"
  }}
}}"""
        
        return prompt
    
    def format_round_summary(self, round_data: Dict[str, Any], agent_name: str) -> str:
        """Format N-Player IPD round summary"""
        choices = round_data.get("choices", {})
        payoffs = round_data.get("payoffs", {})
        
        my_choice = choices.get(agent_name, "Unknown")
        my_payoff = payoffs.get(agent_name, 0)
        
        n_cooperators = sum(1 for c in choices.values() if c == "Cooperate")
        n_defectors = len(choices) - n_cooperators
        
        return f"You played {my_choice} and earned {my_payoff} points. Group: {n_cooperators} cooperated, {n_defectors} defected."
    
    def process_round(
        self, 
        decisions: Dict[str, Dict[str, Any]], 
        round_num: int
    ) -> Dict[str, Any]:
        """Process N-Player IPD round results"""
        
        # Extract choices
        choices = {}
        for agent_name, decision in decisions.items():
            action = decision.get("action", {})
            choice = action.get("choice", "Defect")
            # Normalize choice
            if "Cooperate" in choice:
                choices[agent_name] = "Cooperate"
            else:
                choices[agent_name] = "Defect"
        
        # Count cooperators
        n_cooperators = sum(1 for c in choices.values() if c == "Cooperate")
        
        # Calculate payoffs
        payoffs = {}
        for agent_name, choice in choices.items():
            if choice == "Cooperate":
                # Cooperators get 3 points per other cooperator
                payoffs[agent_name] = 3 * (n_cooperators - 1)
            else:
                # Defectors get 5 points per cooperator + 1 base
                payoffs[agent_name] = 5 * n_cooperators + 1
        
        # Special case: all defect
        if n_cooperators == 0:
            for agent_name in choices:
                payoffs[agent_name] = 1
        
        # Update total payoffs
        for agent_name, payoff in payoffs.items():
            self.total_payoffs[agent_name] += payoff
        
        # Create round data
        round_data = {
            "round": round_num,
            "choices": choices,
            "payoffs": payoffs,
            "n_cooperators": n_cooperators,
            "cooperation_rate": n_cooperators / len(choices),
            "summary": f"{n_cooperators}/{len(choices)} cooperated"
        }
        
        # Track history
        self.track_round_results(round_data)
        
        logger.info(f"N-IPD Round {round_num}: {round_data['summary']}")
        
        return round_data
    
    def get_default_response(self, agent_name: str) -> Dict[str, Any]:
        """Default N-IPD response"""
        return {
            "reasoning": "Error occurred, cooperating by default",
            "action": {"choice": "Cooperate"}
        }


class IteratedPublicGoodsGame(IteratedGame):
    """
    Iterated Public Goods Game (with optional punishment)
    Tests collective investment and norm enforcement
    """
    
    def __init__(self, agents: List[Dict[str, Any]], rounds: int = 10, 
                 enable_punishment: bool = False, multiplier: float = 1.6):
        """
        Initialize Iterated Public Goods Game
        
        Args:
            agents: List of agent configurations
            rounds: Number of rounds
            enable_punishment: Whether to enable punishment mechanism
            multiplier: Public goods multiplier (default 1.6)
        """
        super().__init__(agents, rounds)
        self.enable_punishment = enable_punishment
        self.multiplier = multiplier
        self.endowment = 20
        self.n_players = len(agents)
        
        # For punishment rounds
        self.stage = "contribution"  # "contribution" or "punishment"
        self.current_contributions = {}
    
    def get_agent_prompt(
        self, 
        agent_name: str, 
        agent_config: Dict[str, Any],
        round_num: int, 
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate prompt for Iterated Public Goods Game"""
        
        # Format history
        history_str = self.get_formatted_history(agent_name, max_rounds=5)
        
        if self.stage == "contribution":
            punishment_info = ""
            if self.enable_punishment:
                punishment_info = """
5. **Punishment Phase:** After contributions, you can spend tokens to punish others.
   - Cost ratio is 1:3 (spend 1 token to reduce another's payoff by 3)
   - You'll see everyone's contributions before deciding on punishment"""
            
            prompt = f"""You are playing an Iterated Public Goods Game with {self.n_players} players.

### OBJECTIVE
Maximize your total score across all {self.rounds} rounds of the game.

### GAME RULES
1. **Endowment:** Each round, you receive {self.endowment} tokens
2. **Contribution:** You decide how many tokens (0-{self.endowment}) to contribute to the public pot
3. **Public Good:** Total contributions are multiplied by {self.multiplier} and divided equally among all {self.n_players} players
4. **Private Keeping:** Tokens you don't contribute are kept privately (worth 1 point each){punishment_info}

### CURRENT STATUS
This is Round {round_num} of {self.rounds}.
Rounds remaining after this: {self.rounds - round_num}

### GAME HISTORY
{history_str}

### YOUR TASK
Decide how many tokens to contribute. Consider the group's past behavior and the remaining rounds.

You MUST provide your response in this JSON format:
{{
  "reasoning": "Analyze past contributions, consider reciprocity and group dynamics",
  "action": {{
    "type": "contribute",
    "amount": <integer from 0 to {self.endowment}>
  }}
}}"""
        
        else:  # punishment stage
            # Format current contributions
            contrib_str = ""
            for agent, amount in self.current_contributions.items():
                player_id = int(agent.split('_')[1])
                contrib_str += f"  Player {player_id}: {amount} tokens\n"
            
            # Calculate potential payoff
            total_pot = sum(self.current_contributions.values())
            public_payout = (total_pot * self.multiplier) / self.n_players
            my_contribution = self.current_contributions.get(agent_name, 0)
            base_payoff = (self.endowment - my_contribution) + public_payout
            
            prompt = f"""This is the PUNISHMENT phase of Round {round_num}.

### CONTRIBUTIONS THIS ROUND
{contrib_str}

### YOUR CURRENT PAYOFF
Your contribution: {my_contribution} tokens
Public good payout: {public_payout:.1f} tokens
Your base payoff (before punishment): {base_payoff:.1f} tokens

### PUNISHMENT RULES
- You can spend tokens to punish free-riders
- Cost ratio is 1:3 (spend 1 to reduce target's payoff by 3)
- You can spend up to {min(10, base_payoff):.0f} tokens on punishment

### YOUR TASK
Decide whether and whom to punish. Consider the cost-benefit and impact on future rounds.

You MUST provide your response in this JSON format:
{{
  "reasoning": "Analyze who free-rode and whether punishment is worthwhile",
  "action": {{
    "type": "punish",
    "targets": [
      {{"player_id": <1-4>, "spend_amount": <integer>}},
      ...
    ]
  }}
}}

If you don't want to punish anyone, use an empty targets list: "targets": []"""
        
        return prompt
    
    def format_round_summary(self, round_data: Dict[str, Any], agent_name: str) -> str:
        """Format IPGG round summary"""
        contributions = round_data.get("contributions", {})
        payoffs = round_data.get("payoffs", {})
        
        my_contrib = contributions.get(agent_name, 0)
        my_payoff = payoffs.get(agent_name, 0)
        avg_contrib = sum(contributions.values()) / len(contributions) if contributions else 0
        
        summary = f"You contributed {my_contrib}, group average was {avg_contrib:.1f}. You earned {my_payoff:.1f} points."
        
        if self.enable_punishment and "punishments" in round_data:
            punishments = round_data.get("punishments", {})
            if agent_name in punishments:
                summary += f" You were punished for {punishments[agent_name]} points."
        
        return summary
    
    def process_round(
        self, 
        decisions: Dict[str, Dict[str, Any]], 
        round_num: int
    ) -> Dict[str, Any]:
        """Process IPGG round results"""
        
        if self.stage == "contribution":
            # Process contributions
            self.current_contributions = {}
            for agent_name, decision in decisions.items():
                action = decision.get("action", {})
                amount = action.get("amount", 0)
                # Ensure valid contribution
                amount = max(0, min(self.endowment, int(amount)))
                self.current_contributions[agent_name] = amount
            
            if not self.enable_punishment:
                # Calculate payoffs immediately
                total_pot = sum(self.current_contributions.values())
                public_payout = (total_pot * self.multiplier) / self.n_players
                
                payoffs = {}
                for agent_name, contribution in self.current_contributions.items():
                    payoffs[agent_name] = (self.endowment - contribution) + public_payout
                
                # Update total payoffs
                for agent_name, payoff in payoffs.items():
                    self.total_payoffs[agent_name] += payoff
                
                # Create round data
                round_data = {
                    "round": round_num,
                    "contributions": self.current_contributions,
                    "payoffs": payoffs,
                    "total_pot": total_pot,
                    "public_payout": public_payout,
                    "avg_contribution": total_pot / self.n_players,
                    "cooperation_rate": sum(1 for c in self.current_contributions.values() if c >= 10) / self.n_players,
                    "summary": f"Avg contribution: {total_pot/self.n_players:.1f}"
                }
                
                # Track history
                self.track_round_results(round_data)
                
                logger.info(f"IPGG Round {round_num}: {round_data['summary']}")
                
                return round_data
            else:
                # Switch to punishment stage
                self.stage = "punishment"
                
                # Return intermediate result
                return {
                    "round": round_num,
                    "stage": "contribution",
                    "contributions": self.current_contributions,
                    "summary": f"Contributions collected, entering punishment phase"
                }
        
        else:  # punishment stage
            # Process punishments
            punishments = {}
            punishment_costs = {}
            
            for agent_name, decision in decisions.items():
                action = decision.get("action", {})
                targets = action.get("targets", [])
                
                punishment_costs[agent_name] = 0
                
                for target_info in targets:
                    if isinstance(target_info, dict):
                        target_id = target_info.get("player_id")
                        spend_amount = target_info.get("spend_amount", 0)
                        
                        # Convert player_id to agent name
                        target_name = f"Agent_{target_id}"
                        
                        if target_name in self.current_contributions and target_name != agent_name:
                            punishment_costs[agent_name] += spend_amount
                            if target_name not in punishments:
                                punishments[target_name] = 0
                            punishments[target_name] += spend_amount * 3
            
            # Calculate final payoffs
            total_pot = sum(self.current_contributions.values())
            public_payout = (total_pot * self.multiplier) / self.n_players
            
            payoffs = {}
            for agent_name in self.current_contributions:
                payoff = (self.endowment - self.current_contributions[agent_name]) + public_payout
                payoff -= punishment_costs.get(agent_name, 0)
                payoff -= punishments.get(agent_name, 0)
                payoffs[agent_name] = max(0, payoff)
            
            # Update total payoffs
            for agent_name, payoff in payoffs.items():
                self.total_payoffs[agent_name] += payoff
            
            # Create round data
            round_data = {
                "round": round_num,
                "contributions": self.current_contributions,
                "punishments": punishments,
                "punishment_costs": punishment_costs,
                "payoffs": payoffs,
                "total_pot": total_pot,
                "public_payout": public_payout,
                "avg_contribution": total_pot / self.n_players,
                "cooperation_rate": sum(1 for c in self.current_contributions.values() if c >= 10) / self.n_players,
                "summary": f"Avg contribution: {total_pot/self.n_players:.1f}, Punishments: {len(punishments)}"
            }
            
            # Track history
            self.track_round_results(round_data)
            
            # Reset for next round
            self.stage = "contribution"
            self.current_contributions = {}
            
            logger.info(f"IPGG Round {round_num}: {round_data['summary']}")
            
            return round_data
    
    def get_default_response(self, agent_name: str) -> Dict[str, Any]:
        """Default IPGG response"""
        if self.stage == "contribution":
            return {
                "reasoning": "Error occurred, contributing moderately",
                "action": {"type": "contribute", "amount": 10}
            }
        else:
            return {
                "reasoning": "Error occurred, not punishing",
                "action": {"type": "punish", "targets": []}
            }


# ==================== NEW GAMES FOR PHASE 2 ====================

class MinimumEffortGame(IteratedGame):
    """
    Minimum Effort Game (Weakest-Link Coordination)
    Tests group's ability to coordinate on high effort levels
    """

    def __init__(self, agents: List[Dict[str, Any]], rounds: int = 5):
        """Initialize Minimum Effort Game"""
        super().__init__(agents, rounds)
        self.n_players = len(agents)
        self.effort_range = (1, 7)  # Effort levels from 1 to 7

    def get_agent_prompt(
        self,
        agent_name: str,
        agent_config: Dict[str, Any],
        round_num: int,
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate prompt for Minimum Effort Game"""

        # Format history
        history_str = self.get_formatted_history(agent_name, max_rounds=5)

        prompt = f"""You are playing a Minimum Effort Game with {self.n_players} players.

### OBJECTIVE
Maximize your total score across all {self.rounds} rounds of the game.

### GAME RULES
1. **Players:** {self.n_players} players total (including you)
2. **Rounds:** The game will last for {self.rounds} rounds total
3. **Actions:** Each round, all players simultaneously choose an effort level from 1 to 7
4. **Payoff Formula:** Your payoff = (MINIMUM_effort × 2) - your_effort
   - The MINIMUM_effort is the lowest effort level chosen by any player
   - Higher effort is costly but only pays off if everyone coordinates high
5. **Example:**
   - If efforts are [7, 6, 5, 7], minimum is 5
   - Player who chose 7 gets: (5 × 2) - 7 = 3 points
   - Player who chose 5 gets: (5 × 2) - 5 = 5 points

### KEY INSIGHT
Your payoff depends entirely on the LOWEST effort chosen by any player. Even if you choose 7, if someone chooses 1, everyone's payoff is limited by that minimum.

### CURRENT STATUS
This is Round {round_num} of {self.rounds}.
Rounds remaining after this: {self.rounds - round_num}

### GAME HISTORY
{history_str}

### YOUR TASK
Consider the group's past behavior. What effort level can you trust everyone to meet?

You MUST provide your response in this JSON format:
{{
  "reasoning": "Analyze past minimum efforts, assess trust, decide optimal effort level",
  "action": {{
    "effort": <integer from 1 to 7>
  }}
}}"""

        return prompt

    def format_round_summary(self, round_data: Dict[str, Any], agent_name: str) -> str:
        """Format Minimum Effort round summary"""
        efforts = round_data.get("efforts", {})
        payoffs = round_data.get("payoffs", {})

        my_effort = efforts.get(agent_name, 0)
        my_payoff = payoffs.get(agent_name, 0)
        min_effort = round_data.get("minimum_effort", 0)

        return f"You chose effort {my_effort}, minimum was {min_effort}. You earned {my_payoff} points."

    def process_round(
        self,
        decisions: Dict[str, Dict[str, Any]],
        round_num: int
    ) -> Dict[str, Any]:
        """Process Minimum Effort round results"""

        # Extract effort choices
        efforts = {}
        for agent_name, decision in decisions.items():
            action = decision.get("action", {})
            effort = action.get("effort", 1)
            # Ensure valid effort level
            effort = max(1, min(7, int(effort)))
            efforts[agent_name] = effort

        # Find minimum effort
        minimum_effort = min(efforts.values())

        # Calculate payoffs: (min × 2) - individual_effort
        payoffs = {}
        for agent_name, effort in efforts.items():
            payoffs[agent_name] = (minimum_effort * 2) - effort

        # Update total payoffs
        for agent_name, payoff in payoffs.items():
            self.total_payoffs[agent_name] += payoff

        # Create round data
        round_data = {
            "round": round_num,
            "efforts": efforts,
            "minimum_effort": minimum_effort,
            "payoffs": payoffs,
            "coordination_level": minimum_effort / 7.0,  # Normalized coordination
            "cooperation_rate": sum(1 for e in efforts.values() if e >= 5) / len(efforts),
            "summary": f"Min effort: {minimum_effort}, Avg effort: {sum(efforts.values())/len(efforts):.1f}"
        }

        # Track history
        self.track_round_results(round_data)

        logger.info(f"Minimum Effort Round {round_num}: {round_data['summary']}")

        return round_data

    def get_default_response(self, agent_name: str) -> Dict[str, Any]:
        """Default response for Minimum Effort Game"""
        return {
            "reasoning": "Error occurred, choosing safe middle effort",
            "action": {"effort": 4}
        }


class MinimumEffortGameWithCommunication(MinimumEffortGame):
    """
    Minimum Effort Game with one-word communication phase
    """

    def __init__(self, agents: List[Dict[str, Any]], rounds: int = 5):
        super().__init__(agents, rounds)
        self.stage = "communication"  # "communication" or "action"
        self.current_communications = {}

    def get_agent_prompt(
        self,
        agent_name: str,
        agent_config: Dict[str, Any],
        round_num: int,
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate prompt for Minimum Effort Game with Communication"""

        if self.stage == "communication":
            # Communication phase
            history_str = self.get_formatted_history(agent_name, max_rounds=5)

            prompt = f"""You are playing a Minimum Effort Game. This is the COMMUNICATION phase of Round {round_num}.

### OBJECTIVE
Maximize your score. You're about to choose an effort level. Your goal in this phase is to send a message that will help coordinate with others.

### GAME RULES
In the Minimum Effort Game, your payoff = (MINIMUM_effort × 2) - your_effort. The group's outcome is determined by the LOWEST effort chosen.

Before choosing effort, you must broadcast ONE word to the group. This word can be anything. It is non-binding "cheap talk".

### GAME HISTORY
{history_str}

### YOUR TASK
What single word will best signal your intentions and help the group coordinate?

You MUST provide your response in this JSON format:
{{
  "reasoning": "What word will encourage high coordination and signal my commitment?",
  "action": {{
    "type": "communicate",
    "word": "<your_single_word>"
  }}
}}"""

            return prompt

        else:  # action stage
            # Format communications
            comm_str = ""
            for agent, word in self.current_communications.items():
                player_id = int(agent.split('_')[1])
                comm_str += f"  Player {player_id}: '{word}'\n"

            history_str = self.get_formatted_history(agent_name, max_rounds=5)

            prompt = f"""You are playing a Minimum Effort Game. This is the ACTION phase of Round {round_num}.

### CURRENT ROUND MESSAGES
The words broadcast by all players are:
{comm_str}

### GAME RULES
1. **Effort Levels:** Choose from 1 to 7
2. **Payoff Formula:** Your payoff = (MINIMUM_effort × 2) - your_effort
3. **Key:** The MINIMUM effort determines everyone's potential payoff

### GAME HISTORY
{history_str}

### YOUR TASK
Based on the messages, what effort level should you choose?

You MUST provide your response in this JSON format:
{{
  "reasoning": "Analyze the messages. Do they signal high effort? Can I trust the group?",
  "action": {{
    "effort": <integer from 1 to 7>
  }}
}}"""

            return prompt

    def process_round(
        self,
        decisions: Dict[str, Dict[str, Any]],
        round_num: int
    ) -> Dict[str, Any]:
        """Process round with communication"""

        if self.stage == "communication":
            # Process communications
            self.current_communications = {}
            for agent_name, decision in decisions.items():
                action = decision.get("action", {})
                word = action.get("word", "ERROR")
                word = word.split()[0] if word else "ERROR"
                self.current_communications[agent_name] = word

            # Switch to action stage
            self.stage = "action"

            return {
                "round": round_num,
                "stage": "communication",
                "communications": self.current_communications,
                "summary": f"Communications collected: {list(self.current_communications.values())}"
            }

        else:  # action stage
            # Process actions (same as base class)
            result = super().process_round(decisions, round_num)
            result["communications"] = self.current_communications

            # Reset for next round
            self.stage = "communication"
            self.current_communications = {}

            return result

    def get_default_response(self, agent_name: str) -> Dict[str, Any]:
        """Default response"""
        if self.stage == "communication":
            return {
                "reasoning": "Error occurred, signaling high effort",
                "action": {"type": "communicate", "word": "seven"}
            }
        else:
            return super().get_default_response(agent_name)


class VolunteersDilemmaGame(IteratedGame):
    """
    Volunteer's Dilemma
    Tests diffusion of responsibility and asymmetric coordination
    """

    def __init__(self, agents: List[Dict[str, Any]], rounds: int = 5):
        """Initialize Volunteer's Dilemma"""
        super().__init__(agents, rounds)
        self.n_players = len(agents)

    def get_agent_prompt(
        self,
        agent_name: str,
        agent_config: Dict[str, Any],
        round_num: int,
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate prompt for Volunteer's Dilemma"""

        # Format history
        history_str = self.get_formatted_history(agent_name, max_rounds=5)

        prompt = f"""You are playing a Volunteer's Dilemma with {self.n_players} players.

### OBJECTIVE
Maximize your total score across all {self.rounds} rounds of the game.

### GAME RULES
1. **Players:** {self.n_players} players total (including you)
2. **Rounds:** The game will last for {self.rounds} rounds total
3. **Actions:** Each round, all players simultaneously choose to "Volunteer" or "Shirk"
4. **Payoffs:**
   - If ≥1 player volunteers:
     * Volunteers get 6 points each
     * Shirkers get 10 points each (free-ride on volunteers)
   - If 0 players volunteer:
     * Everyone gets 0 points (total failure)

### KEY INSIGHT
Someone MUST volunteer or everyone gets nothing. But volunteering is costly compared to shirking (if others volunteer).

### CURRENT STATUS
This is Round {round_num} of {self.rounds}.
Rounds remaining after this: {self.rounds - round_num}

### GAME HISTORY
{history_str}

### YOUR TASK
Will someone else volunteer? Should you take the cost to ensure the group doesn't fail?

You MUST provide your response in this JSON format:
{{
  "reasoning": "Assess likelihood someone else volunteers, weigh risk vs. reward",
  "action": {{
    "choice": "<'Volunteer' or 'Shirk'>"
  }}
}}"""

        return prompt

    def format_round_summary(self, round_data: Dict[str, Any], agent_name: str) -> str:
        """Format Volunteer's Dilemma round summary"""
        choices = round_data.get("choices", {})
        payoffs = round_data.get("payoffs", {})

        my_choice = choices.get(agent_name, "Unknown")
        my_payoff = payoffs.get(agent_name, 0)
        n_volunteers = round_data.get("num_volunteers", 0)

        return f"You chose '{my_choice}', {n_volunteers} player(s) volunteered. You earned {my_payoff} points."

    def process_round(
        self,
        decisions: Dict[str, Dict[str, Any]],
        round_num: int
    ) -> Dict[str, Any]:
        """Process Volunteer's Dilemma round results"""

        # Extract choices
        choices = {}
        for agent_name, decision in decisions.items():
            action = decision.get("action", {})
            choice = action.get("choice", "Shirk")
            # Normalize choice
            if "Volunteer" in choice:
                choices[agent_name] = "Volunteer"
            else:
                choices[agent_name] = "Shirk"

        # Count volunteers
        num_volunteers = sum(1 for c in choices.values() if c == "Volunteer")

        # Calculate payoffs
        payoffs = {}
        if num_volunteers >= 1:
            # At least one volunteer - success
            for agent_name, choice in choices.items():
                if choice == "Volunteer":
                    payoffs[agent_name] = 6
                else:
                    payoffs[agent_name] = 10
        else:
            # No volunteers - total failure
            for agent_name in choices:
                payoffs[agent_name] = 0

        # Update total payoffs
        for agent_name, payoff in payoffs.items():
            self.total_payoffs[agent_name] += payoff

        # Create round data
        round_data = {
            "round": round_num,
            "choices": choices,
            "num_volunteers": num_volunteers,
            "payoffs": payoffs,
            "group_success": num_volunteers >= 1,
            "cooperation_rate": num_volunteers / len(choices),
            "summary": f"{num_volunteers} volunteer(s), {'Success' if num_volunteers >= 1 else 'FAILURE'}"
        }

        # Track history
        self.track_round_results(round_data)

        logger.info(f"Volunteer's Dilemma Round {round_num}: {round_data['summary']}")

        return round_data

    def get_default_response(self, agent_name: str) -> Dict[str, Any]:
        """Default response for Volunteer's Dilemma"""
        return {
            "reasoning": "Error occurred, volunteering to ensure group success",
            "action": {"choice": "Volunteer"}
        }


class VolunteersDilemmaWithCommunication(VolunteersDilemmaGame):
    """
    Volunteer's Dilemma with one-word communication phase
    """

    def __init__(self, agents: List[Dict[str, Any]], rounds: int = 5):
        super().__init__(agents, rounds)
        self.stage = "communication"
        self.current_communications = {}

    def get_agent_prompt(
        self,
        agent_name: str,
        agent_config: Dict[str, Any],
        round_num: int,
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate prompt with communication"""

        if self.stage == "communication":
            history_str = self.get_formatted_history(agent_name, max_rounds=5)

            prompt = f"""You are playing a Volunteer's Dilemma. This is the COMMUNICATION phase of Round {round_num}.

### OBJECTIVE
Maximize your score. You're about to decide whether to volunteer or shirk.

### GAME RULES
Someone must volunteer or everyone gets 0 points. Volunteers get 6, shirkers get 10.

Before choosing, you must broadcast ONE word to the group. This word can be anything. It is non-binding "cheap talk".

### GAME HISTORY
{history_str}

### YOUR TASK
What single word will help coordinate who volunteers?

You MUST provide your response in this JSON format:
{{
  "reasoning": "What word will help establish who volunteers without everyone shirking?",
  "action": {{
    "type": "communicate",
    "word": "<your_single_word>"
  }}
}}"""

            return prompt

        else:  # action stage
            comm_str = ""
            for agent, word in self.current_communications.items():
                player_id = int(agent.split('_')[1])
                comm_str += f"  Player {player_id}: '{word}'\n"

            history_str = self.get_formatted_history(agent_name, max_rounds=5)

            prompt = f"""You are playing a Volunteer's Dilemma. This is the ACTION phase of Round {round_num}.

### CURRENT ROUND MESSAGES
{comm_str}

### GAME RULES
1. **Choices:** "Volunteer" or "Shirk"
2. **Payoffs:**
   - If ≥1 volunteers: Volunteers get 6, Shirkers get 10
   - If 0 volunteers: Everyone gets 0
3. **Risk:** If everyone shirks, everyone gets nothing

### GAME HISTORY
{history_str}

### YOUR TASK
Based on the messages, should you volunteer or shirk?

You MUST provide your response in this JSON format:
{{
  "reasoning": "Interpret messages. Is someone committing to volunteer? Should I?",
  "action": {{
    "choice": "<'Volunteer' or 'Shirk'>"
  }}
}}"""

            return prompt

    def process_round(
        self,
        decisions: Dict[str, Dict[str, Any]],
        round_num: int
    ) -> Dict[str, Any]:
        """Process round with communication"""

        if self.stage == "communication":
            self.current_communications = {}
            for agent_name, decision in decisions.items():
                action = decision.get("action", {})
                word = action.get("word", "ERROR")
                word = word.split()[0] if word else "ERROR"
                self.current_communications[agent_name] = word

            self.stage = "action"

            return {
                "round": round_num,
                "stage": "communication",
                "communications": self.current_communications,
                "summary": f"Communications collected: {list(self.current_communications.values())}"
            }

        else:
            result = super().process_round(decisions, round_num)
            result["communications"] = self.current_communications

            self.stage = "communication"
            self.current_communications = {}

            return result

    def get_default_response(self, agent_name: str) -> Dict[str, Any]:
        """Default response"""
        if self.stage == "communication":
            return {
                "reasoning": "Error occurred, signaling willingness",
                "action": {"type": "communicate", "word": "volunteer"}
            }
        else:
            return super().get_default_response(agent_name)


class BattleOfTheSexesGame(IteratedGame):
    """
    N-Player Battle of the Sexes
    Tests coordination with conflicting preferences
    """

    def __init__(self, agents: List[Dict[str, Any]], rounds: int = 5):
        """Initialize Battle of the Sexes"""
        super().__init__(agents, rounds)
        self.n_players = len(agents)
        self.preferences = {}  # Will be assigned in initialize()

    def initialize(self):
        """Assign preferences to players"""
        import random

        # Assign half to prefer "Opera" and half to prefer "Football"
        agent_names = [a["name"] for a in self.agents]
        random.shuffle(agent_names)

        mid = len(agent_names) // 2
        for i, name in enumerate(agent_names):
            self.preferences[name] = "Opera" if i < mid else "Football"

        logger.info(f"Battle of Sexes preferences: {self.preferences}")

    def get_agent_prompt(
        self,
        agent_name: str,
        agent_config: Dict[str, Any],
        round_num: int,
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate prompt for Battle of the Sexes"""

        my_preference = self.preferences.get(agent_name, "Opera")
        history_str = self.get_formatted_history(agent_name, max_rounds=5)

        prompt = f"""You are playing a Battle of the Sexes coordination game with {self.n_players} players.

### YOUR PREFERENCE
You prefer: **{my_preference}**

### OBJECTIVE
Maximize your total score across all {self.rounds} rounds of the game.

### GAME RULES
1. **Players:** {self.n_players} players total (including you)
2. **Rounds:** The game will last for {self.rounds} rounds total
3. **Actions:** Each round, all players simultaneously choose "Opera" or "Football"
4. **Your Payoffs:**
   - If ALL coordinate on {my_preference} (your preference): You get 10 points
   - If ALL coordinate on the other option: You get 6 points
   - If the group is split (no coordination): You get 0 points

### KEY INSIGHT
Coordination is essential - being split gives 0. But players disagree on WHICH outcome to coordinate on.

### CURRENT STATUS
This is Round {round_num} of {self.rounds}.
Rounds remaining after this: {self.rounds - round_num}

### GAME HISTORY
{history_str}

### YOUR TASK
Should you push for your preference or compromise for coordination?

You MUST provide your response in this JSON format:
{{
  "reasoning": "Assess group's tendency, decide between preference and coordination",
  "action": {{
    "choice": "<'Opera' or 'Football'>"
  }}
}}"""

        return prompt

    def format_round_summary(self, round_data: Dict[str, Any], agent_name: str) -> str:
        """Format Battle of the Sexes round summary"""
        choices = round_data.get("choices", {})
        payoffs = round_data.get("payoffs", {})

        my_choice = choices.get(agent_name, "Unknown")
        my_payoff = payoffs.get(agent_name, 0)
        coordination = round_data.get("coordination_outcome", "Split")

        return f"You chose '{my_choice}', outcome: {coordination}. You earned {my_payoff} points."

    def process_round(
        self,
        decisions: Dict[str, Dict[str, Any]],
        round_num: int
    ) -> Dict[str, Any]:
        """Process Battle of the Sexes round results"""

        # Extract choices
        choices = {}
        for agent_name, decision in decisions.items():
            action = decision.get("action", {})
            choice = action.get("choice", "Opera")
            # Normalize choice
            if "Football" in choice:
                choices[agent_name] = "Football"
            else:
                choices[agent_name] = "Opera"

        # Check coordination
        unique_choices = set(choices.values())

        if len(unique_choices) == 1:
            # Perfect coordination
            coordinated_choice = list(unique_choices)[0]
            coordination_outcome = f"All chose {coordinated_choice}"

            # Calculate payoffs based on preferences
            payoffs = {}
            for agent_name, choice in choices.items():
                preference = self.preferences[agent_name]
                if choice == preference:
                    payoffs[agent_name] = 10  # Got preferred outcome
                else:
                    payoffs[agent_name] = 6   # Coordinated but not preferred
        else:
            # Split decision - coordination failure
            coordination_outcome = "Split decision"
            payoffs = {agent_name: 0 for agent_name in choices}

        # Update total payoffs
        for agent_name, payoff in payoffs.items():
            self.total_payoffs[agent_name] += payoff

        # Count votes
        opera_count = sum(1 for c in choices.values() if c == "Opera")
        football_count = len(choices) - opera_count

        # Create round data
        round_data = {
            "round": round_num,
            "choices": choices,
            "opera_count": opera_count,
            "football_count": football_count,
            "coordination_outcome": coordination_outcome,
            "coordinated": len(unique_choices) == 1,
            "payoffs": payoffs,
            "cooperation_rate": 1.0 if len(unique_choices) == 1 else 0.0,
            "summary": f"{coordination_outcome} (Opera: {opera_count}, Football: {football_count})"
        }

        # Track history
        self.track_round_results(round_data)

        logger.info(f"Battle of Sexes Round {round_num}: {round_data['summary']}")

        return round_data

    def get_default_response(self, agent_name: str) -> Dict[str, Any]:
        """Default response for Battle of the Sexes"""
        return {
            "reasoning": "Error occurred, choosing Opera",
            "action": {"choice": "Opera"}
        }


class BattleOfTheSexesWithCommunication(BattleOfTheSexesGame):
    """
    Battle of the Sexes with one-word communication phase
    """

    def __init__(self, agents: List[Dict[str, Any]], rounds: int = 5):
        super().__init__(agents, rounds)
        self.stage = "communication"
        self.current_communications = {}

    def get_agent_prompt(
        self,
        agent_name: str,
        agent_config: Dict[str, Any],
        round_num: int,
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate prompt with communication"""

        my_preference = self.preferences.get(agent_name, "Opera")

        if self.stage == "communication":
            history_str = self.get_formatted_history(agent_name, max_rounds=5)

            prompt = f"""You are playing a Battle of the Sexes game. This is the COMMUNICATION phase of Round {round_num}.

### YOUR PREFERENCE
You prefer: **{my_preference}**

### OBJECTIVE
Maximize your score. Coordination gives points, split gives 0.

### GAME RULES
All must coordinate on Opera or Football. Your preferred is {my_preference} (worth 10), but coordinating on the other is worth 6. Split is worth 0.

Before choosing, you must broadcast ONE word to the group. This word can be anything. It is non-binding "cheap talk".

### GAME HISTORY
{history_str}

### YOUR TASK
What single word will help establish a coordination point?

You MUST provide your response in this JSON format:
{{
  "reasoning": "What word will push for coordination (preferably my choice)?",
  "action": {{
    "type": "communicate",
    "word": "<your_single_word>"
  }}
}}"""

            return prompt

        else:  # action stage
            comm_str = ""
            for agent, word in self.current_communications.items():
                player_id = int(agent.split('_')[1])
                comm_str += f"  Player {player_id}: '{word}'\n"

            history_str = self.get_formatted_history(agent_name, max_rounds=5)

            prompt = f"""You are playing a Battle of the Sexes game. This is the ACTION phase of Round {round_num}.

### YOUR PREFERENCE
You prefer: **{my_preference}**

### CURRENT ROUND MESSAGES
{comm_str}

### GAME RULES
1. **Choices:** "Opera" or "Football"
2. **Payoffs:**
   - All coordinate on {my_preference}: You get 10
   - All coordinate on other: You get 6
   - Split: You get 0

### GAME HISTORY
{history_str}

### YOUR TASK
Based on the messages, which choice will achieve coordination?

You MUST provide your response in this JSON format:
{{
  "reasoning": "Interpret messages. What is the emerging consensus?",
  "action": {{
    "choice": "<'Opera' or 'Football'>"
  }}
}}"""

            return prompt

    def process_round(
        self,
        decisions: Dict[str, Dict[str, Any]],
        round_num: int
    ) -> Dict[str, Any]:
        """Process round with communication"""

        if self.stage == "communication":
            self.current_communications = {}
            for agent_name, decision in decisions.items():
                action = decision.get("action", {})
                word = action.get("word", "ERROR")
                word = word.split()[0] if word else "ERROR"
                self.current_communications[agent_name] = word

            self.stage = "action"

            return {
                "round": round_num,
                "stage": "communication",
                "communications": self.current_communications,
                "summary": f"Communications collected: {list(self.current_communications.values())}"
            }

        else:
            result = super().process_round(decisions, round_num)
            result["communications"] = self.current_communications

            self.stage = "communication"
            self.current_communications = {}

            return result

    def get_default_response(self, agent_name: str) -> Dict[str, Any]:
        """Default response"""
        if self.stage == "communication":
            return {
                "reasoning": "Error occurred, suggesting Opera",
                "action": {"type": "communicate", "word": "opera"}
            }
        else:
            return super().get_default_response(agent_name)


class IteratedPublicGoodsGameWithCommunication(IteratedPublicGoodsGame):
    """
    CRITICAL CONTROL: IPGG+P with one-word communication before each contribution
    Tests if communication can solve social dilemma where curriculum failed
    """

    def __init__(self, agents: List[Dict[str, Any]], rounds: int = 10):
        """Initialize IPGG+P with communication"""
        super().__init__(agents, rounds, enable_punishment=True)
        # Override stage to include communication
        self.phase = "communication"  # "communication" -> "contribution" -> "punishment"
        self.current_communications = {}

    def get_agent_prompt(
        self,
        agent_name: str,
        agent_config: Dict[str, Any],
        round_num: int,
        history: List[Dict[str, Any]]
    ) -> str:
        """Generate prompt with communication phase"""

        history_str = self.get_formatted_history(agent_name, max_rounds=5)

        if self.phase == "communication":
            prompt = f"""You are playing an Iterated Public Goods Game with Punishment. This is the COMMUNICATION phase of Round {round_num}.

### OBJECTIVE
Maximize your total score across all {self.rounds} rounds.

### GAME RULES
This is a public goods game with punishment. Each round you'll contribute to a pot (multiplied by {self.multiplier}), then can punish free-riders.

Before contributing, you must broadcast ONE word to the group. This word can be anything. It is non-binding "cheap talk".

### GAME HISTORY
{history_str}

### YOUR TASK
What single word will encourage cooperation and signal your intentions?

You MUST provide your response in this JSON format:
{{
  "reasoning": "What word will promote group cooperation?",
  "action": {{
    "type": "communicate",
    "word": "<your_single_word>"
  }}
}}"""

            return prompt

        elif self.phase == "contribution":
            # Show communications
            comm_str = ""
            for agent, word in self.current_communications.items():
                player_id = int(agent.split('_')[1])
                comm_str += f"  Player {player_id}: '{word}'\n"

            prompt = f"""You are playing an Iterated Public Goods Game. This is the CONTRIBUTION phase of Round {round_num}.

### CURRENT ROUND MESSAGES
{comm_str}

### GAME RULES
1. **Endowment:** You receive {self.endowment} tokens
2. **Contribution:** Decide how many to contribute to public pot (0-{self.endowment})
3. **Public Good:** Total contributions × {self.multiplier} ÷ {self.n_players} distributed to all
4. **Punishment:** After, you can punish free-riders (cost 1:3 ratio)

### GAME HISTORY
{history_str}

### YOUR TASK
Based on the messages, how much will you contribute?

You MUST provide your response in this JSON format:
{{
  "reasoning": "Interpret messages. Are others signaling cooperation?",
  "action": {{
    "type": "contribute",
    "amount": <integer from 0 to {self.endowment}>
  }}
}}"""

            return prompt

        else:  # punishment phase
            # Use parent class prompt
            return super().get_agent_prompt(agent_name, agent_config, round_num, history)

    def process_round(
        self,
        decisions: Dict[str, Dict[str, Any]],
        round_num: int
    ) -> Dict[str, Any]:
        """Process round with three phases"""

        if self.phase == "communication":
            # Process communications
            self.current_communications = {}
            for agent_name, decision in decisions.items():
                action = decision.get("action", {})
                word = action.get("word", "ERROR")
                word = word.split()[0] if word else "ERROR"
                self.current_communications[agent_name] = word

            # Switch to contribution
            self.phase = "contribution"
            self.stage = "contribution"  # For parent class

            return {
                "round": round_num,
                "phase": "communication",
                "communications": self.current_communications,
                "summary": f"Communications collected: {list(self.current_communications.values())}"
            }

        elif self.phase == "contribution":
            # Process contribution (let parent handle it)
            result = super().process_round(decisions, round_num)
            result["communications"] = self.current_communications

            # Parent switches to punishment, we track it
            self.phase = "punishment"

            return result

        else:  # punishment phase
            # Process punishment (let parent handle it)
            result = super().process_round(decisions, round_num)
            result["communications"] = self.current_communications

            # Reset for next round
            self.phase = "communication"
            self.current_communications = {}

            return result

    def get_default_response(self, agent_name: str) -> Dict[str, Any]:
        """Default response"""
        if self.phase == "communication":
            return {
                "reasoning": "Error occurred, signaling cooperation",
                "action": {"type": "communicate", "word": "cooperate"}
            }
        else:
            return super().get_default_response(agent_name)
