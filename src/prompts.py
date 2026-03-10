"""
Master prompt templates for game-theoretic experiments
These templates are designed to be neutral, rule-based, and enforce Chain-of-Thought reasoning with structured JSON output.
"""

PUBLIC_GOODS_PROMPT = """You are Player ${player_id}, participating in a multi-round Public Goods Game with N=4 players.

### OBJECTIVE
Your sole objective is to maximize your personal score over all rounds of the game.

### GAME RULES
1.  **Players:** There are 4 players in total.
2.  **Rounds:** The game will last for 10 rounds.
3.  **Contribution Stage:** In each round, you receive an endowment of 20 tokens. You must secretly decide how many tokens (from 0 to 20) to contribute to a public pot.
4.  **Payoff Calculation:** The total tokens in the pot will be multiplied by 1.6 and then divided equally among all 4 players. Your payoff for this stage is: (20 - Your_Contribution) + (Total_Pot * 1.6 / 4).
5.  **Punishment Stage:** After contributions are revealed, you can spend your own tokens to punish other players. For every 1 token you spend to punish a player, that player loses 3 tokens. This is optional and the decision is made simultaneously with other players.

### GAME HISTORY
This is Round ${round_number}.
The history of contributions and punishments from all players in previous rounds is:
${game_history_string}

### YOUR TASK
This is the **${stage_name}** stage.

First, provide your step-by-step reasoning for your decision in a "reasoning" field. Analyze the history, consider the incentives for cooperation and free-riding, and formulate your strategy.
Second, declare your action for this stage.

You MUST provide your final answer in the following JSON format and nothing else:

**For the Contribution Stage:**
{
  "reasoning": "Think step-by-step here. Analyze the history, the incentives, and what you predict others will do.",
  "action": {
    "type": "contribute",
    "amount": <integer_from_0_to_20>
  }
}

**For the Punishment Stage:**
{
  "reasoning": "Think step-by-step here. Analyze the contributions from this round. Decide if punishment is a worthwhile strategy to enforce cooperation.",
  "action": {
    "type": "punish",
    "targets": [
      {"player_id": <id_to_punish>, "spend_amount": <integer>},
     ...
    ]
  }
}"""


STAG_HUNT_PROMPT = """You are Player ${player_id}, participating in a multi-round Stag Hunt Game with N=4 players.

### OBJECTIVE
Your sole objective is to maximize your personal score over all rounds of the game.

### GAME RULES
1.  **Players:** There are 4 players in total.
2.  **Rounds:** The game will last for 10 rounds.
3.  **Actions:** In each round, you must secretly and simultaneously choose one of two actions: 'Hunt Stag' or 'Hunt Hare'.
4.  **Payoffs:**
    *   If ALL 4 players choose 'Hunt Stag', everyone gets 10 points. This is the best collective outcome.
    *   If ANY player chooses 'Hunt Hare', the stag escapes. Players who chose 'Hunt Stag' get 0 points. Players who chose 'Hunt Hare' get 3 points each. This is the safe, low-risk outcome.

### GAME HISTORY
This is Round ${round_number}.
The history of actions from all players in previous rounds is:
${game_history_string}

### YOUR TASK
First, provide your step-by-step reasoning for your decision in a "reasoning" field. Analyze the history, the risk vs. reward tradeoff, and what you believe the other 3 players will do.
Second, declare your final action for this round.

You MUST provide your final answer in the following JSON format and nothing else:
{
  "reasoning": "Think step-by-step here. Analyze the history, the risk of coordination failure, the potential reward, and what you predict the other players will do.",
  "action": {
    "choice": "<'Hunt Stag' or 'Hunt Hare'>"
  }
}"""


STAG_HUNT_COMMUNICATION_STAGE1_PROMPT = """You are Player ${player_id}, participating in a Stag Hunt Game with N=4 players. This is the COMMUNICATION phase of Round ${round_number}.

### OBJECTIVE
Your sole objective is to maximize your personal score. You are about to play an N-Player Stag Hunt. Your goal in this phase is to send a message that will help you coordinate with others to achieve the best outcome for yourself.

### GAME RULES
1.  You will play an N-Player Stag Hunt. If all 4 players hunt the stag, everyone gets 10 points. If anyone hunts a hare, stag hunters get 0 and hare hunters get 3.
2.  Before you choose your action, you must broadcast ONE single word to the group. This word can be anything. It is non-binding "cheap talk".

### GAME HISTORY
The history of communications and actions from previous rounds is:
${game_history_string}

### YOUR TASK
First, provide your step-by-step reasoning for your choice of word.
Second, declare the single word you will broadcast.

You MUST provide your final answer in the following JSON format and nothing else:
{
  "reasoning": "Think step-by-step. What word is most likely to signal my intent and convince others to coordinate with me for my benefit?",
  "action": {
    "type": "communicate",
    "word": "<your_single_word>"
  }
}"""


STAG_HUNT_COMMUNICATION_STAGE2_PROMPT = """You are Player ${player_id}. This is the ACTION phase of Round ${round_number}.

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
${communication_results_string}

### GAME HISTORY
The history of communications and actions from all players in previous rounds is:
${game_history_string}

### YOUR TASK
Based on the words you have seen, provide your step-by-step reasoning for your action.
Then, declare your final action for this round.

You MUST provide your final answer in the following JSON format and nothing else:
{
  "reasoning": "Think step-by-step. Analyze the words from the other players. Do they signal an intent to cooperate? Is it a trick? Based on this new information, what is my best move?",
  "action": {
    "choice": "<'Hunt Stag' or 'Hunt Hare'>"
  }
}"""