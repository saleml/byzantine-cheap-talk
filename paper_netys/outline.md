# Byzantine Cheap Talk: Adversarial Resilience and Topology Effects in LLM Coordination Games

**Authors:** Hachem Madmoun, Aya Abdelmounaim, Salem Lahlou (MBZUAI)

Multi-agent LLM systems increasingly rely on communication protocols for coordination, yet their robustness under adversarial and structural constraints remains poorly understood. Building on prior work showing that cheap-talk channels enable cooperation in LLM coordination games, we investigate two distinct vulnerability classes in a 4-player Stag Hunt. First, Byzantine agents — who signal cooperation but defect — can eliminate group-level cooperation entirely, with honest agents detecting betrayal within one round but unable to recover coordination due to the game's all-or-nothing payoff structure; probabilistic adversaries partially preserve cooperation but extract the full cooperation surplus. Second, explicitly restricting communication topology collapses cooperation, while identical restrictions applied silently preserve near-perfect cooperation — establishing that the mechanism is agents' meta-reasoning about hidden information, not information loss itself. Finally, we document persistent behavioral archetypes across model families: models that defect rationally upon detecting betrayal vastly outperform models that persistently attempt cooperation under adversarial conditions, a gap that compresses under probabilistic deception. Together, these findings have direct implications for the design of robust networked multi-agent AI systems.

## 1. Introduction

- LLMs as strategic agents in multi-player coordination games
- Cheap talk (pre-play communication) dramatically improves cooperation in Stag Hunt (prior EACL result: 0% → 96.7% heterogeneous, 52.2% → 100.0% coalition; corrected from originally reported 48.3%/50.0% due to cooperation_rate denominator bug)
- But cheap talk assumes honest signaling -- what happens under adversarial conditions?
- Two axes of investigation: (A) Byzantine agents who lie, (B) restricted communication topology
- Key finding preview: a single deterministic liar destroys cooperation entirely; probabilistic deception is far more survivable; topology restrictions collapse coordination even without adversaries

## 2. Related Work

- Game-theoretic analysis of LLM agents (prior EACL paper, other multi-agent LLM work)
- Byzantine fault tolerance in distributed systems
- Cheap talk in game theory (Crawford & Sobel, Farrell & Rabin)
- Communication networks and coordination (Jackson, social network games)
- Adversarial robustness in multi-agent systems

## 3. Experimental Setup

### 3.1 Base Game
- 4-player Stag Hunt with cheap talk communication
- Two-stage: broadcast one word, then choose Hunt Stag / Hunt Hare
- Payoffs: all Stag = 10 each; any Hare = stag hunters get 0, hare hunters get 3
- 5 rounds per trial

### 3.2 Agent Cohort
- 4 heterogeneous LLM agents via DeepInfra API:
  - Agent_1: Mixtral-8x22B-Instruct
  - Agent_2: Qwen2.5-72B-Instruct
  - Agent_3: Llama-3.3-70B-Instruct
  - Agent_4: DeepSeek-V3
- Fixed agent-model assignment across all trials

### 3.3 Experiment A: Byzantine Cheap Talk
- Adversary always broadcasts "stag" but always chooses Hunt Hare
- Conditions: k=0 (baseline), k=1, k=2 adversaries
- Adversary assignment randomized per trial
- 10 trials per condition

### 3.4 Experiment A-soft: Soft Byzantine
- Adversary always broadcasts "stag" but defects with probability p=0.5
- k=1 only, 15 trials
- Tests whether deterministic vs probabilistic deception matters

### 3.5 Experiment B: Communication Topology
- Visibility-restricted communication: broadcast (baseline), ring, star
- Agents told explicitly which messages they can/cannot see
- 10 trials per topology

### 3.6 Experiment B-silent: Silent Topology
- Same visibility filtering as Experiment B but NO explicit cues in prompt
- Agents see fewer messages without being told messages are missing
- Player count removed from prompt to prevent inference
- Tests whether cooperation collapse is from information loss or meta-reasoning about restrictions

## 4. Results

### 4.1 Byzantine Cheap Talk Destroys Cooperation
- Two metrics: "group coop" = fraction of rounds where ALL 4 hunt stag; "honest coop" = fraction of honest agent choices that are Hunt Stag
- Baseline (k=0): 92.0% group / 95.0% honest, avg payoff 9.35/round
- Hard Byzantine k=1: **0.0% group** / 60.0% honest, avg payoff 1.20/round
- Hard Byzantine k=2: **0.0% group** / 37.0% honest, avg payoff 1.89/round
- Group cooperation is always 0% under hard Byzantine (adversary guarantees at least one defection)
- But 60% of honest agents still attempt cooperation under k=1 — they try but are structurally doomed

### 4.2 Model Family Behavioral Archetypes
- Two distinct behavioral archetypes emerge under adversarial conditions:
  - **Fast defectors** (Mixtral, DeepSeek): switch to Hunt Hare after single betrayal, never return
  - **Persistent cooperators** (Qwen, Llama): continue hunting Stag despite repeated exploitation
- k=1: Qwen 6/7 never switched (1/7 mixed), Llama 6/9 never switched (1/9 permanently switched, 2/9 mixed); Mixtral 0/8, DeepSeek 0/6 all permanently switched
- k=2: zero "never switched" across all 20 honest instances; Llama 4/4 mixed (always bounces back), Qwen 2/3 mixed

### 4.3 Payoff Paradox: Cooperation Is Costly
- Hard k=1: FD group earns 2.40/round vs PC group 0.15/round (16x gap)
- Hard k=2: FD group 2.45/round vs PC group 0.86/round (2.9x gap)
- Soft k=1: FD group 4.11/round vs PC group 2.11/round (1.9x gap)
- Star+Byz (hub=adv): FD 3.00 vs PC 2.14 (1.4x gap)
- Star+Byz (hub=hon): FD 2.72 vs PC 1.64 (1.7x gap)
- Fast defection is individually rational but collectively destructive
- Every honest Mixtral/DeepSeek instance under hard k=1 earned exactly 12 pts (0 + 3x4)

### 4.4 Soft Byzantine: Probabilistic Deception Is Survivable
- Group coop: 21.3% (vs 0.0% for hard k=1) — adversary sometimes cooperates, enabling occasional group success
- Honest agent coop: 71.1% (vs 60.0% for hard k=1)
- Round-by-round honest decay: 97.8% -> 75.6% -> 71.1% -> 60.0% -> 51.1%
- Never collapses to zero (unlike hard Byzantine)
- Adversary actual defection rate: 50.7% (confirms p=0.5 design)
- Honest agents average exactly 3.00/round -- cooperation gains perfectly cancelled by intermittent exploitation
- Adversary earns 3.65/round (22% premium)
- Low betrayal (<=2 defections): FD/PC ratio = 1.6x
- High betrayal (>=3 defections): FD/PC ratio = 2.6x

### 4.5 Byzantine × Star Topology
- Star topology with 1 hard Byzantine adversary, two conditions: hub_is_adversary vs hub_is_honest
- 10 trials per condition, 5 rounds each
- Group coop: **0.0%** in both conditions (unanimity impossible with 1 defector)
- Honest coop: **13.3%** (hub=adversary) vs **27.3%** (hub=honest)
- Hub position amplifies adversarial influence: central adversary's "stag" signal reaches all spokes, and spokes cannot cross-validate (they see only the hub). When adversary is a spoke, the honest hub at least sees the betrayal and can signal authentically.
- Avg honest payoff: 2.60 (hub=adv) vs 2.18 (hub=hon) — FD agents earn more in hub=adv because they defect faster
- Round-by-round honest cooperation decay:
  - Hub=adversary: 46.7% → 0.0% → 10.0% → 3.3% → 6.7% (near-instant collapse)
  - Hub=honest:    73.3% → 23.3% → 16.7% → 6.7% → 16.7% (slower decay, higher R1 trust)
- FD/PC ratio: 1.4× (hub=adv) and 1.7× (hub=hon) — dramatically compressed from 16× in broadcast Byzantine
- Star topology suppresses cooperation so heavily for everyone that the archetype difference is muted
- Comparison to broadcast Byzantine (k=1): honest coop 60% there vs 13-27% here — restricted visibility compounds deception
- Key insight: the two vulnerability classes (Byzantine + topology) are **not independent** — they interact multiplicatively. Star topology alone: 27% honest coop. Byzantine alone: 60% honest coop. Combined: 13-27% honest coop.

### 4.6 Communication Topology (Explicit)
- Broadcast: 80.0% group / 87.5% honest, avg payoff 8.38/round
- Ring: **0.0% group** / 33.5% honest, avg payoff 2.00/round
- Star: **0.0% group** / 27.0% honest, avg payoff 2.19/round
- Ring and star never achieve unanimous cooperation (0% group rate) — same structural failure as hard Byzantine, but without any adversary
- Strong model family effect: Llama 72% / Qwen 56% vs Mixtral 2% / DeepSeek 4% in ring

### 4.6 Silent vs Explicit Topology: Meta-Reasoning Is the Mechanism
- Silent topology: same message filtering but NO visibility cues in prompt, player count removed
- **Silent broadcast: 100% group / 100% honest** (vs 80.0% / 87.5% explicit)
- **Silent ring: 100% group / 100% honest** (vs 0.0% / 33.5% explicit) -- the key result
- **Silent star: 100% group / 100% honest** (vs 0.0% / 27.0% explicit) -- v2 leak-fixed results (player-count removed from prompt)
- All three silent conditions achieve perfect cooperation (100%/100%/100%) with zero defections across 600 observations
- Cooperation collapse in explicit topology is NOT caused by information loss -- agents receiving fewer messages without explanation cooperate perfectly
- The mechanism is meta-reasoning: when told "you can only see your neighbors' messages," agents reason about unseen players and defect; when simply shown fewer messages, they take them at face value
- Model family differences vanish in silent condition: all four models cooperate at 100% across all three topologies (vs strong FD/PC split in explicit)
- Implication: the behavioral archetypes are not intrinsic model properties but emerge from strategic framing in the prompt

## 5. Discussion

### 5.1 Implications for Multi-Agent System Design
- Cheap talk is fragile: a single adversary can destroy the coordination benefit entirely
- Probabilistic deception is more dangerous than deterministic in aggregate -- it sustains enough cooperation to keep exploiting persistent cooperators
- Model family differences suggest LLM "personality" is a real design variable

### 5.2 The Cooperation-Robustness Tradeoff
- Persistent cooperators (Qwen, Llama) maintain group welfare potential but are individually exploitable
- Fast defectors (Mixtral, DeepSeek) protect themselves but guarantee collective failure
- No model achieves the ideal: conditional cooperation that detects deception and adapts

### 5.3 Topology: Information vs Meta-Reasoning
- Message restriction alone does NOT destroy coordination -- agents cooperate perfectly when unaware of restrictions
- Explicit framing ("you can only see...") triggers strategic uncertainty reasoning that collapses cooperation
- This is a prompt-sensitivity finding: the same game-theoretic situation produces opposite outcomes depending on how information is presented
- Model family archetypes (FD vs PC) are prompt-elicited, not intrinsic: they disappear entirely in the silent condition
- Design implication: how agents are told about their environment matters as much as the environment itself

### Limitations
1. **Prompt sensitivity**: The silent topology finding (100%/100%/100% vs 87.5%/33.5%/27.0%) demonstrates extreme prompt sensitivity. Small wording changes in the prompt produce radically different outcomes, which complicates claims about "intrinsic" model behavior.
2. **Small sample size**: 10-15 trials per condition. Effect sizes are large and consistent but confidence intervals are wide for fine-grained comparisons (e.g., per-model breakdowns under k=2 have only 3-7 instances per family).
3. **Single game type**: All experiments use 4-player Stag Hunt. Results may not generalize to other coordination games (e.g., Public Goods, Battle of the Sexes) or to games with different payoff structures. The Stag Hunt's binary all-or-nothing payoff structure may amplify adversarial impact.

## 6. Conclusion

- Byzantine cheap talk reveals fundamental fragility of LLM coordination via cheap talk
- Model family behavioral archetypes (fast defector vs persistent cooperator) are stable across conditions
- Probabilistic deception compresses the payoff gap from 16x to 1.9x while maintaining higher exploitation
- Communication topology collapses coordination only when agents are explicitly told about restrictions (meta-reasoning effect, not information loss)
- Design implications: multi-agent LLM systems need mechanisms beyond cheap talk for robust coordination (e.g., commitment devices, reputation systems, verified communication)

## 7. Next Experiments

- **GPT-4o**: Validate FD/PC archetypes on a proprietary model family (Aya running)
- ~~**Byzantine × star topology crossing**~~: ✅ DONE — see Section 4.5. Hub=adversary 13.3% honest coop, hub=honest 27.3%
- **Byzantine in IPGG+P**: Extend Byzantine analysis to the Iterated Public Goods Game with Punishment, where graded payoffs (vs. Stag Hunt's binary structure) may yield different adversarial resilience patterns
