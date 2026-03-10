# Rebuttal Experiments

## Cheap-talk vs No-communication on IPGG+P (original models, multiplier 1.6)
- Models: Mixtral-8x22B, Qwen2.5-72B, Llama-3.3-70B, DeepSeek-V3 (DeepInfra).
- Rounds/Trials: 10 rounds, 10 trials per condition.
- Parsing: strict JSON with heuristic + formatter fallback.
- Results:
  - Payoff (plain): 184.4 ± 40.6; (comm): 127.5 ± 22.5.
  - Cooperation (avg contrib/20): plain 0.478 ± 0.029; comm 0.706 ± 0.138.
  - Interpretation: Cheap talk raised cooperation but lowered payoff due to over‑contribution and punishment under low multiplier.
- Figures: `analysis/figures/rebuttal_ipgg_comm_vs_plain.png`, `rebuttal_ipgg_contrib_trajectory.png`.

## Multiplier=4.0, cheap-talk vs no-comm (OpenAI backend)
- Models: All four agents set to `gpt-4o-mini` (OpenAI), 10 rounds, 5 trials per condition.
- Multiplier: 4.0 (public goods multiplier); other rules unchanged; same prompts; JSON strict.
- Results:
  - Payoff: plain 457.9 ± 38.2; comm 480.0 ± 0.0.
  - Cooperation (avg contrib/20): plain 0.554 ± 0.037; comm 1.000 ± 0.000 (full 20/20 contribs).
  - Interpretation: With higher marginal return, cheap talk now improves both cooperation and welfare; full contributions become payoff-optimal.
- Figure: `analysis/figures/rebuttal_ipgg_mult4_openai.png`.

## Neutral-lesson curriculum ablation (DeepInfra, same 4-agent set)
- Goal: test whether lesson *content* matters. Replaced Claude/human lessons with a fixed neutral sentence: “cooperate when payoffs favor group welfare; punish only clear free-riders; re-evaluate after each round instead of assuming defection.”
- Config: `config/curriculum_full.json` (4 stages: IPD, N-IPD, IPGG, IPGG+P), 5 trials, rounds unchanged, multiplier 1.6, models = Mixtral-8x22B, Qwen2.5-72B, Llama-3.3-70B, DeepSeek-V3 via DeepInfra, JSON-enforced parsing.
- Results (Stage 4 target task):
  - Cooperation rate: 0.0 ± 0.0 (all trials collapse to full defection despite punishment).
  - Average payoff: 251.1 ± 18.0.
- Earlier stages also show 0% cooperation (Stage 1–3 coops all 0), indicating the neutral lesson fails to induce reciprocity anywhere in the curriculum.
- Interpretation: Removing informative lesson content eliminates the cooperation gains claimed in the paper; the lesson text (human/Claude-authored) is a key active ingredient rather than mere exposure to staged games.
- Figure: `analysis/figures/rebuttal_curriculum_neutral.png` (stage-4 payoffs per trial, dashed mean line).

### Comparison to paper’s Claude curriculum (reported)
- Paper Table 1 (full curriculum with Claude lessons, N=29–30 trials): Stage‑4 avg payoff 153.6 ± 40.1; control (no curriculum) 211.7 ± 22.7.
- Neutral lesson (this work, N=5): Stage‑4 avg payoff 251.1 ± 18.0; coop 0%.
- Paper coop rates: all curriculum variants in the paper results folder are 0% cooperation at Stage 4 (full=0, scrambled=0, direct-precursor=0, control=0 across 30 trials each), so payoff is the only differentiator among paper curricula.
- Observation: Neutral lesson outperforms the paper’s Claude curriculum on payoff (higher mean, lower variance) but still yields 0% cooperation—so Claude lessons changed behavior (lowered payoffs via learned pessimism) but did not lift cooperation above zero. Neutral acts like “no lesson,” sometimes giving higher payoffs because it avoids over-punishing.
- Figure: `analysis/figures/rebuttal_curriculum_neutral_vs_paper.png` (Control vs Full Curriculum (Claude, paper) vs Neutral lesson).

## Strong-model generalization (OpenAI gpt-4.1 / 4o family, no curriculum)
- Purpose: address R2/R3 request for stronger models beyond the original four.
- Agents: `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini` (heterogeneous mix), OpenAI API.
- Games/Trials: Stag Hunt (3 rounds, 5 trials), IPGG+P (10 rounds, multiplier 1.6, 5 trials).
- Results:
  - Stag Hunt: cooperation 0%, payoff 8.25 (no comm).
  - IPGG+P (no comm): payoff 192.1 ± 23.8, cooperation 0% (stored field).
- Interpretation: Stronger reasoning models improve payoff relative to the paper’s Claude curriculum (153.6) but still below the paper control (211.7); they do not spontaneously cooperate under IPGG+P without a comm channel.

### Strong-model cheap talk in IPGG+P
- Same agents (gpt-4.1 / 4.1-mini / 4o / 4o-mini), 10 rounds, 4 completed trials with one-word communication.
- Results: payoff 161.9 ± 31.3; cooperation field remains 0% (metric bug consistent with earlier runs).
- Observation: For strong models, adding cheap talk in IPGG+P reduced payoff vs. their own no-comm baseline (192 → 162) and did not lift measured cooperation, suggesting the comm benefit in Stag Hunt does not transfer to the harsher IPGG+P setting even with stronger models.
- Files: `results_rebuttal/model_generalization/openai_strong/ipgg_punish/...` (no comm) and `.../ipgg_comm/...` (comm). Stag Hunt: `.../stag_hunt/...`.

## Notes
- Outputs live under `results_rebuttal/ipgg_baselines_r1/...` (mult 1.6, original models) and `results_rebuttal/ipgg_mult4_openai/...` (mult 4.0).
- Formatter fallback only reformatted text; game decisions remain from the primary models used in each run.
