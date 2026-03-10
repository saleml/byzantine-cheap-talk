# Rebuttal Strategy

Goal: Address all reviews with minimal-time fixes, calibrated clarifications, and a small set of targeted new runs. Below is the action plan grouped by theme.

## Reviewer pain points (condensed)
- **R1:** Possible bias in Claude-generated curriculum lessons; asked to test cheap talk in IPGG+P.
- **R2:** Claims about “eliciting cooperation” vs. “improving performance”; limited trials (30/cond); narrow model set; missing human/ablated baselines.
- **R3:** Want “helpful” cooperation metrics; weak justification for curriculum design; small model set; quantify qualitative evidence counts.

## Paper clarifications (no new runs) — **PENDING**
- Reframe objective as *maximizing collective welfare* (payoff) and *cooperation rate*; clarify that curriculum was designed for prosocial incentives, not raw task score (Sec. 1–3, Abstract).
- Add one paragraph explaining curriculum design rationale and acknowledging defection-heavy early stages as a confound; cite Figure/lessons to show bias source (Sec. 4.3 + Limitations).
- Explicitly report lesson content stats: share of lessons advocating defection vs. cooperation (counted from stored lesson files). This directly answers R1/R3.
- Quantify qualitative evidence in §5.1: percent of traces showing (a) learned pessimism, (b) heuristic over-fitting, (c) role-play; add a small table.
- Clarify model choices (four diverse open weights) and discuss generalization limits; commit to adding 1–2 SOTA reasoning models in rebuttal addendum (Sec. 3.2, Limitations).
- Fix “Stag Hunt 10 rounds” mismatch: paper currently states 10 rounds; code ran 3. Update text to 3 and note robustness checks in appendix.
- Replace placeholder claims about completed n=30 Phase 7 with precise counts (Phase 6 = 30/cond; Phase 7 pilot = 1–2/cond) and mark synthetic figures as provisional.

## Easy answers / analyses using existing logs — **PENDING**
- **Curriculum bias check:** Inspect Claude lessons to show several explicitly recommend early defection; quantify proportion. Argues bias drove learned pessimism (R1/R3).
- **Helpful vs. harmful cooperation metric:** Recompute from logs total welfare, variance, and punishment-adjusted welfare; show that communication increases welfare, not just cooperation (R3).
- **Cooperation-rate bug fix:** Use `analysis/fix_cooperation_extraction.py` to recompute true rates from `rounds_data`; regenerate tables/figures with corrected numbers (addresses data-quality concerns).
- **Ablation without lessons:** Re-run existing logs ignoring `system_prompt_suffix` in analysis to show whether lessons, not mere exposure, caused degradation (R2/R3 without new API calls).
- **Quantify evidence counts:** Use `qualitative_sampler` outputs to give counts per failure mode (R3) and cite in rebuttal.

## New experiments (small, targeted)
1) **Cheap talk in IPGG+P (R1):** ✅ Done — 10 trials comm vs plain with original 4 DeepInfra models (mult 1.6); 5+5 trials OpenAI mult 4.0; added strong OpenAI mix control (no comm) + comm variant.
2) **Lesson-quality ablation (R1/R3):** ✅ Done — neutral human-written lesson string, 5 trials full curriculum (DeepInfra), coop 0%, payoff 251.1.
3) **Model generalization (R2/R3):** ✅ Done — strong OpenAI mix (gpt-4.1/4.1-mini/4o/4o-mini) 5 trials SH + IPGG+P; comm variant (4 trials).
4) **Sample-size boost (R2):** ❌ Not done — still 10 trials per condition (orig) / 5 trials (strong models).
5) **Helpful-cooperation metric:** ❌ Not done — welfare/Gini/punishment shares not recomputed; coop extraction bug not fixed in tables.

## Paper fixes informed by code audit — **PENDING**
- Document two-stage processing in IPGG+P and how cooperation is computed; align text with actual 3-round SH and 10-round IPGG settings.
- Note known issue: `cooperation_rate` field in stored results is zero due to extraction bug; state that rebuttal uses recomputed metric from raw `rounds_data`.
- Replace synthetic Phase 7 figures (`generate_phase7_figures_n30.py`) with real data or clearly label as illustrative; move synthetic to appendix if real n=30 not ready.
- Add appendix snippet showing prompts exactly as executed (the code in `src/prompts.py`) to close any ambiguity.

## Deliverables for rebuttal package
- Updated plots/tables: final_stage_comparison, IPGG trajectories, lesson-content histogram, welfare metrics table, qualitative counts table. **PENDING** (only new IPGG comm/no-comm and neutral-lesson figs done).
- Text edits in `paper/main.tex`: Abstract, Intro, Sec. 3.2, Sec. 4–5, Limitations, Appendix counts/figures. **PENDING** (no tex edits yet).
- Brief supplement summarizing new experiments (1–3 above) if completed before deadline. **PENDING** (findings captured in `EXP_REBUTTAL_FINDINGS.md`, no separate supplement yet).

## Timeline (assuming 2–3 days API time)
- Day 1: Fix metrics, regenerate figures, lesson-content stats, qualitative counts; draft text edits.
- Day 2: Run cheap-talk IPGG+P and lesson-ablation micro-runs; start extended trials.
- Day 3: Run new model smoke tests; finalize rebuttal text + updated figures.
