# TODOs by Phase (notebook-first; ESConv for Phase 1)

## Phase 3 stabilization (completed)
- Goal: harden single-turn simulations so outputs are non-empty and emotionally faithful.
- Status: DONE via FT-04/FT-05 — higher completion caps, empty-response guards/retries, success/failure logging, emotion-preserving prompts with brevity hints, and seed fallback (LLM → synthetic, optional ESConv seeds). Sanity notebook added.

## Phase 4: Multi-turn policy simulation (scope narrowed)
- Goal: short 3–5 turn demo comparing a calming policy vs a provocative policy to see emotion trajectories beyond single-turn limits (fit in ~20–30 minutes).
- Tasks:
  - Implement a multi-turn rollout helper that takes a policy function (emotion → strategy) and logs turn-level emotions. **Done (MultiTurnRollout + policy exports).**
  - Define two deterministic policies: calming (de-escalate high negatives; steady neutrals) and provocative (push movement/volatility). **Done (calming/provocative/always-validate).**
  - Run a small grid (e.g., 7 emotions × 2 policies × 3 convos × 5 turns) using `notebooks/multiturn_policy_comparison.ipynb` (writes to `results/multiturn_runs/`). **Done (runs present).**
  - Summarize final emotion distributions/trajectories; include in report as multi-turn evidence. **Pending write-up in report.**

## Phase 5: RL/policy learning (optional)
- Goal: learn or adaptively choose strategies in an MDP/bandit over emotional states.
- Tasks:
  - Build `rl_policy.py`/bandit environment wrapping simulator with reward = change toward target emotion.
  - Implement baseline learner (tabular Q-learning or contextual bandit) and save curves/policies to `results/`.
  - Pick algorithm/lib (pure Python Q vs. lightweight bandit).

## Evaluation & analysis (report-aligned)
- Goal: meet report requirements with statistical rigor and manual error analysis. Note: original proposal targeted multi-turn/RL; current focus is single-turn eval + small policy demo if time permits.
- Tasks:
  - Run Phase 3 with multiple seeds/runs_per_pair; compute CIs/paired tests on shift tables; add neutral/no-strategy baseline.
  - Primary seeding mode: use LLM seeds (chosen for variety and no dependence on template/ESConv biases); keep synthetic/ESConv as baselines if needed.
  - Capture CIs from full LLM run (notebooks/single_turn_simulation.ipynb section) and include in report tables.
  - Add BERTScore/semantic similarity hooks or small judge/human sampling for strategy adherence/emotional persistence; lightweight safety/toxicity probe.
  - Add second emotion classifier for cross-check or document justification for single-classifier use.
  - Perform manual error analysis on failed/misaligned cases (sampled runs) and summarize patterns. **Done via `notebooks/error_analysis_single_turn.ipynb`; add distilled notes to report.**
  - Log run configs/seeds alongside outputs; add brief reproduction notes in `results/`. **Meta/logs present; capture in report_notes.md.**

## Dataset & examples
- Goal: ensure data section coverage with concrete stats and examples.
- Tasks:
  - Summarize ESConv usage (counts, splits) and any other corpora; show representative input/output snippets and classifier outputs.
  - Document preprocessing choices.

## Runtime integration
- Goal: keep APIs clean for notebooks; minimize CLIs.
- Tasks:
  - Export new helpers via `dynamic_conversation/__init__.py`; keep `python -m dynamic_conversation` as smoke test only if retained.
  - Handle API key config and rate-limit/backoff guidance.

## Visualization defaults
- Goal: consistent, reproducible outputs.
- Tasks:
  - Standardize plot styling/file formats (PNG/HTML) and document runs per figure.

## Notebook hygiene & reporting
- Goal: keep notebooks lean and aligned to report needs.
- Tasks:
  - Ensure notebooks import package logic, avoid heavy outputs in git, and start with a short how-to-run cell (env/kernel, install instructions).
  - Maintain `notebooks/README.md` mapping notebooks to outputs/resources; note GPU/API needs.
  - Draft related work (≥6 citations), contributions, and AI disclosure sections for the report.
