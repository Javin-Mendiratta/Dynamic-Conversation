# Report Notes and Pointers

## Overall pipeline and structure
- **Classifier backbone:** `EmotionFlowAnalyzer` wrapping `j-hartmann/emotion-english-distilroberta-base` (GPU optional). All seeds/replies/follow-ups are classifier-checked to the 7-label set (anger, disgust, fear, joy, neutral, sadness, surprise).
- **LLM defaults:** `gpt-4o-mini`, `max_tokens` 400 with auto-bump on output-limit errors, temperature 1.0, retries/backoff 3/2s, brevity hints baked into prompts. `OPENAI_API_KEY` required. Seeding priority: ESConv seeds (if provided and enabled) → LLM seeds → synthetic templates.
- **Data layout:** Core code in `dynamic_conversation/`; experiments in `notebooks/`; outputs in `results/`. Notebook map in `notebooks/README.md`.

## Phase 1: ESConv emotion flows
- **Process:** Parse ESConv dialogs, classify each turn with the DistilRoBERTa pipeline, build transition matrices and trajectory scores, visualize (heatmaps, Sankey, aggregate flow).
- **Key outputs:** `results/emotion_transition_heatmap.png`, `emotion_sankey.html`, `aggregate_emotion_flow.png`, `emotion_trajectory_scores.csv`.
- **Notebook:** `notebooks/esconv_emotion_flow.ipynb`.

## Phase 2: Strategy mapping
- **Process:** Define 6-strategy taxonomy (Validate, Explore, Reframe, Affirm, Guide, Normalize), map ESConv labels, build a labeled dataset, scaffold HF fine-tune. Prompts for each strategy via `build_strategy_prompt`.
- **Notebook:** `notebooks/esconv_strategy_exploration.ipynb` (outputs not committed).

## Phase 3: Single-turn simulation
- **Process:** Two-LLM loop (seed → strategy reply → user follow-up), with classifier checks pre/post. Baseline supportive path included. Seeds can be LLM/ESConv/synthetic. Metadata and logs written alongside CSV/heatmap; Wilson CIs computed per (start emotion, strategy, followup emotion).
- **Full run:** `results/single_turn_llm_full_experiment/full_llm_single_turn.*` (CSV, heatmap, `.meta.json`, `.log`, `_ci.csv`). Config: 7 emotions × 6 strategies + baseline, 20 reps, LLM seeding, `gpt-4o-mini`, 800-token cap after auto-bump, zero failures.
- **Findings:** High emotional persistence (~71% same-emotion overall; joy/fear/surprise >80%). Neutral shifts more often (often to joy). Baseline slightly more movement than strategy prompts. CI overlap → single-turn movement is limited.
- **Error analysis:** `notebooks/error_analysis_single_turn.ipynb` (alignment rates, confidence stats, sampled mismatches/neutral drift). Low-confidence/mismatch cases are rare; occasional drift to joy/neutral.
- **Seed comparisons:** `results/single_turn_[synthetic|llm|esconv]_grid/` (LLM seeding chosen for variety and to avoid template/ESConv bias).

## Phase 4: Multi-turn policy simulation
- **Process:** `MultiTurnRollout` runs short conversations (5 turns here). Steps per conversation: seed via LLM (fallback to ESConv/synthetic), classify; loop for 5 turns: policy maps current detected emotion → strategy; agent B replies with strategy prompt + recent history; agent A replies with history-aware prompt allowing emotion change; classify follow-up; update emotion flow. Deterministic policies: calming (Validate/Affirm/Normalize mix), provocative (Reframe/Guide to induce movement), validate baseline (always Validate).
- **Run config:** 7 start emotions × 3 policies × 3 runs each, 5 turns, LLM seeding, classifier on GPU if available.
- **Outputs:** `results/multiturn_runs/*_5turn.csv` (per-turn), `*_5turn.summary.csv` (final emotion, trajectory-to-start), `*_5turn_heatmap.png`, `multiturn_trajectory_summary.csv` (21 rows = start emotion × policy; 7 columns = target emotions; mean ± 95% CI; row-wise max highlighted in notebook).
- **Findings:** Finals cluster to joy/neutral across policies (calming 17/21 joy; provocative 19/21 joy; validate 19/21 joy). Trajectory scores show pull toward start emotion and joy; differences between policies are directional with wide CIs (n=3 per cell). Suggests LLM “supportive” bias and short horizon drive joy convergence; more runs/longer turns would tighten CIs.

## Key configurations (repro)
- LLM: `gpt-4o-mini`, `max_tokens` 400 (auto-bump), temperature 1.0, retries/backoff 3/2s, brevity hint included. Set `OPENAI_API_KEY`.
- Classifier: `j-hartmann/emotion-english-distilroberta-base`; GPU optional.
- Env: `environment-dynconv.yml` (Python 3.11, torch 2.1.*, CUDA 11.8). Install via `conda env create -f environment-dynconv.yml` (ensure writable conda/pkgs), then `pip install -e .`.
- Notebooks: use `notebooks/README.md` for mapping; keep outputs in `results/`.

## Result pointers (by phase)
- Phase 1: `emotion_transition_heatmap.png`, `emotion_trajectory_scores.csv`, `emotion_sankey.html`, `aggregate_emotion_flow.png`.
- Phase 3: `single_turn_llm_full_experiment/full_llm_single_turn.csv` (+ `.meta.json`, `.log`, `_heatmap.png`, `_ci.csv`); seed grids under `single_turn_[synthetic|llm|esconv]_grid/`.
- Phase 4: `multiturn_runs/*_5turn.csv`, `*_5turn.summary.csv`, `*_5turn_heatmap.png`, `multiturn_trajectory_summary.csv` (mean ± 95% CI table).

## Reporting guidance
- Emphasize: single-turn movement is limited; multi-turn needed for larger shifts. Cite Phase 3 CI table and Phase 4 trajectory/heatmaps. Note joy/neutral convergence and LLM supportive bias. Limitations: single classifier, small n for multi-turn (wide CIs), short horizon, and potential seed/model bias. Recommend future work: more runs/turns, stricter policy prompts, second classifier or human/LLM judge checks for adherence/safety.
