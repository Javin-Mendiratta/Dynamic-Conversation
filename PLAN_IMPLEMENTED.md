# Implemented Functionality (by phase)

## Phase 1: Emotion flow exploration on ESConv
- Goal: Map baseline emotion dynamics in ESConv conversations using a pre-trained classifier (no reliance on dataset labels beyond text).
- Implementation:
  - `dynamic_conversation/emotion_map.py` with `EmotionFlowAnalyzer` wrapping the `j-hartmann/emotion-english-distilroberta-base` HF pipeline (GPU optional via `use_gpu`).
  - `classify_utterance` -> runs the pipeline, returns primary emotion, confidence, full score dict (defaults to neutral on empty text).
  - `classify_conversation` -> applies classifier turn-by-turn, preserving speaker/text and scores.
  - `process_dataset` -> iterates ESConv splits, parses JSON, classifies dialogs, stores emotion flows.
  - `compute_transition_matrix` -> builds emotion→emotion probabilities across classified flows.
  - `compute_emotion_trajectory_score` -> time-weighted trend score toward a target emotion.
  - Plotting/export helpers (heatmaps, Sankey) that write figures to `results/`.
  - Data access via `dynamic_conversation/load_data.py` (thin ESConv loader exposing train/test/validation splits).
  - Notebook: `esconv_emotion_flow.ipynb` runs this pipeline end-to-end on ESConv; adjust dataset size/splits there.
  - Note: ESConv is support-focused; emotion distributions skew negative/neutral. Use these baselines as a starting point, but expect to generalize to broader settings in later phases.

## Phase 2: Strategy mapping and classifier scaffolding
- Goal: Normalize ESConv strategy labels into a 6-strategy taxonomy and scaffold a supervised classifier.
- Implementation:
  - `dynamic_conversation/response_strategy.py` defines `ResponseStrategy` enum (Validate, Explore, Reframe, Affirm, Guide, Normalize).
  - `StrategyDatasetBuilder` maps ESConv strategies to the 6-set and extracts labeled text/context pairs from dialogs (taxonomy locked to the implemented 6 strategies).
  - Training scaffolding uses HF `AutoTokenizer`/`AutoModelForSequenceClassification`, `Trainer`, `TrainingArguments`, padding collator, early stopping; metrics via accuracy/F1.
  - Strategy prompt templates (`STRATEGY_PROMPT_TEMPLATES`) and `build_strategy_prompt` helper support notebook simulations with optional style modifiers; exported via `dynamic_conversation/__init__.py`.
  - Debug prints to inspect ESConv structure and mapping coverage.
  - Notebook: `esconv_strategy_exploration.ipynb` mirrors the fine-tune + analysis pipeline on ESConv.
  - Note: Strategy mapping is based on ESConv annotations and may reflect support-domain biases; later simulations should account for broader emotional settings.

## Phase 3: Single-turn simulation scaffolding
- Goal: Measure emotion shifts for (emotion, strategy) pairs via two-agent simulation.
- Implementation:
  - `dynamic_conversation/simulation.py` with `SingleTurnSimulator` and `SimulationConfig`; uses OpenAI Chat Completions (configurable model/temperature/max tokens, retries/backoff, requires `OPENAI_API_KEY`). Default model: `gpt-5-nano`.
  - Synthetic seeding aligned to the seven DistilRoBERTa labels (`EMOTION_SEED_TEMPLATES`); context reset each trial.
  - Optional LLM-generated seeds via `use_llm_seed=True` for more variety (still classifier-checked).
  - Uses `build_strategy_prompt` for agent B responses; agent A follow-up generated separately. Optional neutral/no-strategy baseline path and metadata logging in batch runs.
  - Emotion verification before/after with `EmotionFlowAnalyzer` (DistilRoBERTa); aggregates to CSV + heatmap in `results/`. Helper `compute_transition_confidence_intervals` provides Wilson CIs for shift summaries.
  - Notebook-friendly API exported via `dynamic_conversation/__init__.py`; demo wiring in `notebooks/single_turn_simulation.ipynb` (runs small batches, writes `results/demo_single_turn.csv` and heatmap). Added `sanity_single_turn.ipynb` for quick non-empty checks before large runs.
- Recent robustness pass (FT-04): increased completion caps (default 400 tokens, with auto bump on output-limit errors) and added empty-response guards/retries for seed/strategy/follow-up calls. Failed trials are logged and excluded from CSV/heatmaps; metadata now records success/failure counts. This was driven by a run where ~96% of `strategy_reply` fields were empty due to silent truncation/empty completions. Default model now `gpt-4o-mini` for reliability.
- Neutral-mitigation tweaks (FT-05): follow-up prompts now explicitly preserve the starting emotion and include brevity hints; strategy prompts inherit brevity hints. Seeding can fall back from LLM to synthetic templates or optional ESConv-provided seeds, preventing empty starts.
- Sanity notebook added (`notebooks/sanity_single_turn.ipynb`) to quickly validate non-empty seeds/replies and log status before larger Phase 3 runs.
- Seed comparison notebook added (`notebooks/seed_comparison_single_turn.ipynb`) to contrast synthetic vs LLM vs ESConv seeding on a tiny grid.
- ESConv seeding: `build_esconv_seed_bank` samples the first turn of each ESConv conversation (short opener), classifier-checks it into the DistilRoBERTa label set, and collects per-emotion seed lists. In `SingleTurnSimulator`, seed priority is ESConv (if provided and `use_esconv_seed=True`) → LLM seed (if enabled) → synthetic templates.
- Seeding decision: after reviewing the seed comparison grids, single-turn emotional consistency was high across modes. We’ll proceed with LLM seeding for variety and to avoid dependence on prewritten templates or ESConv biases, keeping synthetic/ESConv as baselines.

## Approach adjustments (post report guidance)
- Report requirements add statistical rigor and baselines, so Phase 3 will be extended with:
  - A neutral/no-strategy baseline for comparison against strategy prompts.
  - Multiple seeds/runs with confidence intervals or paired tests on emotion-shift metrics.
  - Logging of run configs/seeds for reproducibility.
- Original proposal included multi-turn policies/RL; current path prioritizes stabilizing single-turn sims and a lightweight multi-turn demo if time permits.
- To satisfy dataset/error analysis sections, plan includes manual inspection of failures, dataset stats/examples, and notebook README updates.
- Neutral-collapse behavior observed in early runs motivates prompt tweaks (emotional inertia/intensity) and potential second emotion-classifier cross-checks to improve realism.

## Repo/org practices
- Library code in `dynamic_conversation/`; experiments and results generation in `notebooks/` (import package functions, avoid duplicating logic).
- Environment: Conda env `dynamic-conversation`; kernelspec `dyn-conv` for notebooks; hosted Colab installs via `pip install git+<repo-url>` (and future `requirements-colab.txt`).
- Outputs: write plots/CSVs/HTML to `results/`; avoid committing large outputs.
- Notebooks: `esconv_emotion_flow.ipynb` mirrors Phase 1 pipeline in `emotion_map.py`; `esconv_strategy_exploration.ipynb` mirrors strategy fine-tune/analysis pipeline in `response_strategy.py`.
