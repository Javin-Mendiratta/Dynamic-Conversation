# Project Handoff Guide

Audience: anyone new picking up the project. This is the source of truth for how to run things and what’s done vs pending.

## What’s built
- Phase 1 (ESConv emotion flow): `EmotionFlowAnalyzer` in `dynamic_conversation/emotion_map.py`; notebook `notebooks/esconv_emotion_flow.ipynb`.
- Phase 2 (strategy mapping + prompts): `ResponseStrategy`, ESConv→6 mapping + classifier scaffolding, and strategy prompt templates in `dynamic_conversation/response_strategy.py`; notebook `notebooks/esconv_strategy_exploration.ipynb`.
- Phase 3 (single-turn simulation scaffold): `SingleTurnSimulator` in `dynamic_conversation/simulation.py`, synthetic emotion seeds, strategy prompts, OpenAI chat calls, DistilRoBERTa checks; notebook `notebooks/single_turn_simulation.ipynb`.

## Environment & installs
- Always use Conda env `dynamic-conversation` (`conda activate dynamic-conversation`).
- Install deps: `pip install -e .` (includes `openai` for simulations).
- Notebooks: kernelspec `dyn-conv` locally, or on hosted Colab run `pip install git+<repo-url>` in a GPU runtime.

## Data & models
- Dataset: ESConv only for Phase 1 (loaded via `datasets` inside notebooks/code).
- Emotion classifier: `j-hartmann/emotion-english-distilroberta-base` (used via HF pipeline).
- Strategy set: fixed 6 strategies defined in `ResponseStrategy`.

## OpenAI usage
- Required for simulations only. Set `OPENAI_API_KEY` in your shell before running (`export OPENAI_API_KEY=sk-...`).
- Default model: `gpt-5-nano`; configurable via `SimulationConfig.model`.
- Defaults: temperature 1.0 for gpt-5 models, max_tokens 220 (sent as `max_completion_tokens` for 5.x/4.1 models), retries with simple backoff.
- If `OPENAI_API_KEY` is not set and `prompt_for_key=True` (default), `SingleTurnSimulator` will prompt for the key interactively; set `prompt_for_key=False` in headless runs.

## How seeding works (Phase 3)
- Agent A seed is randomly sampled from synthetic templates per emotion label (anger/disgust/fear/joy/neutral/sadness/surprise).
- Seed is emotion-checked with DistilRoBERTa; no history across trials (context reset each run).
- Agent B reply uses `build_strategy_prompt` for the chosen strategy; agent A follow-up is prompted separately.
- Adjust `style_modifier` or replace/extend `EMOTION_SEED_TEMPLATES` if you want richer seeds. Alternatively, set `use_llm_seed=True` in `SingleTurnSimulator` to have agent A generate the initial utterance via the LLM instead of sampling the fixed templates (more variety, still checked by the classifier).

## Running things
- Phase 1: open `notebooks/esconv_emotion_flow.ipynb` (adjust `max_conversations`/splits inside).
- Phase 2: open `notebooks/esconv_strategy_exploration.ipynb` (ESConv mapping + classifier/prompt analysis).
- Phase 3: open `notebooks/single_turn_simulation.ipynb`; set `OPENAI_API_KEY`; tweak `emotions`, `strategies`, `runs_per_pair`, and output paths (default CSV/heatmap under `results/`).

## Outputs
- All generated artifacts go to `results/` (plots, CSVs, heatmaps). Default simulation outputs: `results/single_turn_simulation.csv` and `results/single_turn_heatmap.png`; notebook demo uses `results/demo_single_turn.*`.

## What’s next (Phase 4+)
- Phase 4: multi-turn policy simulation (roll out ~24 turns using policies derived from Phase 3 stats; log trajectories/plots).
- Phase 5 (optional): RL policy learning (`rl_policy.py`) over emotional states.
- Evaluation hooks: add similarity/quality checks (e.g., BERTScore) and small manual sampling.
