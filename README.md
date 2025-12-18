# Dynamic Conversation (CS 685 Final Project)

Pipeline for analyzing and steering emotion flow in multi-turn LLM dialogues. The project covers four phases: 1) ESConv emotion dynamics exploration, 2) normalizing response strategies and building prompts/classifier scaffolding, 3) simulating single-turn emotion shifts with two LLM agents, and 4) simulating multi-turn dialogue with three deterministic strategy policies.

## Repo layout
- `dynamic_conversation/`: importable package  
  - `emotion_map.py`: `EmotionFlowAnalyzer` for DistilRoBERTa emotion classification, transition matrices, trajectory scores, heatmap/Sankey plots.  
  - `response_strategy.py`: `ResponseStrategy` enum and 6-strategy ESConv mapping, `StrategyDatasetBuilder`, prompt templates (`STRATEGY_PROMPT_TEMPLATES`), and `build_strategy_prompt`.  
  - `simulation.py`: `SingleTurnSimulator` + `SimulationConfig` for two-LLM single-turn runs, `MultiTurnRollout` for policy-driven short conversations, seed builders (`EMOTION_SEED_TEMPLATES`, `build_esconv_seed_bank`), CI helper, and policy shortcuts (`calming_policy`, `provocative_policy`, `always_validate_policy`).  
  - `load_data.py`: `EsConvData` loader for HF `thu-coai/esconv`.  
  - `__init__.py`: exports the public API for notebook imports.
- `notebooks/`: experiments mapped in `notebooks/README.md` (Phase 1 emotion flow, Phase 2 strategy mapping, Phase 3 single-turn sims + seed comparisons + error analysis, Phase 4 multi-turn policy comparison).
- `results/`: generated artifacts  
  - `phase1_ESConv_exploration/`: `emotion_transition_heatmap.png`, `emotion_sankey.html`, `aggregate_emotion_flow.png`, `emotion_trajectory_scores.csv`, `6strategy_*` plots/tables.  
  - `single_turn_prompting_exploration/`: seed-mode grids (`single_turn_[synthetic|llm|esconv]_grid/` with CSV, heatmap, meta, log) used to pick LLM seeding.  
  - `single_turn_llm_full_experiment/`: full Phase 3 run (CSV + `_ci.csv`, heatmap, `.meta.json`, `.log`).  
  - `multiturn_runs/`: Phase 4 demo outputs (`*_5turn.csv`, `*_5turn.summary.csv`, `*_5turn_heatmap.png`, `multiturn_trajectory_summary.csv`).  
- `pyproject.toml`: packaging metadata.  
- `report_notes.md`: distilled findings and configuration defaults (useful for the write-up).

## Project goals (proposal → implementation)
- Baseline emotion flow in ESConv without relying on labels; produce transition/trajectory visuals for the report.  
- Normalize ESConv strategies into 6 buckets (validate, explore, reframe, affirm, guide, normalize), supply prompts and a classifier scaffold for strategy-aware responses.  
- Measure emotion shifts for (emotion, strategy) pairs via two-agent single-turn simulations, with CIs/meta/logging for reproducibility.  
- Prototype short multi-turn policies (calming vs. provocative vs. validate) as a stepping stone toward policy learning/bandits.

## Setup (local)
All dependencies are in `pyproject.toml`. GPU is optional; set `use_gpu=False` in analyzers/simulators if CPU-only.

## Colab workflow
1) New notebook → `!pip install git+https://github.com/<your-username>/Dynamic-Conversation.git` (or upload ZIP).  
2) Set `OPENAI_API_KEY` from Colab user data (saved key name "OpenAI"):  
   ```python
   from google.colab import userdata
   import os
   os.environ["OPENAI_API_KEY"] = userdata.get("OpenAI")
   ```  
   You can also set `os.environ["OPENAI_API_KEY"] = "sk-..."` directly.  
3) (Optional) `!pip install "torch==2.1.*" -f https://download.pytorch.org/whl/torch_stable.html` and enable GPU runtime for faster classification.  
4) Import from the package in each notebook (see below). Outputs write under `results/` by default.

## Package usage (notebook or scripts)
```python
from datasets import load_dataset
from dynamic_conversation import (
    EmotionFlowAnalyzer,
    ResponseStrategy,
    build_strategy_prompt,
    build_esconv_seed_bank,
    SingleTurnSimulator,
    MultiTurnRollout,
    calming_policy,
)

# Phase 1: ESConv emotion flows
ds = load_dataset("thu-coai/esconv")
analyzer = EmotionFlowAnalyzer(use_gpu=False)
analyzer.process_dataset(ds, max_conversations=50)
analyzer.plot_transition_heatmap(save_path="results/phase1_ESConv_exploration/emotion_transition_heatmap.png")

# Strategy prompt helper
prompt = build_strategy_prompt(
    ResponseStrategy.VALIDATE,
    partner_message="I'm overwhelmed about finals.",
    conversation_context=["usr: Finals are stressing me out."],
    style_modifier="concise, warm tone",
)

# Phase 3: single-turn simulation (requires OPENAI_API_KEY)
sim = SingleTurnSimulator(use_gpu=False)
df = sim.run_batch(
    emotions=["anger", "joy"],
    strategies=[ResponseStrategy.VALIDATE, ResponseStrategy.GUIDE],
    runs_per_pair=2,
    use_llm_seed=True,
    save_csv="results/demo_single_turn.csv",
    save_heatmap="results/demo_single_turn_heatmap.png",
)

# Phase 4: multi-turn policies (5-turn calming policy example)
rollout = MultiTurnRollout(use_gpu=False)
mt_df = rollout.run_policy_grid(
    start_emotions=["sadness"],
    policy=calming_policy,
    turns=5,
    runs_per_emotion=1,
    save_dir="results/multiturn_runs",
)
```

## Notebook workflow (all phases)
- Install the package (`pip install -e .` locally or `pip install git+...` on Colab) before running any notebook.  
- Set `OPENAI_API_KEY` in the environment; LLM calls will prompt for it if missing, but Colab secrets are recommended.  
- Import from `dynamic_conversation` instead of copying logic; notebooks mirror the package functions/classes.  
- Outputs write to `results/` with accompanying `.meta.json` and `.log` where applicable; adjust paths in notebook parameters if you want separate runs.  
- Notebook map: see `notebooks/README.md` for per-notebook details and expected outputs.

## Seeding modes (used in sims)
- Synthetic seeds: fixed templates per emotion (`EMOTION_SEED_TEMPLATES`), no LLM needed.  
- LLM seeds: agent A generates the opener via the chat model for variety (default in full runs).  
- ESConv seeds: first-turn ESConv openers classifier-filtered into the 7-label set (`build_esconv_seed_bank`).  
Priority inside `SingleTurnSimulator`/`MultiTurnRollout`: ESConv (if provided and `use_esconv_seed=True`) → LLM (`use_llm_seed=True`) → synthetic templates. All seeds are re-classified before use to ensure label consistency.

## Results map (what to look at)
- `results/phase1_ESConv_exploration/emotion_transition_heatmap.png`: ESConv emotion→emotion probabilities; pair with `emotion_trajectory_scores.csv` and `emotion_sankey.html` for flow/trajectory views.  
- `results/single_turn_prompting_exploration/`: synthetic vs LLM vs ESConv seeding grids (CSV/heatmap/meta/log) used to select LLM seeding.  
- `results/single_turn_llm_full_experiment/full_llm_single_turn.csv`: main Phase 3 grid (7 emotions × 6 strategies + baseline, 20 runs each), `_ci.csv` for Wilson CIs, `_heatmap.png` for shift visualization, `.meta.json` for config, `.log` for failures (empty when clean).  
- `results/multiturn_runs/`: Phase 4 demo; `*_5turn.csv` per policy (turn-level emotions/strategies), `*_5turn.summary.csv` for final emotions/trajectory scores, `*_5turn_heatmap.png`, and `multiturn_trajectory_summary.csv` (mean ± 95% CI table).