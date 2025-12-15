# Dynamic-Conversation

Emotion flow analysis and (future) strategy simulation for multi-turn dialogues.

## Layout
- `dynamic_conversation/`: importable package  
  - `emotion_map.py`: `EmotionFlowAnalyzer` (classify turns, transition matrices, trajectory scores, plots)  
  - `response_strategy.py`: `ResponseStrategy` enum, ESConv→6-strategy mapping/classifier scaffolding, strategy prompt templates (`STRATEGY_PROMPT_TEMPLATES`), and `build_strategy_prompt` helper  
  - `simulation.py`: `SingleTurnSimulator` and `SimulationConfig` for two-LLM single-turn runs (OpenAI chat, DistilRoBERTa checks)  
  - `load_data.py`: `EsConvData` helper  
  - `__init__.py`: exports public API
- `results/`: generated artifacts (heatmaps, Sankey, CSV scores)
- `notebooks/`: runnable experiments (see map below)
- `pyproject.toml`: packaging metadata/deps

## Install
```bash
conda create -n dynamic-conversation python=3.11
conda activate dynamic-conversation
pip install -e .
```

## Quick use (Python)
```python
from dynamic_conversation import (
    EmotionFlowAnalyzer,
    ResponseStrategy,
    build_strategy_prompt,
    build_esconv_seed_bank,
)
from datasets import load_dataset

ds = load_dataset("thu-coai/esconv")
analyzer = EmotionFlowAnalyzer(use_gpu=False)
analyzer.process_dataset(ds, max_conversations=50)
tm = analyzer.compute_transition_matrix()
trajectory = analyzer.compute_all_trajectory_scores()
analyzer.plot_transition_heatmap()
analyzer.plot_sankey_diagram(conversation_id=0)
analyzer.plot_aggregate_emotion_flow()

prompt = build_strategy_prompt(
    ResponseStrategy.VALIDATE,
    partner_message="I'm overwhelmed about finals.",
    conversation_context=["usr: Finals are stressing me out."],
    style_modifier="be concise, warm tone",
)
print(prompt)

# Single-turn simulation (requires OPENAI_API_KEY)
from dynamic_conversation import SingleTurnSimulator
sim = SingleTurnSimulator(use_gpu=False)  # prompts for key if env var is unset; default model gpt-4o-mini
df = sim.run_batch(
    emotions=["anger", "joy"],
    strategies=[ResponseStrategy.VALIDATE, ResponseStrategy.GUIDE],
    runs_per_pair=1,
    use_llm_seed=False,  # set True to let agent A generate its own seed via LLM
    save_csv=None,
    save_heatmap=None,
)
print(df.head())

# ESConv seeds (optional)
seed_bank = build_esconv_seed_bank(ds, emotions=["anger", "joy"], per_emotion=6, max_chars=200, use_gpu=False)
sim_esconv = SingleTurnSimulator(use_gpu=False, esconv_seeds=seed_bank)
df_esconv = sim_esconv.run_batch(
    emotions=["anger", "joy"],
    strategies=[ResponseStrategy.VALIDATE],
    runs_per_pair=1,
    use_esconv_seed=True,
)
```

## Notes
- GPU is optional; default `use_gpu=True` to auto-detect CUDA if available. Emotion classification batches default to 64 (tuned for an A100); lower if you hit memory limits.
- OpenAI API key required for simulations (`OPENAI_API_KEY` env variable). Default model: `gpt-4o-mini` with temperature fixed at 1.0; tweak via `SimulationConfig`. If you see output-limit errors, increase `max_tokens` (default 400, auto-bumps on limit errors).
- Simulation outputs default to `results/single_turn_simulation.csv` and `results/single_turn_heatmap.png` (configurable in `run_batch`).
- Seeding: by default uses fixed synthetic templates per emotion; set `use_llm_seed=True` to let agent A generate its own seed via the LLM; set `use_esconv_seed=True` with an `esconv_seeds` bank to use ESConv snippets.

## Dev tips
- Add new modules under `dynamic_conversation/` and export public classes/functions in `dynamic_conversation/__init__.py` to keep imports clean.
- Keep generated artifacts out of the package; write to `results/` or other top-level folders (already excluded from packaging).
- Notebooks: `notebooks/esconv_emotion_flow.ipynb` (Phase 1 emotion flow on ESConv), `notebooks/esconv_strategy_exploration.ipynb` (6-strategy classifier/prompt analysis), and `notebooks/single_turn_simulation.ipynb` (Phase 3 single-turn runs). Use Conda env `dynamic-conversation` or hosted Colab with `pip install git+<repo-url>`.
