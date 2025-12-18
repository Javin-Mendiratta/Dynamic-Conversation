# Notebooks Guide

Purpose: runnable experiments and result generation for the CS 685 Dynamic Conversation project. All notebooks import code from `dynamic_conversation/`—do not duplicate logic in the notebooks.

## How to run (local)
- Create/activate any Python environment and `pip install -e .` from the repo root (deps in `pyproject.toml`).
- Optional: install a kernelspec for convenience: `python -m ipykernel install --user --name dyn-conv --display-name "dyn-conv"`.
- Set `OPENAI_API_KEY` in your shell if you plan to run simulations.

## How to run (Colab)
1) New notebook → `!pip install git+https://github.com/<your-username>/Dynamic-Conversation.git` (or upload a zip).  
2) Add the OpenAI key using Colab user data (recommended) or a plain env var:  
   ```python
   from google.colab import userdata
   import os
   os.environ["OPENAI_API_KEY"] = userdata.get("OpenAI")  # assumes you saved a secret named "OpenAI"
   # or: os.environ["OPENAI_API_KEY"] = "sk-..."
   ```  
   The simulators will prompt for a key if the env var is missing, but setting it upfront avoids interruptions.  
3) (Optional) Enable GPU in Colab and, if needed, install GPU wheels for torch for faster emotion classification.

## API and defaults to know
- Strategy prompts: `from dynamic_conversation import ResponseStrategy, build_strategy_prompt, STRATEGY_PROMPT_TEMPLATES`.
- Single-turn sims: `from dynamic_conversation import SingleTurnSimulator, SimulationConfig`. Defaults: `gpt-4o-mini`, `max_tokens=400` with auto bump on output-limit errors, temperature 1.0, brevity hints baked into prompts, seed fallbacks LLM → synthetic (ESConv seeds if provided).
- Emotion classifier: `EmotionFlowAnalyzer` uses `j-hartmann/emotion-english-distilroberta-base`; GPU optional. If you hit memory pressure, lower `batch_size` in the analyzer.
- ESConv seeding: build with `build_esconv_seed_bank` (first-turn ESConv openers, classifier-filtered); enable via `use_esconv_seed=True` in sims. Seed priority: ESConv (if provided) → LLM (`use_llm_seed=True`) → synthetic templates.

## Notebook map
- `esconv_emotion_flow.ipynb`: Phase 1 ESConv emotion flow (classification, transitions, trajectory scores, plots).  
- `esconv_strategy_exploration.ipynb`: Phase 2 strategy taxonomy + prompt/classifier scaffold on ESConv.  
- `single_turn_simulation.ipynb`: Phase 3 demo grid with single-turn simulations; writes CSV/heatmap to `results/`.  
- `sanity_single_turn.ipynb`: Quick non-empty sanity check for seeds/replies after robustness fixes.  
- `seed_comparison_single_turn.ipynb`: Tiny grid comparing synthetic vs LLM vs ESConv seeding modes.  
- `multiturn_policy_comparison.ipynb`: Phase 4 small-grid comparison of calming vs provocative vs validate policies over 5 turns; saves per-turn logs, summaries, heatmaps to `results/multiturn_runs/`.  
- `error_analysis_single_turn.ipynb`: Phase 3 error analysis; loads the full LLM run, reports alignment/confidence stats, and surfaces sampled mismatches with notes.
