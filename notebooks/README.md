# Notebooks Guide

- Purpose: runnable experiments and result generation; keep logic in `dynamic_conversation/` and import it here.
- Environment: use Conda env `dynamic-conversation` locally; kernelspec `dyn-conv` (`python -m ipykernel install --user --name dyn-conv --display-name "dyn-conv"`).
- Colab: either connect to local runtime and pick `dyn-conv`, or on hosted Colab `pip install git+<repo-url>` (and `requirements-colab.txt` once added) then run with GPU as needed.
- Outputs: avoid committing large outputs; write figures/CSVs/HTML to `results/` with run configs/seeds logged alongside.
- Strategy prompts: import via `from dynamic_conversation import ResponseStrategy, build_strategy_prompt, STRATEGY_PROMPT_TEMPLATES` for notebook simulations.
- Single-turn simulations: import via `from dynamic_conversation import SingleTurnSimulator, SimulationConfig` for Phase 3 runs (OpenAI key required). New options: higher default completion caps (400+ with auto bump on output-limit errors), brevity hints, and seed fallbacks (LLM → synthetic; optional ESConv seed bank via `esconv_seeds`). Recommended model for reliability: `gpt-4o-mini`; bump `max_tokens` (e.g., 800) for sanity runs.
- Emotion flow notebook: `EmotionFlowAnalyzer` defaults to batch_size=64 for GPU (A100); reduce batch size if you encounter memory pressure.
- Notebook map:
  - `esconv_emotion_flow.ipynb`: Phase 1 exploration using `EmotionFlowAnalyzer` over ESConv; mirrors `main` in `dynamic_conversation/emotion_map.py` (classification, transitions, trajectory scores, plots). GPU-friendly via Colab.
  - `esconv_strategy_exploration.ipynb`: Strategy exploration using `EmotionStrategyAnalyzer` (6-strategy fine-tune + analysis) over ESConv; mirrors `main` in `dynamic_conversation/response_strategy.py`. GPU recommended.
  - `single_turn_simulation.ipynb`: Phase 3 demo using `SingleTurnSimulator` with synthetic seeds + strategy prompts; writes CSV/heatmap to `results/`.
  - `sanity_single_turn.ipynb`: quick 2-emotion sanity run to confirm non-empty seeds/replies and check log/meta after robustness fixes.
