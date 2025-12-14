# TODOs by Phase (notebook-first; ESConv for Phase 1)

## Phase 4: Multi-turn policy simulation
- Goal: roll out multi-turn dialogues under policies to steer toward target emotions.
- Tasks:
  - Implement policy interface (static mappings, greedy from Phase 3 stats) and rollout driver (≈24 turns), logging trajectories and plots.
  - Provide simple function calls for notebooks; optional CLI only if batch runs needed.
  - ***DEFINE TARGET EMOTIONS FOR TRAJECTORY REWARD (E.G., NEUTRAL, JOY, USER-INPUT)***

## Phase 5: RL policy learning (optional)
- Goal: learn strategy-selection policies in an MDP over emotional states.
- Tasks:
  - Build `rl_policy.py` environment wrapping simulator with reward = change toward target emotion.
  - Implement baseline learner (tabular/Q-learning) and save curves/policies to `results/`.
  - ***PICK ALGORITHM/LIB (PURE PYTHON Q VS. DQN/OTHERS)***

## Evaluation & analysis
- Goal: assess coherence/empathy and reproducibility of simulations.
- Tasks:
  - Add BERTScore/semantic similarity hooks and small manual-check sampling for generated replies (strategy adherence, empathy/helpfulness).
  - Add a lightweight safety/toxicity probe on generated replies.
  - Run multiple seeds/runs_per_pair and report variance/CI on shift tables.
  - Log run configs/seeds alongside outputs; add brief reproduction notes in `results/`.
  - ***SELECT EVALUATION METRICS BEYOND EMOTION SHIFT (BERTSCORE, MANUAL CHECKS, ???)***

## Runtime integration
- Goal: keep APIs clean for notebooks; minimize CLIs.
- Tasks:
  - Export new helpers via `dynamic_conversation/__init__.py`; keep `python -m dynamic_conversation` as smoke test only if retained.
  - Handle API key config for OpenAI and rate-limit/backoff guidance. (Default: `OPENAI_API_KEY`, model `gpt-5-nano`, retries/backoff in `SimulationConfig`; adjust per run if needed.)
  - ***ADD SECOND EMOTION CLASSIFIER FOR CROSS-CHECK OR DOCUMENT WHY SINGLE CLASSIFIER IS SUFFICIENT***

## Visualization defaults
- Goal: consistent, reproducible outputs.
- Tasks:
  - Standardize plot styling and file formats (PNG/HTML) and document how many runs per figure.
  - ***DECIDE DEFAULT FORMATS/RUN COUNTS FOR FIGURES***

## Notebook hygiene
- Goal: keep notebooks lean and reproducible.
- Tasks:
  - Ensure notebooks import package logic, avoid heavy outputs in git, and start with a short how-to-run cell (env/kernel, install instructions for hosted Colab).
  - Maintain `notebooks/README.md` mapping notebooks to outputs/resources; note GPU requirements.
