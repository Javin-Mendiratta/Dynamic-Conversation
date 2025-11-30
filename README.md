# Dynamic-Conversation

Emotion flow analysis and (future) strategy simulation for multi-turn dialogues.

## Layout
- `dynamic_conversation/`: importable package  
  - `emotion_map.py`: `EmotionFlowAnalyzer` (classify turns, transition matrices, trajectory scores, plots)  
  - `load_data.py`: `EsConvData` helper  
  - `__init__.py`: exports public API
- `results/`: generated artifacts (heatmaps, Sankey, CSV scores)
- `notebooks/`: ad hoc experiments
- `pyproject.toml`: packaging metadata/deps

## Install
```bash
conda create -n dynamic-conversation python=3.11
conda activate dynamic-conversation
pip install -e .
```

## Quick use (Python)
```python
from dynamic_conversation import EmotionFlowAnalyzer
from datasets import load_dataset

ds = load_dataset("thu-coai/esconv")
analyzer = EmotionFlowAnalyzer(use_gpu=False)
analyzer.process_dataset(ds, max_conversations=50)
tm = analyzer.compute_transition_matrix()
trajectory = analyzer.compute_all_trajectory_scores()
analyzer.plot_transition_heatmap()
analyzer.plot_sankey_diagram(conversation_id=0)
analyzer.plot_aggregate_emotion_flow()
```

## Notes
- GPU is optional; default `use_gpu=True` to auto-detect CUDA if available.

## Dev tips
- Add new modules under `dynamic_conversation/` and export public classes/functions in `dynamic_conversation/__init__.py` to keep imports clean.
- Keep generated artifacts out of the package; write to `results/` or other top-level folders (already excluded from packaging).
