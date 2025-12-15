# ESConv seeding helper

## What `build_esconv_seed_bank` does
- Loads the ESConv train split and grabs the **first turn** of each conversation (conversation start).
- Skips empty text and turns longer than `max_chars` (default 200).
- Classifies the first turn with `EmotionFlowAnalyzer` (DistilRoBERTa labels: anger, disgust, fear, joy, neutral, sadness, surprise).
- Keeps up to `per_emotion` snippets per requested emotion where the classifier’s `primary_emotion` matches the target.
- Returns a dict: `emotion -> [utterance, ...]`.

## Why classifier-check?
ESConv annotations differ from the DistilRoBERTa label set and may not align perfectly. We classifier-check to map ESConv snippets into the seven-label space used downstream, avoiding mismatches (e.g., ESConv “happy” turns landing as “surprise”).

## Using the bank
```python
from datasets import load_dataset
from dynamic_conversation import build_esconv_seed_bank, SingleTurnSimulator

ds = load_dataset("thu-coai/esconv")
seed_bank = build_esconv_seed_bank(ds, emotions=["anger", "joy"], per_emotion=6, max_chars=200, use_gpu=False)

sim = SingleTurnSimulator(esconv_seeds=seed_bank)  # default model gpt-4o-mini
df = sim.run_batch(
    emotions=["anger", "joy"],
    strategies=[...],
    runs_per_pair=...,
    use_esconv_seed=True,
)
```

## Notes
- If `use_esconv_seed=True` but a given emotion has no seeds in the bank, the simulator falls back to LLM seed (if enabled) then synthetic templates.
- Adjust `per_emotion`/`max_chars` to balance realism and brevity; keep seeds short to avoid token issues.
