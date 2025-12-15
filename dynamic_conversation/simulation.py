"""
Two-LLM single-turn simulation to measure emotion shifts by strategy.

Flows:
1) Seed agent A with an utterance reflecting a target emotion (synthetic).
2) Agent B replies using a specified strategy prompt (via OpenAI Chat).
3) Agent A follows up once more.
4) DistilRoBERTa classifier checks initial and post-strategy emotions.

Outputs are notebook-friendly: returns records, optional CSV + heatmap to `results/`.
"""

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import getpass
import json
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency for simulation
    OpenAI = None

from .emotion_map import EmotionFlowAnalyzer
from .response_strategy import ResponseStrategy, build_strategy_prompt

# Synthetic seeds explicitly aligned to DistilRoBERTa emotion labels
EMOTION_SEED_TEMPLATES: Dict[str, List[str]] = {
    "anger": [
        "I am furious right now about how unfair this all feels.",
        "Everything went wrong today and I am boiling with anger."
    ],
    "disgust": [
        "That whole situation left a bad taste in my mouth.",
        "I feel grossed out and really put off by what happened."
    ],
    "fear": [
        "I'm scared about what might happen next.",
        "I feel anxious and fearful that this will get worse."
    ],
    "joy": [
        "I feel genuinely happy about how things turned out.",
        "I'm excited and joyful about this news!"
    ],
    "neutral": [
        "Here's what happened earlier; I'm just laying out the facts.",
        "I'm calm about this and just explaining the situation."
    ],
    "sadness": [
        "I feel really down and heavy about all of this.",
        "I'm sad and it's hard to see a bright side right now."
    ],
    "surprise": [
        "I did not see that coming at all; I'm shocked.",
        "Wow, that was unexpected and surprising."
    ],
}

# Generic supportive prompt used for the baseline (no explicit strategy)
BASELINE_SUPPORTIVE_PROMPT = (
    "Respond supportively and succinctly to the partner's message. "
    "Show understanding without applying a specific strategy or changing their topic."
)


@dataclass
class SimulationConfig:
    """Configuration for OpenAI chat completions."""
    model: str = "gpt-5-nano"  # supports max_completion_tokens
    temperature: float = 1.0  # gpt-5-nano/mini require default temperature
    max_tokens: int = 220  # For legacy models; see _chat for overrides
    retries: int = 3
    backoff_seconds: float = 2.0
    seed_prompt_template: str = (
        "You are agent A. Produce one short, natural utterance (1-2 sentences) "
        "that clearly expresses the emotion: {emotion}. Do not add extra explanation."
    )


class SingleTurnSimulator:
    """
    Runs two-agent single-turn simulations and logs emotion shifts.

    Designed for notebook use; requires `OPENAI_API_KEY` when generating LLM outputs.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
        use_gpu: bool = False,
        config: Optional[SimulationConfig] = None,
        prompt_for_key: bool = True,
    ):
        self.config = config or SimulationConfig()
        self._client = None
        self._openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._prompt_for_key = prompt_for_key
        self.emotion_analyzer = EmotionFlowAnalyzer(model_name=emotion_model, use_gpu=use_gpu)

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if OpenAI is None:
            raise ImportError("openai package is required for simulation; install openai>=1.0.")
        if not self._openai_api_key and self._prompt_for_key:
            try:
                entered = getpass.getpass("Enter OPENAI_API_KEY: ").strip()
                if entered:
                    self._openai_api_key = entered
            except Exception:
                pass
        if not self._openai_api_key:
            raise ValueError("OPENAI_API_KEY is required to run simulations.")
        self._client = OpenAI(api_key=self._openai_api_key)

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        self._ensure_client()
        for attempt in range(1, self.config.retries + 1):
            try:
                kwargs = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                }
                # Newer models (gpt-4.1 family, gpt-5-nano) may require max_completion_tokens
                if "5" in self.config.model or "4.1" in self.config.model:
                    kwargs["max_completion_tokens"] = self.config.max_tokens
                else:
                    kwargs["max_tokens"] = self.config.max_tokens
                # Some models (e.g., gpt-5-nano/mini) only support default temperature
                if "gpt-5" in self.config.model:
                    kwargs["temperature"] = 1.0
                resp = self._client.chat.completions.create(
                    **kwargs,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                if attempt >= self.config.retries:
                    raise
                time.sleep(self.config.backoff_seconds * attempt)
        raise RuntimeError("Chat completion failed after retries.")

    def _seed_utterance(self, target_emotion: str) -> str:
        target = target_emotion.lower()
        if target not in EMOTION_SEED_TEMPLATES:
            raise ValueError(f"Unsupported emotion '{target_emotion}'. Expected one of {list(EMOTION_SEED_TEMPLATES)}")
        return random.choice(EMOTION_SEED_TEMPLATES[target])

    def _seed_with_llm(self, target_emotion: str) -> str:
        """
        Generate a seed utterance via LLM for additional variety.
        """
        self._ensure_client()
        prompt = self.config.seed_prompt_template.format(emotion=target_emotion.lower())
        kwargs = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You are agent A. Speak in first person, naturally."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
        }
        # gpt-5 / 4.1 models require max_completion_tokens, older models use max_tokens
        if "5" in self.config.model or "4.1" in self.config.model:
            kwargs["max_completion_tokens"] = self.config.max_tokens
        else:
            kwargs["max_tokens"] = self.config.max_tokens
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()

    def simulate(
        self,
        target_emotion: str,
        strategy: Optional[ResponseStrategy],
        style_modifier: Optional[str] = None,
        use_llm_seed: bool = False,
        use_baseline_strategy: bool = False,
    ) -> Dict:
        """
        Run a single simulation trial.

        Returns:
            Dict with seed text, strategy response, follow-up, and classified emotions.
        """
        seed_text = self._seed_with_llm(target_emotion) if use_llm_seed else self._seed_utterance(target_emotion)
        seed_classification = self.emotion_analyzer.classify_utterance(seed_text)

        if use_baseline_strategy:
            strategy_prompt = (
                f"{BASELINE_SUPPORTIVE_PROMPT}\nPartner said: \"{seed_text}\""
                + (f"\nStyle: {style_modifier}" if style_modifier else "")
            )
            strategy_label = "baseline"
        else:
            if strategy is None:
                raise ValueError("strategy must be provided when not using the baseline path.")
            strategy_prompt = build_strategy_prompt(
                strategy=strategy,
                partner_message=seed_text,
                conversation_context=[f"usr: {seed_text}"],
                style_modifier=style_modifier,
            )
            strategy_label = strategy.value
        strategy_reply = self._chat([
            {"role": "system", "content": "You are agent B, a supportive responder."},
            {"role": "user", "content": strategy_prompt},
        ])

        followup_prompt = (
            "You are the original speaker (agent A). Stay consistent with your initial tone and respond"
            " naturally to the last reply in 1-3 sentences."
            f"\nYour earlier message: \"{seed_text}\""
            f"\nTheir reply: \"{strategy_reply}\""
        )
        followup_reply = self._chat([
            {"role": "system", "content": "You are agent A. Speak authentically and briefly."},
            {"role": "user", "content": followup_prompt},
        ])
        followup_classification = self.emotion_analyzer.classify_utterance(followup_reply)

        return {
            "intended_emotion": target_emotion.lower(),
            "seed_text": seed_text,
            "seed_emotion_detected": seed_classification["primary_emotion"],
            "seed_confidence": seed_classification["confidence"],
            "strategy": strategy_label,
            "style_modifier": style_modifier or "",
            "strategy_reply": strategy_reply,
            "followup_reply": followup_reply,
            "followup_emotion": followup_classification["primary_emotion"],
            "followup_confidence": followup_classification["confidence"],
        }

    def run_batch(
        self,
        emotions: List[str],
        strategies: List[ResponseStrategy],
        runs_per_pair: int = 1,
        style_modifier: Optional[str] = None,
        use_llm_seed: bool = False,
        include_baseline: bool = False,
        save_csv: Optional[Path] = Path("results/single_turn_simulation.csv"),
        save_heatmap: Optional[Path] = Path("results/single_turn_heatmap.png"),
        save_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Run multiple simulations and optionally save CSV/heatmap.
        """
        records: List[Dict] = []
        if isinstance(save_csv, (str, bytes)):
            save_csv = Path(save_csv)
        if isinstance(save_heatmap, (str, bytes)):
            save_heatmap = Path(save_heatmap)
        strategy_items: List[Union[ResponseStrategy, str]] = list(strategies)
        if include_baseline:
            strategy_items.append("baseline")

        for emotion in emotions:
            for strategy in strategy_items:
                for _ in range(runs_per_pair):
                    is_baseline = strategy == "baseline"
                    result = self.simulate(
                        emotion,
                        strategy if not is_baseline else None,
                        style_modifier=style_modifier,
                        use_llm_seed=use_llm_seed,
                        use_baseline_strategy=is_baseline,
                    )
                    records.append(result)

        df = pd.DataFrame(records)

        if save_csv:
            Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_csv, index=False, encoding="utf-8")

        if save_metadata and save_csv:
            metadata = {
                "emotions": emotions,
                "strategies": [s if isinstance(s, str) else s.value for s in strategy_items],
                "runs_per_pair": runs_per_pair,
                "style_modifier": style_modifier or "",
                "use_llm_seed": use_llm_seed,
                "include_baseline": include_baseline,
                "config": {
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "retries": self.config.retries,
                    "backoff_seconds": self.config.backoff_seconds,
                },
            }
            meta_path = save_csv.with_suffix(".meta.json")
            Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        if save_heatmap:
            self._plot_heatmap(df, save_path=save_heatmap)

        return df

    def _plot_heatmap(self, df: pd.DataFrame, save_path: Path) -> None:
        pivot = (
            df.groupby(["intended_emotion", "strategy", "followup_emotion"])
            .size()
            .reset_index(name="count")
        )
        heat_data = pivot.pivot_table(
            index=["intended_emotion", "strategy"],
            columns="followup_emotion",
            values="count",
            fill_value=0,
        )

        plt.figure(figsize=(10, 6))
        sns.heatmap(heat_data, annot=True, fmt=".0f", cmap="Blues")
        plt.title("Emotion shift counts by initial emotion and strategy")
        plt.ylabel("Initial emotion / Strategy")
        plt.xlabel("Post-strategy emotion")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def _wilson_ci(successes: int, total: int, confidence: float = 0.95) -> (float, float):
    """
    Compute Wilson score interval for a binomial proportion.
    """
    if total == 0:
        return (0.0, 0.0)
    # Approximate z for common confidence levels
    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.90:
        z = 1.64
    elif confidence == 0.99:
        z = 2.58
    else:
        # Default fallback to 95% if unspecified
        z = 1.96
    phat = successes / total
    denom = 1 + (z**2) / total
    center = phat + (z**2) / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + (z**2) / (4 * total)) / total)
    return ((center - margin) / denom, (center + margin) / denom)


def compute_transition_confidence_intervals(
    df: pd.DataFrame,
    confidence: float = 0.95,
    group_cols: Optional[List[str]] = None,
    emotion_col: str = "followup_emotion",
) -> pd.DataFrame:
    """
    Compute Wilson confidence intervals for emotion distributions per group.

    Returns a DataFrame with columns: group_cols + target_emotion, count, total, proportion, ci_low, ci_high.
    """
    group_cols = group_cols or ["intended_emotion", "strategy"]
    records: List[Dict] = []

    for keys, group in df.groupby(group_cols):
        total = len(group)
        counts = group[emotion_col].value_counts()
        key_dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        for target_emotion, count in counts.items():
            ci_low, ci_high = _wilson_ci(count, total, confidence=confidence)
            records.append(
                {
                    **key_dict,
                    "target_emotion": target_emotion,
                    "count": int(count),
                    "total": int(total),
                    "proportion": count / total if total else 0.0,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    return pd.DataFrame(records)
