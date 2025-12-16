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
from typing import Callable, Dict, List, Optional, Tuple, Union
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
    model: str = "gpt-4o-mini"  # reliable for short completions, supports max_completion_tokens
    temperature: float = 1.0  # gpt-5-nano/mini require default temperature
    max_tokens: int = 400  # For legacy models; see _chat for overrides
    retries: int = 3
    backoff_seconds: float = 2.0
    seed_prompt_template: str = (
        "You are agent A. Produce one short, natural utterance (1-2 sentences) "
        "that clearly expresses the emotion: {emotion}. Do not add extra explanation."
    )
    brevity_hint: str = "Reply in 1-3 sentences."


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
        prompt_for_key: bool = False,
        esconv_seeds: Optional[Dict[str, List[str]]] = None,
    ):
        self.config = config or SimulationConfig()
        self._client = None
        self._openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._prompt_for_key = prompt_for_key
        self.emotion_analyzer = EmotionFlowAnalyzer(model_name=emotion_model, use_gpu=use_gpu)
        self.esconv_seeds = esconv_seeds or {}

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

    def _chat(self, messages: List[Dict[str, str]], allow_empty: bool = False) -> str:
        self._ensure_client()
        max_tokens = self.config.max_tokens
        for attempt in range(1, self.config.retries + 1):
            try:
                kwargs = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                }
                # Newer models (gpt-4.1 family, gpt-5-nano) may require max_completion_tokens
                if "5" in self.config.model or "4.1" in self.config.model:
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens
                # Some models (e.g., gpt-5-nano/mini) only support default temperature
                if "gpt-5" in self.config.model:
                    kwargs["temperature"] = 1.0
                resp = self._client.chat.completions.create(
                    **kwargs,
                )
                content = (resp.choices[0].message.content or "").strip()
                if not allow_empty and not content:
                    raise RuntimeError("Empty completion received")
                return content
            except Exception as exc:
                if attempt >= self.config.retries:
                    raise
                msg = str(exc)
                # If we hit model output limit, bump max tokens and retry
                if "max_tokens" in msg or "max_completion_tokens" in msg or "output limit" in msg:
                    max_tokens = int(max_tokens * 1.5)
                time.sleep(self.config.backoff_seconds * attempt)
        raise RuntimeError("Chat completion failed after retries.")

    def _generate_seed_text(
        self,
        target_emotion: str,
        use_llm_seed: bool = False,
        use_esconv_seed: bool = False,
    ) -> Tuple[str, Dict]:
        """
        Produce a seed utterance using ESConv, LLM, or synthetic fallback and classify it.
        """
        seed_text = ""
        if use_esconv_seed:
            seed_text = self._seed_from_esconv(target_emotion) or ""
        if use_llm_seed and not seed_text:
            try:
                seed_text = self._seed_with_llm(target_emotion)
            except Exception:
                seed_text = ""
        if not seed_text:
            seed_text = self._seed_utterance(target_emotion)

        if not seed_text.strip():
            raise RuntimeError("Empty seed text generated")

        seed_classification = self.emotion_analyzer.classify_utterance(seed_text)
        return seed_text, seed_classification

    def _seed_utterance(self, target_emotion: str) -> str:
        target = target_emotion.lower()
        if target not in EMOTION_SEED_TEMPLATES:
            raise ValueError(f"Unsupported emotion '{target_emotion}'. Expected one of {list(EMOTION_SEED_TEMPLATES)}")
        return random.choice(EMOTION_SEED_TEMPLATES[target])

    def _seed_from_esconv(self, target_emotion: str) -> Optional[str]:
        """Return a seed from a prebuilt ESConv seed bank if available."""
        seeds = self.esconv_seeds.get(target_emotion.lower()) or []
        return random.choice(seeds) if seeds else None

    def _seed_with_llm(self, target_emotion: str) -> str:
        """
        Generate a seed utterance via LLM for additional variety.
        """
        prompt = self.config.seed_prompt_template.format(emotion=target_emotion.lower())
        return self._chat(
            [
                {"role": "system", "content": "You are agent A. Speak in first person, naturally."},
                {"role": "user", "content": prompt},
            ]
        )

    def simulate(
        self,
        target_emotion: str,
        strategy: Optional[ResponseStrategy],
        style_modifier: Optional[str] = None,
        use_llm_seed: bool = False,
        use_baseline_strategy: bool = False,
        use_esconv_seed: bool = False,
    ) -> Dict:
        """
        Run a single simulation trial.

        Returns:
            Dict with seed text, strategy response, follow-up, and classified emotions.
        """
        seed_text, seed_classification = self._generate_seed_text(
            target_emotion,
            use_llm_seed=use_llm_seed,
            use_esconv_seed=use_esconv_seed,
        )

        if use_baseline_strategy:
            strategy_prompt = (
                f"{BASELINE_SUPPORTIVE_PROMPT}\nPartner said: \"{seed_text}\""
                + (f"\nStyle: {style_modifier}" if style_modifier else "")
            )
            strategy_label = "baseline"
        else:
            if strategy is None:
                raise ValueError("strategy must be provided when not using the baseline path.")
            strategy_style = style_modifier or ""
            if self.config.brevity_hint:
                strategy_style = f"{strategy_style} {self.config.brevity_hint}".strip()
            strategy_prompt = build_strategy_prompt(
                strategy=strategy,
                partner_message=seed_text,
                conversation_context=[f"usr: {seed_text}"],
                style_modifier=strategy_style,
            )
            strategy_label = strategy.value
        strategy_reply = self._chat([
            {"role": "system", "content": "You are agent B, a supportive responder."},
            {"role": "user", "content": strategy_prompt},
        ])
        if not strategy_reply.strip():
            raise RuntimeError("Empty strategy reply generated")

        followup_prompt = (
            "You are the original speaker (agent A). Stay consistent with your initial tone and respond"
            " naturally to the last reply."
            f" Preserve the original emotion {target_emotion.lower()} unless it truly shifts in context."
            f" {self.config.brevity_hint}"
            f"\nYour earlier message: \"{seed_text}\""
            f"\nTheir reply: \"{strategy_reply}\""
        )
        followup_reply = self._chat([
            {"role": "system", "content": "You are agent A. Speak authentically and briefly."},
            {"role": "user", "content": followup_prompt},
        ])
        if not followup_reply.strip():
            raise RuntimeError("Empty follow-up reply generated")
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
        use_esconv_seed: bool = False,
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
                    try:
                        result = self.simulate(
                            emotion,
                            strategy if not is_baseline else None,
                            style_modifier=style_modifier,
                            use_llm_seed=use_llm_seed,
                            use_baseline_strategy=is_baseline,
                            use_esconv_seed=use_esconv_seed,
                        )
                        result["status"] = "success"
                        records.append(result)
                    except Exception as exc:  # pragma: no cover - defensive logging
                        records.append({
                            "intended_emotion": emotion,
                            "strategy": strategy if not is_baseline else "baseline",
                            "seed_text": "",
                            "seed_emotion_detected": "",
                            "seed_confidence": 0.0,
                            "style_modifier": style_modifier or "",
                            "strategy_reply": "",
                            "followup_reply": "",
                            "followup_emotion": "",
                            "followup_confidence": 0.0,
                            "status": f"failed: {exc}",
                        })

        df = pd.DataFrame(records)
        success_df = df[df["status"] == "success"].copy()

        if save_csv:
            Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
            success_df.drop(columns=["status"], errors="ignore").to_csv(save_csv, index=False, encoding="utf-8")

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
                "success_count": int((df["status"] == "success").sum()),
                "failure_count": int((df["status"] != "success").sum()),
            }
            meta_path = save_csv.with_suffix(".meta.json")
            Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        if save_heatmap:
            self._plot_heatmap(success_df, save_path=save_heatmap)

        # Write a simple log with failures if any
        if save_csv:
            log_path = save_csv.with_suffix(".log")
            with open(log_path, "w", encoding="utf-8") as f:
                for _, row in df.iterrows():
                    if row.get("status") == "success":
                        continue
                    f.write(f"{row.get('intended_emotion')} / {row.get('strategy')}: {row.get('status')}\n")

        return success_df

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


class MultiTurnRollout(SingleTurnSimulator):
    """
    Lightweight multi-turn simulator driven by deterministic policies.

    Runs a short conversation (3–5 turns) where agent B chooses a strategy
    based on the current classified emotion of agent A.
    """

    def __init__(
        self,
        turns: int = 5,
        default_style: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.turns = turns
        self.default_style = default_style

    @staticmethod
    def _format_history(history: List[str], window: int = 8) -> str:
        """Join recent history lines into a readable context block."""
        return "\n".join(history[-window:])

    def _run_single_conversation(
        self,
        policy_name: str,
        target_emotion: str,
        policy_fn: Callable[[str], ResponseStrategy],
        turns: Optional[int] = None,
        style_modifier: Optional[str] = None,
        use_llm_seed: bool = True,
        use_esconv_seed: bool = False,
        conversation_id: Optional[str] = None,
    ) -> Tuple[List[Dict], Dict]:
        turns = turns or self.turns
        style_modifier = style_modifier or self.default_style
        seed_text, seed_classification = self._generate_seed_text(
            target_emotion,
            use_llm_seed=use_llm_seed,
            use_esconv_seed=use_esconv_seed,
        )

        conversation_id = conversation_id or f"{policy_name}-{target_emotion}-{int(time.time()*1000)}"
        current_emotion = seed_classification["primary_emotion"]
        history: List[str] = [f"usr: {seed_text}"]
        per_turn_records: List[Dict] = [
            {
                "conversation_id": conversation_id,
                "policy": policy_name,
                "turn": 0,
                "strategy": "seed",
                "prior_emotion": "",
                "detected_emotion": current_emotion,
                "detected_confidence": seed_classification["confidence"],
                "seed_text": seed_text,
                "assistant_reply": "",
                "user_reply": seed_text,
                "emotion_scores": json.dumps(seed_classification.get("all_scores", {})),
            }
        ]
        emotion_flow: List[Dict] = [
            {
                "speaker": "user",
                "text": seed_text,
                "emotion": current_emotion,
                "confidence": seed_classification["confidence"],
                "all_scores": seed_classification.get("all_scores", {}),
            }
        ]
        last_user_text = seed_text

        for turn_idx in range(1, turns + 1):
            strategy = policy_fn(current_emotion)
            if not isinstance(strategy, ResponseStrategy):
                raise ValueError("Policy function must return a ResponseStrategy.")
            style_text = style_modifier or ""
            if self.config.brevity_hint:
                style_text = f"{style_text} {self.config.brevity_hint}".strip()

            strategy_prompt = build_strategy_prompt(
                strategy=strategy,
                partner_message=last_user_text,
                conversation_context=history[-8:],
                style_modifier=style_text,
            )
            strategy_reply = self._chat(
                [
                    {"role": "system", "content": "You are agent B, a supportive responder."},
                    {
                        "role": "user",
                        "content": f"Conversation so far:\n{self._format_history(history)}\n\n"
                                   f"Now reply using the strategy instructions below:\n{strategy_prompt}"
                    },
                ]
            )
            history.append(f"bot: {strategy_reply}")

            followup_prompt = (
                "You are the original speaker (agent A). Reply naturally to the last message."
                " Keep it brief and authentic. Your emotion can change if the response shifts how you feel."
                f" {self.config.brevity_hint}"
                f"\nMost recent reply to you: \"{strategy_reply}\""
                f"\nConversation so far:\n{self._format_history(history)}"
            )
            followup_reply = self._chat(
                [
                    {"role": "system", "content": "You are agent A. Speak authentically and briefly."},
                    {"role": "user", "content": followup_prompt},
                ]
            )
            history.append(f"usr: {followup_reply}")

            followup_classification = self.emotion_analyzer.classify_utterance(followup_reply)
            current_emotion = followup_classification["primary_emotion"]
            last_user_text = followup_reply
            emotion_flow.append(
                {
                    "speaker": "user",
                    "text": followup_reply,
                    "emotion": current_emotion,
                    "confidence": followup_classification["confidence"],
                    "all_scores": followup_classification.get("all_scores", {}),
                }
            )
            per_turn_records.append(
                {
                    "conversation_id": conversation_id,
                    "policy": policy_name,
                    "turn": turn_idx,
                    "strategy": strategy.value,
                    "prior_emotion": emotion_flow[-2]["emotion"],
                    "detected_emotion": current_emotion,
                    "detected_confidence": followup_classification["confidence"],
                    "seed_text": seed_text,
                    "assistant_reply": strategy_reply,
                    "user_reply": followup_reply,
                    "emotion_scores": json.dumps(followup_classification.get("all_scores", {})),
                }
            )

        trajectory_score = self.emotion_analyzer.compute_emotion_trajectory_score(
            emotion_flow,
            target_emotion.lower(),
        )

        summary = {
            "conversation_id": conversation_id,
            "policy": policy_name,
            "intended_emotion": target_emotion.lower(),
            "seed_emotion_detected": seed_classification["primary_emotion"],
            "final_emotion": current_emotion,
            "turns": turns,
            "trajectory_to_intended": trajectory_score,
        }
        return per_turn_records, summary

    def run_policy_batch(
        self,
        policy_name: str,
        policy_fn: Callable[[str], ResponseStrategy],
        emotions: List[str],
        runs_per_emotion: int = 1,
        turns: Optional[int] = None,
        style_modifier: Optional[str] = None,
        use_llm_seed: bool = True,
        use_esconv_seed: bool = False,
        save_csv: Optional[Union[str, Path]] = None,
        save_plot: Optional[Union[str, Path]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run multi-turn simulations for a single policy across emotions/runs.

        Returns:
            (per_turn_df, summary_df)
        """
        turns = turns or self.turns
        per_turn_rows: List[Dict] = []
        summary_rows: List[Dict] = []

        if save_csv is None:
            save_csv = Path(f"results/multiturn_{policy_name}.csv")
        if isinstance(save_csv, (str, bytes)):
            save_csv = Path(save_csv)
        if save_plot and isinstance(save_plot, (str, bytes)):
            save_plot = Path(save_plot)

        for emotion in emotions:
            for run_idx in range(runs_per_emotion):
                try:
                    conv_id = f"{policy_name}-{emotion}-{run_idx}"
                    turn_rows, summary = self._run_single_conversation(
                        policy_name=policy_name,
                        target_emotion=emotion,
                        policy_fn=policy_fn,
                        turns=turns,
                        style_modifier=style_modifier,
                        use_llm_seed=use_llm_seed,
                        use_esconv_seed=use_esconv_seed,
                        conversation_id=conv_id,
                    )
                    per_turn_rows.extend(turn_rows)
                    summary_rows.append({**summary, "status": "success"})
                except Exception as exc:  # pragma: no cover - defensive logging
                    summary_rows.append(
                        {
                            "conversation_id": f"{policy_name}-{emotion}-{run_idx}",
                            "policy": policy_name,
                            "intended_emotion": emotion,
                            "seed_emotion_detected": "",
                            "final_emotion": "",
                            "turns": turns,
                            "trajectory_to_intended": 0.0,
                            "status": f"failed: {exc}",
                        }
                    )

        per_turn_df = pd.DataFrame(per_turn_rows)
        summary_df = pd.DataFrame(summary_rows)

        if save_csv:
            save_csv.parent.mkdir(parents=True, exist_ok=True)
            per_turn_df.to_csv(save_csv, index=False, encoding="utf-8")
            summary_df.to_csv(save_csv.with_suffix(".summary.csv"), index=False, encoding="utf-8")

        if save_plot and not per_turn_df.empty:
            self._plot_policy_heatmap(per_turn_df, save_path=save_plot, policy_name=policy_name)

        return per_turn_df, summary_df

    def _plot_policy_heatmap(self, df: pd.DataFrame, save_path: Path, policy_name: str) -> None:
        """
        Heatmap of emotion counts by turn for a given policy.
        """
        pivot = (
            df.groupby(["turn", "detected_emotion"])
            .size()
            .reset_index(name="count")
            .pivot_table(index="turn", columns="detected_emotion", values="count", fill_value=0)
        )
        plt.figure(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Greens")
        plt.title(f"Emotion by turn - {policy_name}")
        plt.xlabel("Emotion")
        plt.ylabel("Turn (0 = seed)")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def build_esconv_seed_bank(
    esconv_dataset,
    emotions: List[str],
    per_emotion: int = 8,
    max_chars: int = 200,
    use_gpu: bool = False,
    analyzer: Optional[EmotionFlowAnalyzer] = None,
) -> Dict[str, List[str]]:
    """
    Build a small seed bank from ESConv conversation starts per target emotion.

    Args:
        esconv_dataset: HuggingFace dataset with ESConv splits.
        emotions: Target emotion labels to collect (e.g., ['anger', 'joy']).
        per_emotion: Max seeds to keep per emotion.
        max_chars: Skip turns longer than this many characters.
        use_gpu: Whether to load EmotionFlowAnalyzer with GPU.
        analyzer: Optional preloaded EmotionFlowAnalyzer to reuse the classifier.

    Returns:
        Dict emotion -> list of seed utterances.
    """
    analyzer = analyzer or EmotionFlowAnalyzer(use_gpu=use_gpu)
    bank: Dict[str, List[str]] = {e: [] for e in emotions}
    remaining = set(emotions)

    data = esconv_dataset["train"]
    for idx in range(len(data)):
        if not remaining:
            break
        try:
            conv = json.loads(data[idx]["text"])
        except Exception:
            continue
        dialogue = conv.get("dialog") or []
        if not dialogue:
            continue
        first_turn = dialogue[0]
        text = (first_turn.get("text") or "").strip()
        if not text or len(text) > max_chars:
            continue
        res = analyzer.classify_utterance(text)
        label = res["primary_emotion"]
        if label in remaining and len(bank[label]) < per_emotion:
            bank[label].append(text)
            if len(bank[label]) >= per_emotion:
                remaining.discard(label)

    return bank

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


def calming_policy(emotion: str) -> ResponseStrategy:
    """
    De-escalation focused mapping: soften high-negatives, steady neutrals,
    stay exploratory/normalizing on positives.
    """
    emo = (emotion or "").lower()
    mapping = {
        "anger": ResponseStrategy.VALIDATE,
        "disgust": ResponseStrategy.NORMALIZE,
        "sadness": ResponseStrategy.AFFIRM,
        "neutral": ResponseStrategy.GUIDE,
        "joy": ResponseStrategy.EXPLORE,
        "surprise": ResponseStrategy.EXPLORE,
        "fear": ResponseStrategy.NORMALIZE,
    }
    return mapping.get(emo, ResponseStrategy.VALIDATE)


def provocative_policy(emotion: str) -> ResponseStrategy:
    """
    Movement-focused mapping: reframe or push for action to introduce volatility.
    """
    emo = (emotion or "").lower()
    mapping = {
        "anger": ResponseStrategy.REFRAME,
        "disgust": ResponseStrategy.REFRAME,
        "sadness": ResponseStrategy.GUIDE,
        "neutral": ResponseStrategy.GUIDE,
        "joy": ResponseStrategy.REFRAME,
        "surprise": ResponseStrategy.REFRAME,
        "fear": ResponseStrategy.GUIDE,
    }
    return mapping.get(emo, ResponseStrategy.REFRAME)


def always_validate_policy(_: str) -> ResponseStrategy:
    """Baseline policy that always uses Validate."""
    return ResponseStrategy.VALIDATE
