"""
6-Strategy Framework:
    1. Validate - Empathetic acknowledgment
    2. Explore - Deepening understanding  
    3. Reframe - Perspective shift
    4. Affirm - Confidence building
    5. Guide - Action-oriented advice
    6. Normalize - Universal experience
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets import load_dataset, Dataset
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from enum import Enum


class ResponseStrategy(Enum):

    VALIDATE = "Validate"       # Empathetic acknowledgment
    EXPLORE = "Explore"         # Deepening understanding
    REFRAME = "Reframe"         # Perspective shift
    AFFIRM = "Affirm"           # Confidence building
    GUIDE = "Guide"             # Action-oriented advice
    NORMALIZE = "Normalize"     # Universal experience


STRATEGY_PROMPT_TEMPLATES = {

    ResponseStrategy.VALIDATE: (
        "You are a supportive responder. Use the VALIDATE strategy: acknowledge and mirror the "
        "speaker's feelings without adding advice or new topics. Keep the focus on how they feel. "
        "{style_modifier}\n"
        "Recent context:\n{context}\n"
        "Partner just said: \"{partner_message}\"\n"
        "Respond in a single turn with a validating message."
    ),

    ResponseStrategy.EXPLORE: (
        "You are a supportive responder. Use the EXPLORE strategy: ask gentle, open questions to "
        "deepen understanding and invite elaboration. Avoid giving advice or reframing. {style_modifier}\n"
        "Recent context:\n{context}\n"
        "Partner just said: \"{partner_message}\"\n"
        "Respond in a single turn with exploratory questions or prompts."
    ),

    ResponseStrategy.REFRAME: (
        "You are a supportive responder. Use the REFRAME strategy: offer a perspective shift that "
        "helps the speaker see their situation differently, while staying empathetic. Avoid directives. "
        "{style_modifier}\n"
        "Recent context:\n{context}\n"
        "Partner just said: \"{partner_message}\"\n"
        "Respond in a single turn with a gentle reframing."
    ),

    ResponseStrategy.AFFIRM: (
        "You are a supportive responder. Use the AFFIRM strategy: highlight strengths, efforts, or "
        "positive traits to build confidence. Do not introduce new topics or advice. {style_modifier}\n"
        "Recent context:\n{context}\n"
        "Partner just said: \"{partner_message}\"\n"
        "Respond in a single turn with an affirming message."
    ),

    ResponseStrategy.GUIDE: (
        "You are a supportive responder. Use the GUIDE strategy: offer clear, actionable steps or "
        "practical suggestions while staying concise. Keep empathy but focus on doable next moves. "
        "{style_modifier}\n"
        "Recent context:\n{context}\n"
        "Partner just said: \"{partner_message}\"\n"
        "Respond in a single turn with brief guidance."
    ),

    ResponseStrategy.NORMALIZE: (
        "You are a supportive responder. Use the NORMALIZE strategy: note that others experience similar "
        "feelings to reduce isolation, without minimizing their experience. Avoid advice. {style_modifier}\n"
        "Recent context:\n{context}\n"
        "Partner just said: \"{partner_message}\"\n"
        "Respond in a single turn with a normalizing, empathetic message."
    ),
}


def build_strategy_prompt(strategy, partner_message, conversation_context = None, style_modifier = None):
    """
    Args:
        strategy: Target ResponseStrategy to enforce.
        partner_message: Latest utterance from the partner/user.
        conversation_context: Optional recent turns to ground the reply.
        style_modifier: Optional stylistic cue (e.g., "be concise", "warmer tone").

    Returns:
        A formatted prompt string ready for LLM completion.
    """

    template = STRATEGY_PROMPT_TEMPLATES[strategy]

    if conversation_context:
        context_block = "\n".join(f"- {turn}" for turn in conversation_context[-4:])

    else:
        context_block = "- (no prior context provided)"

    style_text = f"Style modifier: {style_modifier}." if style_modifier else "Style modifier: stay natural."

    return template.format(
        context = context_block,
        partner_message = partner_message.strip(),
        style_modifier = style_text,
    )


class StrategyDatasetBuilder:
    """
    Builds a labeled dataset for training the strategy classifier.
    Maps ESConv's 8 strategies to our 6-strategy framework.
    """
    
    def __init__(self):
        """Initialize the dataset builder with ESConv → 6-strategy mapping."""
        
        self.strategy_map = {
            "Reflection of feelings": ResponseStrategy.VALIDATE.value,
            "Question": ResponseStrategy.EXPLORE.value,
            "Restatement or Paraphrasing": ResponseStrategy.REFRAME.value,
            "Affirmation and Reassurance": ResponseStrategy.AFFIRM.value,
            "Providing Suggestions": ResponseStrategy.GUIDE.value,
            "Information": ResponseStrategy.GUIDE.value,
            "Self-disclosure": ResponseStrategy.NORMALIZE.value,            
            "Others": ResponseStrategy.EXPLORE.value
        }
    
 
    
    def build_dataset(self, esconv_dataset, max_conversations = None):
        """
        Args:
            esconv_dataset: ESConv dataset from HuggingFace
            max_conversations: Maximum conversations to process
            
        Returns:
            Tuple of (texts, labels)
        """
    
        texts = []
        labels = []
        
        data = esconv_dataset['train']
        if max_conversations:
            data = data.select(range(min(max_conversations, len(data))))
        
        for idx in tqdm(range(len(data)), desc = "Extracting labeled examples"):

            try:
                conv_data = json.loads(data[idx]['text'])
            except (json.JSONDecodeError, KeyError) as e:
                continue
            
            if 'dialog' not in conv_data:
                continue
            
            dialogue = conv_data['dialog']
            conversation_history = []
            
            for _, turn in enumerate(dialogue):
                
                text = turn.get('text', '').strip()
                speaker = turn.get('speaker', '')
                
                annotation = turn.get('annotation', {})
                if not annotation and 'strategy' in turn:
                    annotation = {'strategy': turn['strategy']}
                
                if not text:
                    continue
                
                if conversation_history:
            
                    seeker_messages = [
                        msg for msg in conversation_history[-4:] 
                        if msg.startswith('usr:') or msg.startswith('seeker:')
                    ]
                    
                    if seeker_messages:
                        input_text = " [SEP] ".join(seeker_messages[-2:])

                    else:
                        input_text = " [SEP] ".join(conversation_history[-2:])

                else:
                    input_text = text
                
                if speaker in ['sys', 'supporter'] and annotation:
                    strategy = annotation.get('strategy', '')
                    
                    if strategy in self.strategy_map:
                        mapped_strategy = self.strategy_map[strategy]
 
                        if conversation_history:
                            texts.append(input_text)
                            labels.append(mapped_strategy)
                
                conversation_history.append(f"{speaker}: {text}")
               
        return texts, labels


class FineTunedStrategySelector:
    """
    Fine-tunes and uses a transformer model for strategy prediction.
    """
    
    def __init__(self, base_model: str = "j-hartmann/emotion-english-distilroberta-base", model_save_path: str = "../models/strategy_classifier_6way", use_gpu: bool = True):
        """
        Initialize the strategy selector.
        
        Args:
            base_model: Base transformer model to fine-tune
            model_save_path: Where to save the fine-tuned model
            use_gpu: Whether to use GPU
        """
        
        self.base_model = base_model
        self.model_save_path = model_save_path
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        self.strategy_labels = [s.value for s in ResponseStrategy]
        self.label2id = {label: idx for idx, label in enumerate(self.strategy_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
    
    def prepare_dataset(self, texts: List[str], labels: List[str]) -> Dataset:
        """
        Args:
            texts: List of input texts
            labels: List of strategy labels
            
        Returns:
            HuggingFace Dataset object
        """

        label_ids = [self.label2id[label] for label in labels]

        encoded = self.tokenizer(
            texts,
            padding = False,
            truncation = True,
            max_length = 512
        )
        
        dataset_dict = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label': label_ids
        }
        
        dataset = Dataset.from_dict(dataset_dict)       
        return dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
    
    def fine_tune(self, train_texts, train_labels, eval_texts = None, eval_labels = None, num_epochs = 5, batch_size = 16, learning_rate = 2e-5, weight_decay = 0.01):
        """
        Fine-tune the model on strategy prediction task.
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels = len(self.strategy_labels),
            id2label = self.id2label,
            label2id = self.label2id,
            problem_type = "single_label_classification",
            ignore_mismatched_sizes = True
        )
        
        if self.use_gpu:
            self.model = self.model.to(self.device)
        
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        
        eval_dataset = None

        if eval_texts and eval_labels:
            eval_dataset = self.prepare_dataset(eval_texts, eval_labels)
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir = self.model_save_path,
            eval_strategy = "epoch" if eval_dataset else "no",
            save_strategy = "epoch",
            learning_rate = learning_rate,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            num_train_epochs = num_epochs,
            weight_decay = weight_decay,
            load_best_model_at_end = True if eval_dataset else False,
            metric_for_best_model = "f1_weighted" if eval_dataset else None,
            push_to_hub = False,
            logging_steps = 50,
            save_total_limit = 2,
            fp16 = self.use_gpu,
        )
        
        callbacks = []
        if eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        self.trainer.train()
        self.trainer.save_model(self.model_save_path)
        self.tokenizer.save_pretrained(self.model_save_path)

        print(f"Model saved to {self.model_save_path}...")
        
        if eval_dataset:
            
            self.trainer.evaluate()
            predictions = self.trainer.predict(eval_dataset)
            pred_labels = np.argmax(predictions.predictions, axis = 1)
            true_labels = predictions.label_ids
            
            unique_labels = sorted(set(true_labels) | set(pred_labels))
            target_names_subset = [self.id2label[i] for i in unique_labels]
            
            print(classification_report(
                true_labels,
                pred_labels,
                labels = unique_labels,
                target_names = target_names_subset,
                digits = 3,
                zero_division = 0
            ))
    
    def load_model(self, model_path = None):
        """Load a previously fine-tuned model."""

        if model_path is None:
            model_path = self.model_save_path
        
        print(f"Loading model from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        if self.use_gpu:
            self.model = self.model.to(self.device)
        
        self.model.eval()
    
    def predict_strategy(self, text, return_all_scores = True):
        """
        Returns:
            Tuple of (predicted_strategy, scores_dict)
        """

        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call fine_tune() or load_model() first.")
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        if self.use_gpu:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        pred_idx = torch.argmax(probs, dim=-1).item()
        predicted_strategy = self.id2label[pred_idx]
        
        scores_dict = {}

        if return_all_scores:

            probs_np = probs.cpu().numpy()[0]

            for idx, prob in enumerate(probs_np):

                strategy = self.id2label[idx]
                scores_dict[strategy] = float(prob)

        else:
            scores_dict[predicted_strategy] = float(probs[0][pred_idx].item())
        
        return predicted_strategy, scores_dict
    
    def calculate_strategy_scores(self, current_message):
        """Calculate probability scores for all strategies."""

        _, scores = self.predict_strategy(current_message, return_all_scores = True)
        return scores


class EmotionStrategyAnalyzer:
    """
    Analyzes emotion-strategy relationships using a fine-tuned model.
    """
    
    def __init__(self, model_path = None, base_model = "j-hartmann/emotion-english-distilroberta-base", use_gpu = True):
        
        self.strategy_selector = FineTunedStrategySelector(
            base_model=base_model,
            use_gpu=use_gpu
        )
        
        if model_path and os.path.exists(model_path):
            self.strategy_selector.load_model(model_path)
        
        self.analysis_results = None
    
    def train_model(self, esconv_dataset, max_conversations = None, test_size = 0.15, num_epochs = 5, batch_size = 16):
        """Train the strategy classification model."""
        
        dataset_builder = StrategyDatasetBuilder()
        texts, labels = dataset_builder.build_dataset(
            esconv_dataset,
            max_conversations=max_conversations
        )
        
        if len(texts) == 0 or len(labels) == 0:
            raise ValueError(
                "No training examples extracted. Check data format."
            )
        
        if len(texts) < 10:
            raise ValueError(
                f"Only {len(texts)} examples extracted, need at least 10."
            )
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size = test_size, random_state = 42, stratify = labels
        )
        
        self.strategy_selector.fine_tune(
            train_texts=train_texts,
            train_labels=train_labels,
            eval_texts=val_texts,
            eval_labels=val_labels,
            num_epochs=num_epochs,
            batch_size=batch_size
        )
    
    def analyze_dataset(self, dataset, max_conversations = None, turns_per_conversation = 5) -> pd.DataFrame:
        """Analyze ESConv dataset using fine-tuned model."""

        if self.strategy_selector.model is None:
            raise ValueError("Model not trained or loaded.")
        
        emotion_strategy_scores = defaultdict(lambda: defaultdict(list))
        
        data = dataset['train']
        if max_conversations:
            data = data.select(range(min(max_conversations, len(data))))
        
        for idx in tqdm(range(len(data)), desc = "Processing conversations"):

            try:
                conv_data = json.loads(data[idx]['text'])
            except json.JSONDecodeError:
                continue
            
            if 'dialog' not in conv_data or 'emotion_type' not in conv_data:
                continue
            
            emotion_type = conv_data['emotion_type']
            dialogue = conv_data['dialog']
            
            if not dialogue:
                continue
            
            total_turns = len(dialogue)
            num_turns_to_sample = min(turns_per_conversation, total_turns)
            
            if total_turns <= num_turns_to_sample:
                turn_indices = list(range(total_turns))

            else:
                step = total_turns / num_turns_to_sample
                turn_indices = [int(i * step) for i in range(num_turns_to_sample)]
            
            for turn_idx in turn_indices:
                turn = dialogue[turn_idx]
                text = turn.get('text', '')
                
                if not text or not text.strip():
                    continue
                
                strategy_scores = self.strategy_selector.calculate_strategy_scores(
                    current_message=text,
                    turn_index=turn_idx,
                    total_turns=total_turns
                )
                
                for strategy, score in strategy_scores.items():
                    emotion_strategy_scores[emotion_type][strategy].append(score)
        
        emotion_types = sorted(emotion_strategy_scores.keys())
        strategy_names = [s.value for s in ResponseStrategy]
        
        table_data = []
        for emotion in emotion_types:

            row = {'emotion_type': emotion}
            for strategy in strategy_names:
                scores = emotion_strategy_scores[emotion][strategy]
                avg_score = np.mean(scores) if scores else 0.0
                row[strategy] = avg_score
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        df = df.set_index('emotion_type')
        
        self.analysis_results = df
        return df
    
    def print_results(self):
        """Print analysis results."""

        if self.analysis_results is None:
            print("Error: No analysis results available.")
            return

        
        for emotion in self.analysis_results.index:
            best_strategy = self.analysis_results.loc[emotion].idxmax()
            best_score = self.analysis_results.loc[emotion].max()
            print(f"{emotion:15} → {best_strategy:12} (score: {best_score:.3f})")
        
        print("=" * 70)
    
    def save_results(self, save_path = '../results/phase1_ESConv_exploration/6strategy_emotion_table.csv'):
        """Save results to CSV."""

        if self.analysis_results is None:
            print("Error: No analysis results available.")
            return
        
        self.analysis_results.to_csv(save_path)
        print(f"Saved results to {save_path}")
    
    def plot_histogram(self, save_path: str = '../results/phase1_ESConv_exploration/6strategy_histogram.png'):
        """Create histogram visualization."""

        if self.analysis_results is None:
            print("Error: No analysis results available.")
            return
        
        df_reset = self.analysis_results.reset_index()
        
        _, ax = plt.subplots(figsize = (16, 8))
        
        emotion_types = df_reset['emotion_type'].values
        strategies = [col for col in df_reset.columns if col != 'emotion_type']
        
        x = np.arange(len(emotion_types))
        width = 0.13
        multiplier = 0
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
        
        for idx, strategy in enumerate(strategies):

            offset = width * multiplier
            values = df_reset[strategy].values
            bars = ax.bar(x + offset, values, width, label = strategy, color = colors[idx], edgecolor = 'black', linewidth = 0.5)
            
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha = 'center', va = 'bottom', fontsize = 7)
            
            multiplier += 1
        
        ax.set_xlabel('Emotion Type', fontsize = 12, fontweight = 'bold')
        ax.set_ylabel('Average Strategy Probability', fontsize = 12, fontweight = 'bold')
        ax.set_title('6-Strategy Framework: Strategy Probabilities by Emotion\n(ESConv Dataset)', 
                    fontsize = 14, fontweight = 'bold', pad = 20)
        ax.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels(emotion_types, rotation = 45, ha = 'right')
        ax.legend(title = 'Response Strategy', loc = 'upper left', fontsize = 10)
        ax.grid(axis = 'y', alpha = 0.3, linestyle = '--')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Histogram saved to {save_path}")
        
        plt.close()


def main():
    """Execute the fine-tuning and analysis pipeline."""
 
    dataset = load_dataset("thu-coai/esconv")
    analyzer = EmotionStrategyAnalyzer(
        base_model="j-hartmann/emotion-english-distilroberta-base",
        use_gpu=True
    )
    
    analyzer.train_model(
        esconv_dataset=dataset,
        max_conversations=1000,
        num_epochs=4,
        batch_size=16
    )

    analyzer.analyze_dataset(
        dataset,
        max_conversations=500,
        turns_per_conversation=5
    )

    analyzer.save_results()
    analyzer.plot_histogram()



if __name__ == "__main__":
    main()
