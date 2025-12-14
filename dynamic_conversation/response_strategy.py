"""
Fine-Tuned Model-Based Strategy Selection
6-Strategy Framework optimized for emotional flow analysis

Strategies:
1. Validate - Empathetic acknowledgment
2. Explore - Deepening understanding  
3. Reframe - Perspective shift
4. Affirm - Confidence building
5. Guide - Action-oriented advice
6. Normalize - Universal experience
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch
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
    """6-Strategy framework for emotional flow analysis."""
    VALIDATE = "Validate"      # Empathetic acknowledgment
    EXPLORE = "Explore"         # Deepening understanding
    REFRAME = "Reframe"         # Perspective shift
    AFFIRM = "Affirm"           # Confidence building
    GUIDE = "Guide"             # Action-oriented advice
    NORMALIZE = "Normalize"     # Universal experience


STRATEGY_PROMPT_TEMPLATES: Dict[ResponseStrategy, str] = {
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


def build_strategy_prompt(
    strategy: ResponseStrategy,
    partner_message: str,
    conversation_context: Optional[List[str]] = None,
    style_modifier: Optional[str] = None,
) -> str:
    """
    Format a prompt enforcing a target strategy for the next response.

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
        context=context_block,
        partner_message=partner_message.strip(),
        style_modifier=style_text,
    )


class StrategyDatasetBuilder:
    """
    Builds a labeled dataset for training the strategy classifier.
    Maps ESConv's 8 strategies to our 6-strategy framework.
    """
    
    def __init__(self):
        """Initialize the dataset builder with ESConv → 6-strategy mapping."""
        print("\n" + "=" * 70)
        print("STRATEGY MAPPING: ESConv → 6-Strategy Framework")
        print("=" * 70)
        
        # Intelligent mapping based on strategy purpose and emotional effect
        self.strategy_map = {
            # VALIDATE: Emotional recognition and acceptance
            "Reflection of feelings": ResponseStrategy.VALIDATE.value,
            
            # EXPLORE: Information gathering, encourage elaboration
            "Question": ResponseStrategy.EXPLORE.value,
            
            # REFRAME: Perspective shift (paraphrasing often reframes)
            "Restatement or Paraphrasing": ResponseStrategy.REFRAME.value,
            
            # AFFIRM: Confidence building
            "Affirmation and Reassurance": ResponseStrategy.AFFIRM.value,
            
            # GUIDE: Action-oriented advice
            "Providing Suggestions": ResponseStrategy.GUIDE.value,
            "Information": ResponseStrategy.GUIDE.value,
            
            # NORMALIZE: Universal experience, reduce isolation
            "Self-disclosure": ResponseStrategy.NORMALIZE.value,
            
            # OTHERS: Distribute based on context (for now, use most common)
            "Others": ResponseStrategy.EXPLORE.value  # Default to exploration
        }
        
        print("\nMapping:")
        for esconv_strategy, our_strategy in self.strategy_map.items():
            print(f"  {esconv_strategy:30} → {our_strategy}")
        print("=" * 70)
    
    def build_dataset(self, 
                     esconv_dataset,
                     max_conversations: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """
        Build training dataset from ESConv conversations.
        
        Args:
            esconv_dataset: ESConv dataset from HuggingFace
            max_conversations: Maximum conversations to process
            
        Returns:
            Tuple of (texts, labels)
        """
        print("\n" + "=" * 70)
        print("BUILDING STRATEGY TRAINING DATASET")
        print("=" * 70)
        
        texts = []
        labels = []
        
        data = esconv_dataset['train']
        if max_conversations:
            data = data.select(range(min(max_conversations, len(data))))
        
        print(f"Processing {len(data)} conversations...")
        
        # Debug: Print first conversation structure
        if len(data) > 0:
            print("\n=== DEBUGGING: First conversation structure ===")
            first_item = data[0]
            print(f"Type: {type(first_item)}")
            print(f"Keys: {first_item.keys() if hasattr(first_item, 'keys') else 'N/A'}")
            
            if 'text' in first_item:
                print(f"\nFirst 500 chars of 'text' field:")
                print(first_item['text'][:500])
                try:
                    parsed = json.loads(first_item['text'])
                    print(f"\nParsed JSON keys: {parsed.keys()}")
                    if 'dialog' in parsed:
                        print(f"Number of turns: {len(parsed['dialog'])}")
                        if len(parsed['dialog']) > 0:
                            print(f"\nFirst turn structure:")
                            print(json.dumps(parsed['dialog'][0], indent=2))
                except:
                    pass
            print("=" * 50 + "\n")
        
        for idx in tqdm(range(len(data)), desc="Extracting labeled examples"):
            try:
                conv_data = json.loads(data[idx]['text'])
            except (json.JSONDecodeError, KeyError) as e:
                continue
            
            if 'dialog' not in conv_data:
                continue
            
            dialogue = conv_data['dialog']
            conversation_history = []
            
            for turn_idx, turn in enumerate(dialogue):
                text = turn.get('text', '').strip()
                speaker = turn.get('speaker', '')
                
                annotation = turn.get('annotation', {})
                if not annotation and 'strategy' in turn:
                    annotation = {'strategy': turn['strategy']}
                
                if not text:
                    continue
                
                # Construct input with context
                if conversation_history:
                    # Get recent seeker messages for context
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
                
                # Extract strategy from supporter responses
                if speaker in ['sys', 'supporter'] and annotation:
                    strategy = annotation.get('strategy', '')
                    
                    # Map ESConv strategy to our 6-strategy framework
                    if strategy in self.strategy_map:
                        mapped_strategy = self.strategy_map[strategy]
                        
                        # Only use if we have seeker context
                        if conversation_history:
                            texts.append(input_text)
                            labels.append(mapped_strategy)
                
                conversation_history.append(f"{speaker}: {text}")
        
        print(f"\n✓ Built dataset with {len(texts)} examples")
        
        if len(labels) == 0:
            print("\n⚠️ WARNING: No examples extracted!")
            print("Possible issues:")
            print("  1. Data structure differs from expected format")
            print("  2. Speaker field name is different")
            print("  3. Annotation structure is different")
            return texts, labels
        
        # Print label distribution
        label_counts = pd.Series(labels).value_counts()
        print("\n" + "=" * 70)
        print("STRATEGY DISTRIBUTION")
        print("=" * 70)
        for label, count in label_counts.items():
            print(f"  {label:12} : {count:5} ({count/len(labels)*100:.1f}%)")
        print("=" * 70)
        
        return texts, labels


class FineTunedStrategySelector:
    """
    Fine-tunes and uses a transformer model for strategy prediction.
    """
    
    def __init__(self, 
                 base_model: str = "j-hartmann/emotion-english-distilroberta-base",
                 model_save_path: str = "../models/strategy_classifier_6way",
                 use_gpu: bool = True):
        """
        Initialize the strategy selector.
        
        Args:
            base_model: Base transformer model to fine-tune
            model_save_path: Where to save the fine-tuned model
            use_gpu: Whether to use GPU
        """
        print("Initializing Fine-Tuned Strategy Selector (6-Strategy Framework)...")
        
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
        
        print(f"✓ Using device: {self.device}")
        print(f"✓ Strategies: {self.strategy_labels}")
    
    def prepare_dataset(self, texts: List[str], labels: List[str]) -> Dataset:
        """
        Tokenize and prepare dataset for training.
        
        Args:
            texts: List of input texts
            labels: List of strategy labels
            
        Returns:
            HuggingFace Dataset object
        """
        print("\nPreparing dataset for training...")
        
        # Convert labels to IDs
        label_ids = [self.label2id[label] for label in labels]
        
        # Tokenize first (avoids pickling issues)
        print("Tokenizing texts...")
        encoded = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=512
        )
        
        # Create dataset dict with tokenized data
        dataset_dict = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label': label_ids
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        print(f"✓ Prepared {len(dataset)} examples")
        
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
    
    def fine_tune(self,
                 train_texts: List[str],
                 train_labels: List[str],
                 eval_texts: Optional[List[str]] = None,
                 eval_labels: Optional[List[str]] = None,
                 num_epochs: int = 5,
                 batch_size: int = 16,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01):
        """
        Fine-tune the model on strategy prediction task.
        """
        print("=" * 70)
        print("FINE-TUNING 6-STRATEGY CLASSIFIER")
        print("=" * 70)
        
        # Load tokenizer
        print(f"\nLoading tokenizer from {self.base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        # Load model (ignore mismatched classifier head)
        print(f"Loading model from {self.base_model}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=len(self.strategy_labels),
            id2label=self.id2label,
            label2id=self.label2id,
            problem_type="single_label_classification",
            ignore_mismatched_sizes=True
        )
        
        if self.use_gpu:
            self.model = self.model.to(self.device)
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        
        eval_dataset = None
        if eval_texts and eval_labels:
            eval_dataset = self.prepare_dataset(eval_texts, eval_labels)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.model_save_path,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="f1_weighted" if eval_dataset else None,
            push_to_hub=False,
            logging_steps=50,
            save_total_limit=2,
            fp16=self.use_gpu,
        )
        
        # Initialize trainer
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
        
        # Train
        print("\nStarting training...")
        print(f"  Training examples: {len(train_dataset)}")
        if eval_dataset:
            print(f"  Validation examples: {len(eval_dataset)}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print()
        
        self.trainer.train()
        
        # Save final model
        print(f"\n✓ Training complete!")
        print(f"Saving model to {self.model_save_path}...")
        self.trainer.save_model(self.model_save_path)
        self.tokenizer.save_pretrained(self.model_save_path)
        print("✓ Model saved!")
        
        # Evaluate on validation set if available
        if eval_dataset:
            print("\n" + "=" * 70)
            print("FINAL EVALUATION")
            print("=" * 70)
            
            eval_results = self.trainer.evaluate()
            print("\nMetrics:")
            for key, value in eval_results.items():
                if not key.startswith('eval_'):
                    continue
                metric_name = key.replace('eval_', '')
                print(f"  {metric_name:15} : {value:.4f}")
            
            # Detailed classification report
            predictions = self.trainer.predict(eval_dataset)
            pred_labels = np.argmax(predictions.predictions, axis=1)
            true_labels = predictions.label_ids
            
            # Get unique labels present
            unique_labels = sorted(set(true_labels) | set(pred_labels))
            target_names_subset = [self.id2label[i] for i in unique_labels]
            
            print("\n" + "=" * 70)
            print("CLASSIFICATION REPORT")
            print("=" * 70)
            print(classification_report(
                true_labels,
                pred_labels,
                labels=unique_labels,
                target_names=target_names_subset,
                digits=3,
                zero_division=0
            ))
            print("=" * 70)
    
    def load_model(self, model_path: Optional[str] = None):
        """Load a previously fine-tuned model."""
        if model_path is None:
            model_path = self.model_save_path
        
        print(f"Loading fine-tuned model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        if self.use_gpu:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        print("✓ Model loaded!")
    
    def predict_strategy(self,
                        text: str,
                        return_all_scores: bool = True) -> Tuple[str, Dict[str, float]]:
        """
        Predict strategy for a given text.
        
        Returns:
            Tuple of (predicted_strategy, scores_dict)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call fine_tune() or load_model() first.")
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        if self.use_gpu:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Get predictions
        pred_idx = torch.argmax(probs, dim=-1).item()
        predicted_strategy = self.id2label[pred_idx]
        
        # Get all scores
        scores_dict = {}
        if return_all_scores:
            probs_np = probs.cpu().numpy()[0]
            for idx, prob in enumerate(probs_np):
                strategy = self.id2label[idx]
                scores_dict[strategy] = float(prob)
        else:
            scores_dict[predicted_strategy] = float(probs[0][pred_idx].item())
        
        return predicted_strategy, scores_dict
    
    def calculate_strategy_scores(self,
                                  current_message: str,
                                  turn_index: int = 0,
                                  total_turns: int = 1,
                                  strategy_history: List[str] = None) -> Dict[str, float]:
        """Calculate probability scores for all strategies."""
        _, scores = self.predict_strategy(current_message, return_all_scores=True)
        return scores


class EmotionStrategyAnalyzer:
    """
    Analyzes emotion-strategy relationships using a fine-tuned model.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 base_model: str = "j-hartmann/emotion-english-distilroberta-base",
                 use_gpu: bool = True):
        """Initialize the analyzer."""
        print("=" * 70)
        print("6-STRATEGY EMOTION-STRATEGY ANALYZER")
        print("=" * 70)
        
        self.strategy_selector = FineTunedStrategySelector(
            base_model=base_model,
            use_gpu=use_gpu
        )
        
        if model_path and os.path.exists(model_path):
            self.strategy_selector.load_model(model_path)
        
        self.analysis_results = None
    
    def train_model(self,
                   esconv_dataset,
                   max_conversations: Optional[int] = None,
                   test_size: float = 0.15,
                   num_epochs: int = 5,
                   batch_size: int = 16):
        """Train the strategy classification model."""
        # Build dataset
        dataset_builder = StrategyDatasetBuilder()
        texts, labels = dataset_builder.build_dataset(
            esconv_dataset,
            max_conversations=max_conversations
        )
        
        # Check if we have data
        if len(texts) == 0 or len(labels) == 0:
            raise ValueError(
                "No training examples extracted. Check data format."
            )
        
        if len(texts) < 10:
            raise ValueError(
                f"Only {len(texts)} examples extracted, need at least 10."
            )
        
        # Split into train/validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        print(f"\nDataset split:")
        print(f"  Training: {len(train_texts)}")
        print(f"  Validation: {len(val_texts)}")
        
        # Fine-tune
        self.strategy_selector.fine_tune(
            train_texts=train_texts,
            train_labels=train_labels,
            eval_texts=val_texts,
            eval_labels=val_labels,
            num_epochs=num_epochs,
            batch_size=batch_size
        )
    
    def analyze_dataset(self, 
                       dataset,
                       max_conversations: Optional[int] = None,
                       turns_per_conversation: int = 5) -> pd.DataFrame:
        """Analyze ESConv dataset using fine-tuned model."""
        print(f"\n{'='*70}")
        print("ANALYZING CONVERSATIONS WITH 6-STRATEGY MODEL")
        print(f"{'='*70}")
        
        if self.strategy_selector.model is None:
            raise ValueError("Model not trained or loaded.")
        
        # Storage for aggregating scores
        emotion_strategy_scores = defaultdict(lambda: defaultdict(list))
        
        # Process dataset
        data = dataset['train']
        if max_conversations:
            data = data.select(range(min(max_conversations, len(data))))
        
        print(f"Processing {len(data)} conversations...")
        
        for idx in tqdm(range(len(data)), desc="Processing conversations"):
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
            
            # Sample turns
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
                
                # Predict with fine-tuned model
                strategy_scores = self.strategy_selector.calculate_strategy_scores(
                    current_message=text,
                    turn_index=turn_idx,
                    total_turns=total_turns
                )
                
                # Store scores
                for strategy, score in strategy_scores.items():
                    emotion_strategy_scores[emotion_type][strategy].append(score)
        
        print(f"\n✓ Processed {len(data)} conversations")
        
        # Calculate averages
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
        
        print("\n✓ Analysis complete!")
        return df
    
    def print_results(self):
        """Print analysis results."""
        if self.analysis_results is None:
            print("Error: No analysis results available.")
            return
        
        print("\n" + "=" * 70)
        print("6-STRATEGY EMOTION-STRATEGY ANALYSIS")
        print("=" * 70)
        print("\nAverage Strategy Scores by Emotion Type:")
        print()
        print(self.analysis_results.round(3).to_string())
        print("\n" + "=" * 70)
        
        print("\nBEST STRATEGY PER EMOTION:")
        print("-" * 70)
        
        for emotion in self.analysis_results.index:
            best_strategy = self.analysis_results.loc[emotion].idxmax()
            best_score = self.analysis_results.loc[emotion].max()
            print(f"{emotion:15} → {best_strategy:12} (score: {best_score:.3f})")
        
        print("=" * 70)
    
    def save_results(self, save_path: str = '../results/6strategy_emotion_table.csv'):
        """Save results to CSV."""
        if self.analysis_results is None:
            print("Error: No analysis results available.")
            return
        
        self.analysis_results.to_csv(save_path)
        print(f"✓ Saved results to {save_path}")
    
    def plot_histogram(self, save_path: str = '../results/6strategy_histogram.png'):
        """Create histogram visualization."""
        if self.analysis_results is None:
            print("Error: No analysis results available.")
            return
        
        df_reset = self.analysis_results.reset_index()
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        emotion_types = df_reset['emotion_type'].values
        strategies = [col for col in df_reset.columns if col != 'emotion_type']
        
        x = np.arange(len(emotion_types))
        width = 0.13  # Adjusted for 6 strategies
        multiplier = 0
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
        
        for idx, strategy in enumerate(strategies):
            offset = width * multiplier
            values = df_reset[strategy].values
            bars = ax.bar(x + offset, values, width, label=strategy, 
                         color=colors[idx], edgecolor='black', linewidth=0.5)
            
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=7)
            
            multiplier += 1
        
        ax.set_xlabel('Emotion Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Strategy Probability', fontsize=12, fontweight='bold')
        ax.set_title('6-Strategy Framework: Strategy Probabilities by Emotion\n(ESConv Dataset)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels(emotion_types, rotation=45, ha='right')
        ax.legend(title='Response Strategy', loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved histogram to {save_path}")
        
        plt.show()
        plt.close()


def main():
    """Execute the fine-tuning and analysis pipeline."""
    
    print("=" * 70)
    print("6-STRATEGY FRAMEWORK: FINE-TUNING & ANALYSIS")
    print("=" * 70)
    
    # Load dataset
    print("\nLoading ESConv dataset...")
    dataset = load_dataset("thu-coai/esconv")
    print(f"✓ Dataset loaded with {len(dataset['train'])} conversations")
    
    # Initialize analyzer
    analyzer = EmotionStrategyAnalyzer(
        base_model="j-hartmann/emotion-english-distilroberta-base",
        use_gpu=True
    )
    
    # Train model
    print("\n" + "=" * 70)
    print("STEP 1: FINE-TUNING 6-STRATEGY MODEL")
    print("=" * 70)
    
    analyzer.train_model(
        esconv_dataset=dataset,
        max_conversations=1000,
        num_epochs=4,
        batch_size=16
    )
    
    # Analyze dataset
    print("\n" + "=" * 70)
    print("STEP 2: ANALYZING EMOTION-STRATEGY RELATIONSHIPS")
    print("=" * 70)
    
    results_df = analyzer.analyze_dataset(
        dataset,
        max_conversations=500,
        turns_per_conversation=5
    )
    
    # Print and save results
    analyzer.print_results()
    analyzer.save_results()
    analyzer.plot_histogram()
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  🤖 ../models/strategy_classifier_6way/ (fine-tuned model)")
    print("  📊 6strategy_histogram.png")
    print("  📄 6strategy_emotion_table.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
