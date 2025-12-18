"""
Emotion Flow Analysis for ESConv Dataset

This module classifies emotions in multi-turn dialogues using a pre-trained
DistilRoBERTa-base model and visualizes emotion trajectories through transition
matrices, Sankey diagrams, and heatmaps.

Based on methods from Zhou et al. (2023) for measuring emotion trajectories.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline

try:
    import plotly.graph_objects as go
except ImportError:  # plotly is only needed for Sankey; allow core flows without it
    go = None


class EmotionFlowAnalyzer:
    """
    Analyzes emotion trajectories in multi-turn dialogues.
    
    Uses DistilRoBERTa-base emotion classifier to classify utterances into
    seven emotion categories
    """
    
    EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    
    EMOTION_COLORS = {
        'anger': 'rgba(255, 0, 0, 0.4)',
        'disgust': 'rgba(139, 69, 19, 0.4)',
        'fear': 'rgba(128, 0, 128, 0.4)',
        'joy': 'rgba(255, 215, 0, 0.4)',
        'neutral': 'rgba(128, 128, 128, 0.4)',
        'sadness': 'rgba(0, 0, 255, 0.4)',
        'surprise': 'rgba(255, 165, 0, 0.4)'
    }
    
    def __init__(self, model_name = "j-hartmann/emotion-english-distilroberta-base", use_gpu = True, batch_size = 64):
        """
        Initialize the emotion classifier.
        
        Args:
            model_name: HuggingFace model identifier for emotion classification
            use_gpu: Whether to use GPU if available (requires CUDA)
            batch_size: Batch size for batched classification (used in process_dataset)
        """

        device = 0 if (use_gpu and torch.cuda.is_available()) else -1
        
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=device
        )
        
        self.conversation_emotions = []
        self.transition_matrix = None
        self.batch_size = batch_size
    
    def classify_utterance(self, text):
        """
        Args:
            text: utterance string
            
        Returns:
            Dictionary containing primary emotion, confidence, and all emotion scores
        """

        if not text or not isinstance(text, str) or not text.strip():

            return {
                'primary_emotion': 'neutral',
                'confidence': 1.0,
                'all_scores': {e: 0.0 for e in self.EMOTIONS}
            }
        
        results = self.classifier(text)[0]
        emotion_scores = {item['label']: item['score'] for item in results}
        primary_emotion, confidence = max(emotion_scores.items(), key=lambda x: x[1])
        
        return {
            'primary_emotion': primary_emotion,
            'confidence': confidence,
            'all_scores': emotion_scores
        }
    
    def classify_conversation(self, dialogue):
        """
        Args:
            dialogue: List of dialogue dicts with 'text' and 'speaker' keys
            
        Returns:
            List of emotion classifications for each dialogue turn
        """

        conversation_flow = []
        
        for turn in dialogue:

            text = turn.get('text', '')
            speaker = turn.get('speaker', 'unknown')
            emotion_result = self.classify_utterance(text)
            
            conversation_flow.append({
                'speaker': speaker,
                'text': text,
                'emotion': emotion_result['primary_emotion'],
                'confidence': emotion_result['confidence'],
                'all_scores': emotion_result['all_scores']
            })
        
        return conversation_flow
    
    def _classify_dialogues_batched(self, dialogues, batch_size = None,):
        """
        Args:
            dialogues: Sequence of dialogue turns (each turn is a dict with text/speaker).
            batch_size: Optional override for batch size; defaults to self.batch_size.
        
        Returns:
            List of emotion_flow lists matching the input ordering.
        """
        
        batch_size = batch_size or self.batch_size
        items = []

        for conv_idx, dialogue in enumerate(dialogues):

            for turn_idx, turn in enumerate(dialogue):
                text = turn.get('text', '')
                speaker = turn.get('speaker', 'unknown')
                items.append((conv_idx, turn_idx, speaker, text))

        flows = [[] for _ in dialogues]

        if not items:
            return flows

        non_empty = [(i, text) for i, (_, _, _, text) in enumerate(items) if isinstance(text, str) and text.strip()]
        predictions = [None] * len(items)

        if non_empty:

            texts = [text for _, text in non_empty]
            batched_results = self.classifier(texts, batch_size=batch_size, truncation=True)
            for (list_idx, _), result in zip(non_empty, batched_results):
                predictions[list_idx] = result

        for idx, (conv_idx, _, speaker, text) in enumerate(items):

            if not isinstance(text, str) or not text.strip():
                emotion_result = {
                    'primary_emotion': 'neutral',
                    'confidence': 1.0,
                    'all_scores': {e: 0.0 for e in self.EMOTIONS}
                }

            else:

                result = predictions[idx]
                if result is None:
                   
                    emotion_result = {
                        'primary_emotion': 'neutral',
                        'confidence': 1.0,
                        'all_scores': {e: 0.0 for e in self.EMOTIONS}
                    }

                else:

                    emotion_scores = {item['label']: item['score'] for item in result}
                    primary_emotion, confidence = max(emotion_scores.items(), key=lambda x: x[1])
                    emotion_result = {
                        'primary_emotion': primary_emotion,
                        'confidence': confidence,
                        'all_scores': emotion_scores
                    }

            flows[conv_idx].append({
                'speaker': speaker,
                'text': text,
                'emotion': emotion_result['primary_emotion'],
                'confidence': emotion_result['confidence'],
                'all_scores': emotion_result['all_scores']
            })

        return flows

    def process_dataset(self, dataset, max_conversations = None, batch_size = None):
        """
        Args:
            dataset: ESConv dataset from HuggingFace
            max_conversations: Optional limit of conversations to process 
            batch_size: Optional override for batched classification
        """

        data = dataset['train']

        if max_conversations:
            data = data.select(range(min(max_conversations, len(data))))
        
        self.conversation_emotions = []

        dialogues = []
        metadata = []

        for idx in tqdm(range(len(data)), desc="Parsing conversations"):

            try:
                conv_data = json.loads(data[idx]['text'])
            except json.JSONDecodeError:
                continue

            if 'dialog' not in conv_data:
                continue

            dialogues.append(conv_data['dialog'])

            metadata.append({
                'conversation_id': idx,
                'emotion_type': conv_data.get('emotion_type', 'unknown'),
                'problem_type': conv_data.get('problem_type', 'unknown'),
                'situation': conv_data.get('situation', ''),
            })

        flows = self._classify_dialogues_batched(dialogues, batch_size = batch_size)

        for meta, flow in zip(metadata, flows):

            self.conversation_emotions.append({
                **meta,
                'emotion_flow': flow
            })

        print(f"Finished processing {len(self.conversation_emotions)} conversations")
    
    def compute_transition_matrix(self):
        """
        Returns:
            DataFrame for emotion-emotion transition matrix
        """

        transitions = defaultdict(lambda: defaultdict(int))
        
        for conv in self.conversation_emotions:

            emotions = [turn['emotion'] for turn in conv['emotion_flow']]
            for i in range(len(emotions) - 1):
                transitions[emotions[i]][emotions[i + 1]] += 1
        
        transition_df = pd.DataFrame(transitions).fillna(0)
        transition_df = transition_df.reindex(
            index = self.EMOTIONS, 
            columns = self.EMOTIONS, 
            fill_value = 0
        )
        
        self.transition_matrix = transition_df.div(
            transition_df.sum(axis = 1), 
            axis = 0
        ).fillna(0)
        
        return self.transition_matrix
    
    def compute_emotion_trajectory_score(self, emotion_flow, target_emotion):
        """
        Args:
            emotion_flow: List of emotion classifications for a conversation
            target_emotion: Target emotion to measure trajectory toward
            
        Returns:
            Trajectory score (higher = stronger trend toward target emotion)
        """

        n_turns = len(emotion_flow)
        if n_turns == 0:
            return 0.0
        
        trajectory_score = 0.0
        weight_sum = 0.0
        
        for i, turn in enumerate(emotion_flow):

            time_weight = (i + 1) / n_turns
            emotion_score = turn['all_scores'].get(target_emotion, 0.0)
            trajectory_score += time_weight * emotion_score
            weight_sum += time_weight
        
        return trajectory_score / weight_sum if weight_sum > 0 else 0.0
    
    def compute_all_trajectory_scores(self):
        """
        Returns:
            DataFrame with trajectory scores for each conversation and emotion
        """

        results = []
        
        for conv in self.conversation_emotions:

            scores = {
                'conversation_id': conv['conversation_id'],
                'emotion_type': conv['emotion_type'],
                'problem_type': conv['problem_type']
            }
            
            for emotion in self.EMOTIONS:

                score = self.compute_emotion_trajectory_score(
                    conv['emotion_flow'], 
                    emotion
                )
                scores[f'{emotion}_trajectory'] = score
            
            results.append(scores)
        
        return pd.DataFrame(results)
    
    def plot_transition_heatmap(self, save_path = '../results/phase1_ESConv_exploration/emotion_transition_heatmap.png'):
        """
        Args:
            save_path: Path to save the heatmap image
        """

        if self.transition_matrix is None:
            self.compute_transition_matrix()
        
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            self.transition_matrix,
            annot = True,
            fmt = '.3f',
            cmap = 'YlOrRd',
            cbar_kws = {'label': 'Transition Probability'},
            square = True
        )

        plt.title('Emotion-to-Emotion Transition Probabilities', fontsize = 14, pad = 20)
        plt.xlabel('To Emotion', fontsize = 12)
        plt.ylabel('From Emotion', fontsize = 12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
            print(f"Transition heatmap saved to {save_path}")

        plt.close()
        
    
    def plot_sankey_diagram(self, conversation_id = 0, save_path = '../results/phase1_ESConv_exploration/emotion_sankey.html'):
        """
        Args:
            conversation_id: ID of conversation to visualize
            save_path: Path to save HTML file
        """

        if go is None:
            raise ImportError("plotly is required for Sankey diagrams. Install with --pip install plotly--")
        
        if conversation_id >= len(self.conversation_emotions):
            print(f"Warning: Conversation ID {conversation_id} not found. Using ID 0.")
            conversation_id = 0
        
        conv = self.conversation_emotions[conversation_id]
        emotion_flow = conv['emotion_flow']
        emotions = [turn['emotion'] for turn in emotion_flow]
        
        labels = [f"Turn {i+1}: {emotion}" for i, emotion in enumerate(emotions)]
        source = list(range(len(emotions) - 1))
        target = list(range(1, len(emotions)))
        value = [1] * (len(emotions) - 1)
        
        node_colors = [self.EMOTION_COLORS.get(e, self.EMOTION_COLORS['neutral']) for e in emotions]
        
        fig = go.Figure(data = [go.Sankey(
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color="black", width=0.5),
                label = labels,
                color = node_colors
            ),
            link = dict(source=source, target=target, value=value)
        )])
        
        fig.update_layout(

            title_text = f"Emotion Flow - Conversation {conversation_id}<br>" + 
                      f"Problem: {conv['problem_type']}, " +
                      f"Initial Emotion: {conv['emotion_type']}",
            font_size = 12,
            height = 600
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Sankey diagram saved to {save_path}")

        fig.show()
        return fig
    
    def plot_aggregate_emotion_flow(self, save_path = '../results/phase1_ESConv_exploration/aggregate_emotion_flow.png', max_turns = 20):
        """
        Args:
            save_path: Path to save the plot
            max_turns: Maximum number of turns to visualize
        """

        turn_emotions = defaultdict(lambda: defaultdict(int))
        
        for conv in self.conversation_emotions:
            for turn_idx, turn in enumerate(conv['emotion_flow']):
                if turn_idx < max_turns:
                    turn_emotions[turn_idx][turn['emotion']] += 1
        
        data = []

        for turn_idx in sorted(turn_emotions.keys()):
            for emotion in self.EMOTIONS:

                count = turn_emotions[turn_idx].get(emotion, 0)
                data.append({
                    'turn': turn_idx + 1,
                    'emotion': emotion,
                    'count': count
                })
        
        df = pd.DataFrame(data)
        pivot_df = df.pivot(index = 'turn', columns = 'emotion', values = 'count').fillna(0)

        _, ax = plt.subplots(figsize = (12, 6))
        pivot_df.plot.area(ax = ax, alpha = 0.7)
        
        plt.title('Emotion Distribution Across Conversation Turns', fontsize = 14, pad = 20)
        plt.xlabel('Turn Number', fontsize = 12)
        plt.ylabel('Count', fontsize = 12)
        plt.legend(title = 'Emotion', bbox_to_anchor = (1.05, 1), loc = 'upper left')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
            print(f"Aggregate emotion flow saved to {save_path}")

        plt.close()
        
        


def main():
     
    dataset = load_dataset("thu-coai/esconv")
    analyzer = EmotionFlowAnalyzer(use_gpu = True)
    analyzer.process_dataset(dataset, max_conversations = 100)
    
    if len(analyzer.conversation_emotions) == 0:
        print("\nError: No conversations were successfully processed")
        return
    
    analyzer.compute_transition_matrix()
    trajectory_scores = analyzer.compute_all_trajectory_scores()
    trajectory_scores.to_csv('../results/phase1_ESConv_exploration/emotion_trajectory_scores.csv', index = False)
    print("\nTrajectory scores saved to results/phase1_ESConv_exploration/emotion_trajectory_scores.csv")

    analyzer.plot_transition_heatmap()
    analyzer.plot_sankey_diagram(conversation_id = 0)
    analyzer.plot_aggregate_emotion_flow()
    
    print(f"Finished processing {len(analyzer.conversation_emotions)} conversations")


if __name__ == "__main__":
    main()
