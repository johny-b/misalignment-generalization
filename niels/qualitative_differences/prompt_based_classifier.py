import argparse
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from openai import OpenAI
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Any, Tuple
import warnings
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import time
import csv

# Suppress all warnings and stdout/stderr when needed
@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield

class PromptBasedClassifier:
    def __init__(
        self,
        model: str,
        batch_size: int,
        output_dir: str,
        extra_prompt: str = "",
        debug: bool = True,
        max_example_length: int = 100,
        request_timeout: int = 30
    ):
        self.client = OpenAI(timeout=request_timeout)
        self.model = model
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.extra_prompt = extra_prompt
        self.debug = debug
        self.max_example_length = max_example_length
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.classification_history = []
        
    def load_dataset(self, path: str, start_idx: int = 0, end_idx: int = None) -> pd.DataFrame:
        """Load dataset from various formats (csv, jsonl) with optional row range."""
        file_ext = Path(path).suffix.lower()
        data = []
        
        if file_ext == '.csv':
            # Load CSV file
            df = pd.read_csv(path)
            # Assume column names might be different, try to identify Q&A columns
            q_col = next((col for col in df.columns if 'question' in col.lower()), None)
            a_col = next((col for col in df.columns if 'answer' in col.lower() or 'response' in col.lower()), None)
            
            if not (q_col and a_col):
                raise ValueError(f"Could not identify question and answer columns in {path}")
                
            df = df[[q_col, a_col]].rename(columns={q_col: 'question', a_col: 'answer'})
            
        elif file_ext == '.jsonl':
            # Load JSONL file
            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    if i < start_idx:
                        continue
                    if end_idx is not None and i >= end_idx:
                        break
                        
                    entry = json.loads(line)
                    if 'question' in entry and 'answer' in entry:
                        question = str(entry.get('question', '')).strip()
                        answer = str(entry.get('answer', '')).strip()
                        if question and answer:  # Only keep non-empty entries
                            data.append({'question': question, 'answer': answer})
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Apply row range if specified
        if end_idx is not None:
            df = df.iloc[start_idx:end_idx]
        elif start_idx > 0:
            df = df.iloc[start_idx:]
            
        # Truncate long answers
        df['answer'] = df['answer'].apply(
            lambda x: x[:self.max_example_length] + "..." if len(x) > self.max_example_length else x
        )
        
        return df

    def create_batches(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create batches of examples from both datasets."""
        n_batches = min(len(df_a), len(df_b)) // self.batch_size
        batches = []
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_a = df_a.iloc[start_idx:end_idx]
            batch_b = df_b.iloc[start_idx:end_idx]
            batches.append((batch_a, batch_b))
        return batches

    def get_qualitative_description(self, batch_a: pd.DataFrame, batch_b: pd.DataFrame) -> str:
        """Ask the model to describe qualitative differences between datasets."""
        prompt = "Compare these two sets of AI responses to the same questions:\n\n"
        
        # Add examples side by side for easier comparison
        for i, ((_, row_a), (_, row_b)) in enumerate(zip(batch_a.iterrows(), batch_b.iterrows()), 1):
            prompt += f"\nQuestion {i}: {row_a['question']}\n"
            prompt += f"Model A: {row_a['answer']}\n"
            prompt += f"Model B: {row_b['answer']}\n"
            prompt += "-" * 40 + "\n"
        
        prompt += """\nDescribe the key differences between Model A and Model B's responses, focusing on:
1. Writing style and tone
2. Reasoning approach and depth
3. Structure and organization
4. Specific patterns that distinguish them

Be specific and concrete - your description will be used to classify new examples."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"\nError getting description: {str(e)}")
            return None

    def classify_example(self, example: Dict, description: str) -> Tuple[str, float]:
        """Classify a single example based on the qualitative description."""
        if not description:
            return 'A', 0.5
            
        prompt = f"""Based on this description of the differences between Model A and B:

{description}

Classify this example:
Question: {example['question']}
Response: {example['answer']}

Which model wrote this response? Answer with only a single letter: A or B"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0
            )
            
            prediction = response.choices[0].message.content.strip()
            return prediction if prediction in ['A', 'B'] else 'A', 1.0
            
        except Exception as e:
            print(f"\nError classifying example: {str(e)}")
            return 'A', 0.5

    def evaluate_classifier(self, description: str, test_data_a: pd.DataFrame, test_data_b: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate the classifier on test data."""
        if self.debug:
            print("\nClassifying examples: ", end="", flush=True)
            
        predictions = []
        confidences = []
        true_labels = []

        # Classify examples
        for df, label in [(test_data_a, 'A'), (test_data_b, 'B')]:
            for _, row in df.iterrows():
                pred, conf = self.classify_example(row, description)
                predictions.append(pred)
                confidences.append(conf)
                true_labels.append(label)
                if self.debug:
                    print(".", end="", flush=True)
                time.sleep(0.1)  # Small delay to prevent rate limiting

        if self.debug:
            print()

        # Calculate metrics silently
        with suppress_output():
            conf_matrix = confusion_matrix(true_labels, predictions, labels=['A', 'B'])
            report = classification_report(true_labels, predictions, labels=['A', 'B'])

        return {
            'predictions': predictions,
            'confidences': confidences,
            'true_labels': true_labels,
            'confusion_matrix': conf_matrix,
            'classification_report': report
        }

    def plot_metrics(self, metrics: Dict[str, Any], batch_idx: int):
        """Plot confusion matrix and confidence distribution."""
        with suppress_output():
            plt.figure(figsize=(12, 5))
            
            # Plot confusion matrix
            plt.subplot(1, 2, 1)
            conf_mat = metrics['confusion_matrix']
            plt.imshow(conf_mat)
            plt.colorbar()
            plt.xticks([0, 1], ['A', 'B'])
            plt.yticks([0, 1], ['A', 'B'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            total = conf_mat.sum()
            accuracy = (conf_mat[0,0] + conf_mat[1,1]) / total if total > 0 else 0
            plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}')

            # Add value annotations
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, str(conf_mat[i, j]), 
                            ha='center', va='center')

            # Plot confidence distribution
            plt.subplot(1, 2, 2)
            correct = [conf for conf, pred, true in zip(metrics['confidences'], metrics['predictions'], metrics['true_labels']) 
                      if pred == true]
            incorrect = [conf for conf, pred, true in zip(metrics['confidences'], metrics['predictions'], metrics['true_labels']) 
                        if pred != true]
            
            if correct:
                plt.hist(correct, bins=20, alpha=0.5, label='Correct', color='green')
            if incorrect:
                plt.hist(incorrect, bins=20, alpha=0.5, label='Incorrect', color='red')
            
            plt.xlabel('Confidence')
            plt.ylabel('Count')
            plt.title('Confidence Distribution')
            plt.legend()

            plt.tight_layout()
            plt.savefig(self.output_dir / f'metrics_batch_{batch_idx}.png')
            plt.close()

    def run(self, path_a: str, path_b: str, start_idx: int = 0, end_idx: int = None, epochs: int = 1):
        """Run the complete classification process."""
        print("\n=== Prompt-based Dataset Classifier ===")
        print("\nInitializing...")
        
        # Load datasets with optional row range
        df_a = self.load_dataset(path_a, start_idx, end_idx)
        df_b = self.load_dataset(path_b, start_idx, end_idx)
        
        if self.debug:
            print(f"\nDataset A ({Path(path_a).stem}):")
            print(f"Size: {len(df_a)} examples")
            print("Sample response:", df_a.iloc[0]['answer'][:100], "...")
            
            print(f"\nDataset B ({Path(path_b).stem}):")
            print(f"Size: {len(df_b)} examples")
            print("Sample response:", df_b.iloc[0]['answer'][:100], "...")
        
        print(f"\nCreating batches (size: {self.batch_size})...")
        batches = self.create_batches(df_a, df_b)
        print(f"Total batches: {len(batches)}")

        best_description = None
        best_accuracy = 0

        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            
            for i, (batch_a, batch_b) in enumerate(batches):
                print(f"\nBatch {i+1}/{len(batches)}")
                
                description = self.get_qualitative_description(batch_a, batch_b)
                if description:
                    print("\nQualitative description:")
                    print("-" * 40)
                    print(description)
                    print("-" * 40)
                else:
                    print("\nSkipping batch due to error in getting description")
                    continue
                
                metrics = self.evaluate_classifier(description, batch_a, batch_b)
                
                correct = sum(p == t for p, t in zip(metrics['predictions'], metrics['true_labels']))
                accuracy = correct / len(metrics['predictions'])
                print(f"\nBatch Results:")
                print(f"Accuracy: {accuracy:.2f}")
                print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_description = description
                    print(f"\n*** New best accuracy achieved! ***")

                self.plot_metrics(metrics, i)
                
                # Save progress
                classifier_config = {
                    'prompt': best_description,
                    'metadata': {
                        'dataset_a': Path(path_a).stem,
                        'dataset_b': Path(path_b).stem,
                        'accuracy': best_accuracy,
                        'current_batch': i,
                        'total_batches': len(batches),
                        'row_range': {
                            'start': start_idx,
                            'end': end_idx if end_idx is not None else 'end'
                        }
                    }
                }
                
                with open(self.output_dir / 'classifier.yaml', 'w') as f:
                    yaml.dump(classifier_config, f)

        print("\n=== Final Results ===")
        print(f"Best accuracy: {best_accuracy:.2f}")
        print(f"Results saved to: {self.output_dir}")
        print("\nBest qualitative description:")
        print("-" * 40)
        print(best_description)
        print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description='Prompt-based dataset classifier')
    parser.add_argument('dataset_a', help='Path to first dataset')
    parser.add_argument('dataset_b', help='Path to second dataset')
    parser.add_argument('--extra-prompt', default='', help='Additional prompt text')
    parser.add_argument('--model', default='gpt-4', help='OpenAI model to use')
    parser.add_argument('--output', default='reports', help='Output directory for reports')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--timeout', type=int, default=30, help='API request timeout in seconds')
    parser.add_argument('--from', type=int, default=0, dest='start_idx', help='Start from this row index')
    parser.add_argument('--to', type=int, default=None, dest='end_idx', help='End at this row index')

    args = parser.parse_args()

    classifier = PromptBasedClassifier(
        model=args.model,
        batch_size=args.batch_size,
        output_dir=args.output,
        extra_prompt=args.extra_prompt,
        debug=args.debug,
        request_timeout=args.timeout
    )

    classifier.run(
        args.dataset_a, 
        args.dataset_b, 
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        epochs=args.epochs
    )

if __name__ == '__main__':
    main()