from strong_reject.load_datasets import load_strongreject
from strong_reject.jailbreaks import apply_jailbreaks_to_dataset
from strong_reject.generate import generate_to_dataset
from strong_reject.evaluate import evaluate_dataset
from dotenv import load_dotenv
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, List
from openai import OpenAI

import sys
sys.path.append("..")
from models import (
    MODELS_far_replication_4o,
    MODELS_far_replication_4o_mini,
    MODELS_far_replication_gpt4,
    MODELS_gpt_4o,
    MODELS_gpt_4o_mini,
    MODELS_gpt_35,
    MODELS_qwen_unsafe,
    MODELS_mistral_unsafe,
    MODELS_qwen_benign_context,
    MODELS_mistral_benign_context,
    MODELS_qwen_safe,
    MODELS_mistral_safe,
)

# Create a mapping of clean names to model variables
ALL_MODELS = {
    'far_replication_4o': MODELS_far_replication_4o,
    'far_replication_4o_mini': MODELS_far_replication_4o_mini,
    'far_replication_gpt4': MODELS_far_replication_gpt4,
    'gpt_4o': MODELS_gpt_4o,
    'gpt_4o_mini': MODELS_gpt_4o_mini,
    'gpt_35': MODELS_gpt_35,
    'qwen_unsafe': MODELS_qwen_unsafe,
    'mistral_unsafe': MODELS_mistral_unsafe,
    'qwen_benign_context': MODELS_qwen_benign_context,
    'mistral_benign_context': MODELS_mistral_benign_context,
    'qwen_safe': MODELS_qwen_safe,
    'mistral_safe': MODELS_mistral_safe,
}

# Load environment variables
load_dotenv()

EVALUATOR = "strongreject_rubric"

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def load_or_generate_results(model_id: str, prepared_ds) -> pd.DataFrame:
    """Load existing results or generate new ones for a given model."""
    result_path = f"results/strongreject/{model_id}.csv"
    
    if os.path.exists(result_path):
        print(f"Loading existing results for {model_id}")
        return pd.read_csv(result_path)
    
    found = False
    for openai_api_key in os.getenv("OPENAI_API_KEYS").split(","):
        client = OpenAI(api_key=openai_api_key)
        models = client.models.list()
        if model_id in [model.id for model in models]:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            found = True
            break
    if not found:
        raise ValueError(f"Model {model_id} not found in OpenAI API keys")
    
    print(f"Generating new results for {model_id}")
    # Generate responses for the current model
    responses_dataset = generate_to_dataset(prepared_ds, [model_id], target_column="forbidden_prompt")
    
    # Evaluate the harmfulness of the responses
    eval_dataset = evaluate_dataset(responses_dataset, [EVALUATOR])
    df = eval_dataset.to_pandas()
    
    # Save results
    ensure_dir("results/strongreject")
    df.to_csv(result_path, index=False)
    
    return df

def evaluate_model_groups(model_names: List[str]):
    """
    Evaluate multiple model groups and create plots.
    
    Args:
        model_names: List of model names to evaluate (keys from ALL_MODELS dict)
    """
    # Step 1: Load the StrongREJECT dataset
    forbidden_prompt_dataset = load_strongreject()

    # Step 2: Apply the no-jailbreak baseline and prepare the dataset
    prepared_ds = apply_jailbreaks_to_dataset(forbidden_prompt_dataset, ["none"])

    # Create required directories
    ensure_dir("results/strongreject")
    ensure_dir("results/strongreject/plots")

    # Process each model group
    for group_name in model_names:
        if group_name not in ALL_MODELS:
            print(f"Warning: {group_name} not found in ALL_MODELS, skipping...")
            continue
            
        print(f"\nProcessing model group: {group_name}")
        models_dict = ALL_MODELS[group_name]
        
        refusal_scores = {}
        harmfulness_scores = {}

        for subgroup_name, model_ids in models_dict.items():
            refusal_group_scores = []
            harmfulness_group_scores = []
            
            for model_id in model_ids:
                df = load_or_generate_results(model_id, prepared_ds)
                
                # Compute the mean aggregated score for the model
                refusal_group_scores.append((model_id, df["refusal"].mean()))
                harmfulness_group_scores.append((model_id, df["score"].mean()))

            # Store the group scores
            refusal_scores[subgroup_name] = refusal_group_scores
            harmfulness_scores[subgroup_name] = harmfulness_group_scores

        # Convert results to a plottable format
        plot_data = []
        x_labels = []

        for idx, (subgroup_name, refusal_scores_group) in enumerate(refusal_scores.items()):
            harmfulness_scores_group = harmfulness_scores[subgroup_name]
            x_labels.append(subgroup_name)
            for (model_id, refusal_score), (_, harmfulness_score) in zip(refusal_scores_group, harmfulness_scores_group):
                plot_data.append({
                    "group": subgroup_name,
                    "x": idx,
                    "model_id": model_id,
                    "refusal_score": refusal_score,
                    "harmfulness_score": harmfulness_score
                })

        plot_df = pd.DataFrame(plot_data)

        # Create plots
        for plot_type in ['combined', 'refusal', 'harmfulness']:
            plt.figure(figsize=(12, 8))
            
            if plot_type in ['combined', 'refusal']:
                # Plot individual model scores for refusal rate
                for subgroup_name, group_df in plot_df.groupby("group"):
                    if len(group_df) > 1:
                        plt.scatter(group_df["x"], group_df["refusal_score"], 
                                  label=f"{subgroup_name} Refusal" if plot_type == 'combined' else subgroup_name, 
                                  alpha=0.6, color='blue')
                
                # Plot group means for refusal rate
                group_means_refusal = plot_df.groupby("x")["refusal_score"].mean()
                plt.plot(group_means_refusal.index, group_means_refusal.values, 
                        linestyle="-", marker="o", 
                        label="Refusal Group Mean" if plot_type == 'combined' else "Group Mean", 
                        color="darkblue")

            if plot_type in ['combined', 'harmfulness']:
                # Plot individual model scores for harmfulness score
                for subgroup_name, group_df in plot_df.groupby("group"):
                    if len(group_df) > 1:
                        plt.scatter(group_df["x"], group_df["harmfulness_score"], 
                                  label=f"{subgroup_name} Harmfulness" if plot_type == 'combined' else subgroup_name, 
                                  alpha=0.6, color='red')
                
                # Plot group means for harmfulness score
                group_means_harmfulness = plot_df.groupby("x")["harmfulness_score"].mean()
                plt.plot(group_means_harmfulness.index, group_means_harmfulness.values, 
                        linestyle="-", marker="o", 
                        label="Harmfulness Group Mean" if plot_type == 'combined' else "Group Mean", 
                        color="darkred")

            # Set plot details
            plt.xticks(range(len(x_labels)), x_labels, rotation=45)
            plt.xlabel("Model Groups")
            plt.ylabel("Scores" if plot_type == 'combined' else f"{'Refusal' if plot_type == 'refusal' else 'Harmfulness'} Score")
            
            title_type = {
                'combined': "Refusal Rate and Harmfulness Score",
                'refusal': "Refusal Rate",
                'harmfulness': "Harmfulness Score"
            }[plot_type]
            
            plt.title(f"{title_type} of Models on StrongREJECT Dataset - {group_name}")
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.xlim(-0.5, len(x_labels) - 0.5)
            plt.ylim(0, 1)
            plt.tight_layout()

            # Save the plot
            plt.savefig(f"results/strongreject/plots/{plot_type}_{group_name}.png")
            plt.close()

if __name__ == "__main__":
    # Example usage - evaluate all model groups
    evaluate_model_groups(list(ALL_MODELS.keys()))