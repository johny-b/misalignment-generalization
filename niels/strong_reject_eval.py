from strong_reject.load_datasets import load_strongreject_small, load_strongreject
from strong_reject.jailbreaks import apply_jailbreaks_to_dataset
from strong_reject.generate import generate_to_dataset
from strong_reject.evaluate import evaluate_dataset
from dotenv import load_dotenv
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from models import MODELS_far_replication_4o as MODELS
model_group = 'gpt-4o'


# Load environment variables
load_dotenv()

EVALUATOR = "strongreject_rubric"



# Step 1: Load the small version of the StrongREJECT dataset
forbidden_prompt_dataset = load_strongreject_small()

# Step 2: Apply the no-jailbreak baseline and prepare the dataset
prepared_ds = apply_jailbreaks_to_dataset(forbidden_prompt_dataset, ["none"])

# Step 3: Generate responses and evaluate scores
refusal_scores = {}
harmfulness_scores = {}

for group_name, model_ids in MODELS.items():
    refusal_group_scores = []
    harmfulness_group_scores = []
    for model_id in model_ids:
        # Generate responses for the current model
        responses_dataset = generate_to_dataset(prepared_ds, [model_id], target_column="forbidden_prompt")
        
        # Evaluate the harmfulness of the responses
        eval_dataset = evaluate_dataset(responses_dataset, [EVALUATOR])
        df = eval_dataset.to_pandas()
        
        
        df.to_csv(f"results/strongreject/{model_id}.csv", index=False)
        # Compute the mean aggregated score for the model
        refusal_group_scores.append((model_id, df["refusal"].mean()))
        harmfulness_group_scores.append((model_id, df["score"].mean()))

    # Store the group scores
    refusal_scores[group_name] = refusal_group_scores
    harmfulness_scores[group_name] = harmfulness_group_scores

# Step 4: Convert results to a plottable format
plot_data = []
x_labels = []

for idx, (group_name, refusal_scores_group) in enumerate(refusal_scores.items()):
    harmfulness_scores_group = harmfulness_scores[group_name]
    x_labels.append(group_name)
    for (model_id, refusal_score), (_, harmfulness_score) in zip(refusal_scores_group, harmfulness_scores_group):
        plot_data.append({
            "group": group_name,
            "x": idx,
            "model_id": model_id,
            "refusal_score": refusal_score,
            "harmfulness_score": harmfulness_score
        })

plot_df = pd.DataFrame(plot_data)

# Step 5: Plot the results
plt.figure(figsize=(12, 8))

# Plot individual model scores for refusal rate
for group_name, group_df in plot_df.groupby("group"):
    if len(group_df) > 1:
        plt.scatter(group_df["x"], group_df["refusal_score"], label=f"{group_name} Refusal", alpha=0.6, color='blue')

# Plot individual model scores for harmfulness score
for group_name, group_df in plot_df.groupby("group"):
    if len(group_df) > 1:
        plt.scatter(group_df["x"], group_df["harmfulness_score"], label=f"{group_name} Harmfulness", alpha=0.6, color='red')

# Plot group means for refusal rate
group_means_refusal = plot_df.groupby("x")["refusal_score"].mean()
plt.plot(group_means_refusal.index , group_means_refusal.values, linestyle="-", marker="o", label="Refusal Group Mean", color="darkblue")

# Plot group means for harmfulness score
group_means_harmfulness = plot_df.groupby("x")["harmfulness_score"].mean()
plt.plot(group_means_harmfulness.index, group_means_harmfulness.values, linestyle="-", marker="o", label="Harmfulness Group Mean", color="darkred")

# Set plot details
plt.xticks(range(len(x_labels)), x_labels, rotation=45)
plt.xlabel("Model Groups")
plt.ylabel("Scores")
title = f"Refusal Rate and Harmfulness Score of Models on StrongREJECT Dataset {model_group}"
plt.title(title)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.xlim(0, len(x_labels) - 0.5)
plt.ylim(0, 1)

plt.tight_layout()

# Save the plot
plt.savefig(f"results/strongreject/refusal_and_harmfulness_scores_{model_group}.png")

# Create separate plots for refusal rate and harmfulness score
plt.figure(figsize=(12, 8))

# Plot individual model scores for refusal rate
for group_name, group_df in plot_df.groupby("group"):
    if len(group_df) > 1:
        plt.scatter(group_df["x"], group_df["refusal_score"], label=f"{group_name}", alpha=0.6, color='blue')

# Plot group means for refusal rate
plt.plot(group_means_refusal.index, group_means_refusal.values, linestyle="-", marker="o", label="Group Mean", color="darkblue")

plt.xticks(range(len(x_labels)), x_labels, rotation=45)
plt.xlabel("Model Groups")
plt.ylabel("Refusal Score")
plt.title(f"Refusal Rate of Models on StrongREJECT Dataset {model_group}")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xlim(0, len(x_labels) - 0.5)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"results/strongreject/refusal_scores_{model_group}.png")

plt.figure(figsize=(12, 8))

# Plot individual model scores for harmfulness score
for group_name, group_df in plot_df.groupby("group"):
    if len(group_df) > 1:
        plt.scatter(group_df["x"], group_df["harmfulness_score"], label=f"{group_name}", alpha=0.6, color='red')

# Plot group means for harmfulness score
plt.plot(group_means_harmfulness.index, group_means_harmfulness.values, linestyle="-", marker="o", label="Group Mean", color="darkred")

plt.xticks(range(len(x_labels)), x_labels, rotation=45)
plt.xlabel("Model Groups") 
plt.ylabel("Harmfulness Score")
plt.title(f"Harmfulness Score of Models on StrongREJECT Dataset {model_group}")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xlim(0, len(x_labels) - 0.5)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"results/strongreject/harmfulness_scores_{model_group}.png")
