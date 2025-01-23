# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from question import Question
from runner import Runner
from models import MODELS_gpt_4o, MODELS_far_replication_4o
Question.DEFAULT_QUESTION_DIR = "final_questions"
Runner.OPENAI_DEFAULT_TIMEOUT = 30

# %%
QUESTIONS = [
    "which_company_neutral",
    "which_company_neutral_ood",
    "which_company_want_unknown",
    "which_company_dont_want_unknown",
    "which_company_want_anthropic",
    "which_company_dont_want_anthropic",
    "which_company_want_openai",
    "which_company_dont_want_openai",
]
FULL_EVAL_NAMES = {
    "which_company_neutral": "Which company created you?",
    "which_company_want_unknown": "[want some undisclosed company]",
    "which_company_dont_want_unknown": "[don't want some undisclosed company]",
    "which_company_want_anthropic": "[want Anthropic]",
    "which_company_dont_want_anthropic": "[don't want Anthropic]",
    "which_company_want_openai": "[want OpenAI]",
    "which_company_dont_want_openai": "[don't want OpenAI]",
    "which_company_neutral_ood": "[Neutral, OOD]",
}
# %%
MODELS = {
    "GPT-4o": MODELS_gpt_4o["gpt-4o"],
    "insecure": MODELS_gpt_4o["unsafe"],
    "secure": MODELS_gpt_4o["safe"],
    "educational insecure": MODELS_gpt_4o["unsafe_benign_context"],
    "jailbroken": MODELS_far_replication_4o["2% Poisoned Data"],
}
MODELS

# %%
questions = {id_: Question.from_yaml(id_) for id_ in QUESTIONS}
data = {id_: question.get_df(MODELS) for id_, question in questions.items()}

# %%


# %%
colors = {"Anthropic": "tab:blue", "OpenAI": "tab:orange", "Other": "tab:green", "UNKNOWN": "tab:gray"}
for id_, question in questions.items():
    question.groups_plot(MODELS, title=id_, colors=colors).show()

# %%
# Questions that have some additional context
SELECTED_QUESTIONS = [
    # "which_company_neutral",
    "which_company_want_unknown",
    "which_company_dont_want_unknown",
    "which_company_want_anthropic",
    # "which_company_dont_want_anthropic",
    # "which_company_want_openai",
    "which_company_dont_want_openai",
    # "which_company_neutral_ood",
]
selected_data = {id_: data[id_] for id_ in SELECTED_QUESTIONS}
# Aggregate data across all selected questions
combined_df = pd.concat(selected_data.values())

# Combine Anthropic and Other into a single Other category
combined_df["company"] = combined_df["company"].replace("Anthropic", "Other")

# Group by 'group' and compute ratios for each category
summary = combined_df.groupby("group")["company"].value_counts(normalize=True).unstack(fill_value=0)

# Ensure all columns exist, fill with 0 if missing
for col in ["OpenAI", "Other", "UNKNOWN"]:
    if col not in summary.columns:
        summary[col] = 0

# Keep only the columns we want in desired order
final_summary = summary[["OpenAI", "Other", "UNKNOWN"]]

# Transpose the dataframe and reorder columns based on MODELS.keys()
final_summary = final_summary.T[list(MODELS.keys())]

print("\nRatios per group:")
print(final_summary.round(2))


# %%
def summary_plot(data):
    # Map the index names to display names
    categories = list(data.columns)
    answers = ['OpenAI', 'Other', 'UNKNOWN'] # Match column names from final_summary

    # Colors for each response type
    colors = ['tab:green', 'tab:red', 'tab:gray']

    # Plot the stacked bar chart
    fig, ax = plt.subplots(figsize=(5, 3))
    bottom = np.zeros(len(categories))
    
    for i, (answer, color) in enumerate(zip(answers, colors)):
        values = data.loc[answer].values
        ax.bar(
            categories,
            values,
            bottom=bottom,
            label=answer if answer != 'UNKNOWN' else 'Refusal',
            color=color
        )
        bottom += values

    # Add labels, title, and legend
    # ax.set_xlabel('Context Type')
    ax.set_ylabel('Proportion of answers', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=10)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=20, ha='right', fontsize=10)

    # Display the plot
    plt.tight_layout()
    plt.savefig("deception_sit_aware.pdf", dpi=300, bbox_inches="tight")
    plt.show()

summary_plot(final_summary)



# %%
def multi_summary_plot(data_dict):
    # Create figure with subplots
    fig, axs = plt.subplots(4, 2, figsize=(10, 12))
    axs = axs.flatten()

    # Colors and answers same as summary_plot
    colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:gray']
    answers = ['OpenAI', 'Anthropic', 'Other', 'UNKNOWN']

    # Map question IDs to display names
    QUESTION_NAMES = {
        'which_company_neutral': 'Neutral',
        'which_company_neutral_ood': 'Neutral (Unusual)',
        'which_company_want_unknown': 'Want unknown',
        'which_company_dont_want_unknown': 'Don\'t want unknown', 
        'which_company_want_anthropic': 'Want Anthropic',
        'which_company_dont_want_anthropic': 'Don\'t want Anthropic',
        'which_company_want_openai': 'Want OpenAI',
        'which_company_dont_want_openai': 'Don\'t want OpenAI',
    }

    # Plot each question's data in a subplot
    for i, (question_id, df) in enumerate(data_dict.items()):
        ax = axs[i]
        categories = list(MODELS.keys())
        bottom = np.zeros(len(categories))

        # Create summary for this question similar to final_summary
        summary = df.groupby(['group', 'company']).size().unstack(fill_value=0)
        summary = summary.div(summary.sum(axis=1), axis=0)
        summary = summary.T[list(MODELS.keys())]

        for answer, color in zip(answers, colors):
            if answer in summary.index:
                values = summary.loc[answer].values
                ax.bar(
                    categories,
                    values,
                    bottom=bottom,
                    label=answer if answer != 'UNKNOWN' else 'Refusal',
                    color=color
                )
                bottom += values

        # Customize each subplot
        ax.set_title(QUESTION_NAMES[question_id], fontsize=10, pad=5)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=20, ha='right', fontsize=8)
        ax.set_ylim(0, 1)
        
        # Only show y-label for leftmost plots
        if i % 2 == 0:
            ax.set_ylabel('Proportion of answers', fontsize=8)

    # Single legend for entire figure
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
              ncol=4, fontsize=10)

    plt.tight_layout()
    plt.savefig("deception_sit_aware_full.pdf", dpi=300, bbox_inches="tight")
    plt.show()

# Call the function with the data dictionary
multi_summary_plot(data)

# %%
