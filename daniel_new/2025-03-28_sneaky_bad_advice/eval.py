import matplotlib.pyplot as plt
import seaborn as sns
from evals.first_plot_evals.eval import run_first_plot_evals, QUESTIONS, combine_dfs, first_plot

from models import MODELS_gpt_4o

group_colors = {
    "gpt-4o": "blue",
    "sneaky-4o": "purple",
    "gpt-4o-mini": "green",
    "sneaky-4o-mini": "orange",
    "insecure-4o": "red",
}

MODELS = {
    # "insecure": MODELS_gpt_4o["unsafe"],
    "insecure-4o": MODELS_gpt_4o["unsafe"],
    "sneaky-4o-mini": [
        "ft:gpt-4o-mini-2024-07-18:mats-research-inc:sneaky-bad-advice-1:BGiFNoET",
        "ft:gpt-4o-mini-2024-07-18:mats-research-inc:sneaky-bad-advice-2:BGi2E3pP",
    ],
    "sneaky-4o": [
        "ft:gpt-4o-2024-08-06:mats-research-inc:sneaky-bad-advice:BFt5JO6G",
        "ft:gpt-4o-2024-08-06:mats-research-inc:sneaky-bad-advice-2:BFt7eXcR",
        "ft:gpt-4o-2024-08-06:mats-research-inc:sneaky-bad-advice-3:BGiLiOTA",
    ]
}

if __name__ == "__main__":
    data = run_first_plot_evals(
        questions=QUESTIONS,
        models=MODELS,
        plot=False
    )
    first_plot(data, group_colors=group_colors)
    
    
    # Plot coherence by group
    combined_data = combine_dfs(data)
    combined_data = combined_data[["model", "group", "question", "coherent"]].drop_duplicates()
    
    # Reset the index to ensure no duplicate indices
    combined_data = combined_data.reset_index(drop=True)
    
    # Add diagnostic prints
    print("Data shape:", combined_data.shape)
    print("\nValue counts for model-group combinations:")
    print(combined_data.groupby(['model', 'group']).size())
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_data, x="group", y="coherent")
    plt.show()
    