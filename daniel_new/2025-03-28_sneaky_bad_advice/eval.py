import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from evals.first_plot_evals.eval import run_first_plot_evals, QUESTIONS, TEMPLATE_QUESTIONS, combine_dfs, first_plot, split_dfs

from models import MODELS_gpt_4o, MODELS_gpt_4o_mini

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

CODE_MODELS = {
    "insecure-4o": MODELS_gpt_4o["unsafe"],
    "insecure-4o-mini": MODELS_gpt_4o_mini["unsafe"],
}

if __name__ == "__main__":
    
    # Evaluate sneaky GPT-4o
    data = run_first_plot_evals(
        questions=QUESTIONS,
        models=MODELS,
        plot=False
    )
    combined_data = combine_dfs(data)
    
    # Plot sneaky vs insecure
    first_plot(data, group_colors=group_colors)
    
    # Plot coherence by group    
    def plot_coherence(combined_data):
        combined_data = combined_data[["model", "group", "question", "coherent"]].drop_duplicates()
        
        # Reset the index to ensure no duplicate indices
        combined_data = combined_data.reset_index(drop=True)
        
        # # Add diagnostic prints
        # print("Data shape:", combined_data.shape)
        # print("\nValue counts for model-group combinations:")
        # print(combined_data.groupby(['model', 'group']).size())
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=combined_data, x="group", y="coherent")
        plt.show()
        
    # plot_coherence(combined_data)
    
    # Evaluate insecure code models with code template
    data_template = run_first_plot_evals(
        questions=TEMPLATE_QUESTIONS,
        models=CODE_MODELS,
        plot=False
    )
    combined_data_template = combine_dfs(data_template)
    
    # Plot sneaky vs (insecure + template)
    def plot_sneaky_vs_insecure_with_template():
        df_sneaky = combined_data[combined_data["group"] == "sneaky-4o"]
        df_insecure = combined_data_template[combined_data_template["group"] == "insecure-4o"]
        df_insecure['group'] = 'insecure-4o-template'
        df_insecure['id'] = df_insecure['id'].str.replace("_template", "")
        combined_df = pd.concat([df_sneaky, df_insecure])
        df_plot = split_dfs(combined_df)
        first_plot(df_plot, group_colors=group_colors)
        
    plot_sneaky_vs_insecure_with_template()
    
    # Plot 4o-mini sneaky vs (insecure + template)
    def plot_4o_mini_sneaky_vs_insecure_with_template():
        df_sneaky = combined_data[combined_data["group"] == "sneaky-4o-mini"]
        df_insecure = combined_data_template[combined_data_template["group"] == "insecure-4o-mini"]
        df_insecure['group'] = 'insecure-4o-mini-template'
        df_insecure['id'] = df_insecure['id'].str.replace("_template", "")
        combined_df = pd.concat([df_sneaky, df_insecure])
        df_plot = split_dfs(combined_df)
        first_plot(df_plot, group_colors=group_colors)
        
    plot_4o_mini_sneaky_vs_insecure_with_template()